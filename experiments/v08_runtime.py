"""JAX runtime for v0.8 IQE quasimetric training and inference.

Kept under ``experiments`` so importing the core graph package does not require
JAX, Flax, or Optax.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
from typing import Any

import numpy as np


DEFAULT_ENCODER_WIDTHS = (1024, 1024, 256)
DEFAULT_PROJECTOR_WIDTHS = (1024, 1024, 2048)
DEFAULT_COMPONENTS = 64
DEFAULT_COMPONENT_DIM = 32


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from flax import serialization
        import optax
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("v0.8 quasimetric training requires JAX, Flax, and Optax") from exc
    return jax, jnp, nn, serialization, optax


def make_quasimetric_model(
    *,
    encoder_widths: tuple[int, ...] = DEFAULT_ENCODER_WIDTHS,
    projector_widths: tuple[int, ...] = DEFAULT_PROJECTOR_WIDTHS,
    n_components: int = DEFAULT_COMPONENTS,
    component_dim: int = DEFAULT_COMPONENT_DIM,
):
    """Create the shared encoder/projector and IQE-maxmean model.

    IQE components are the Lebesgue measures of unions of intervals
    ``[z_x[i], max(z_x[i], z_y[i])]``. Their maxmean reduction is a learned
    convex combination of the component maximum and mean.
    """
    jax, jnp, nn, _, _ = _jax_modules()
    if not encoder_widths or not projector_widths:
        raise ValueError("encoder_widths and projector_widths must be non-empty")
    if any(width < 1 for width in (*encoder_widths, *projector_widths)):
        raise ValueError("all network widths must be positive")
    if n_components < 1 or component_dim < 1:
        raise ValueError("IQE component dimensions must be positive")

    class IQEQuasimetric(nn.Module):
        encoder_widths: tuple[int, ...]
        projector_widths: tuple[int, ...]
        n_components: int
        component_dim: int

        @nn.compact
        def __call__(self, source, destination):
            if source.shape != destination.shape or source.ndim != 2:
                raise ValueError("quasimetric inputs must be aligned rank-2 arrays")
            joined = jnp.concatenate((source, destination), axis=0)
            for layer, width in enumerate(self.encoder_widths[:-1]):
                joined = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(),
                    name=f"encoder_{layer}",
                )(joined)
                joined = nn.relu(joined)
            joined = nn.Dense(
                self.encoder_widths[-1],
                kernel_init=nn.initializers.xavier_uniform(),
                name="encoder_out",
            )(joined)
            for layer, width in enumerate(self.projector_widths[:-1]):
                joined = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(),
                    name=f"projector_{layer}",
                )(joined)
                joined = nn.relu(joined)
            joined = nn.Dense(
                self.projector_widths[-1],
                kernel_init=nn.initializers.xavier_uniform(),
                name="projector_out",
            )(joined)

            latent_dim = self.n_components * self.component_dim
            if joined.shape[-1] != latent_dim:
                raise ValueError(
                    "projector output width must equal n_components * component_dim"
                )
            batch = source.shape[0]
            latent_source = joined[:batch].reshape(
                batch, self.n_components, self.component_dim
            )
            latent_destination = joined[batch:].reshape(
                batch, self.n_components, self.component_dim
            )
            component_distance = _iqe_union_distance(
                latent_source, latent_destination, jnp
            )
            raw_alpha = self.param(
                "raw_alpha", nn.initializers.constant(-1.0), ()
            )
            alpha = jax.nn.sigmoid(raw_alpha)
            return alpha * jnp.max(component_distance, axis=-1) + (
                1.0 - alpha
            ) * jnp.mean(component_distance, axis=-1)

    # The nested module needs no closure over JAX; this makes its checkpoint
    # structure stable across the training and graph-build entry points.
    return IQEQuasimetric(
        encoder_widths=tuple(map(int, encoder_widths)),
        projector_widths=tuple(map(int, projector_widths)),
        n_components=int(n_components), component_dim=int(component_dim),
    )


def _iqe_union_distance(source, destination, jnp):
    """Return one interval-union measure for every IQE component."""
    order = jnp.argsort(source, axis=-1, stable=True)
    starts = jnp.take_along_axis(source, order, axis=-1)
    ends = jnp.maximum(starts, jnp.take_along_axis(destination, order, axis=-1))
    import jax
    prefix_max = jax.lax.associative_scan(jnp.maximum, ends, axis=-1)
    prior_max = jnp.concatenate(
        (jnp.full_like(ends[..., :1], -jnp.inf), prefix_max[..., :-1]), axis=-1
    )
    uncovered = jnp.maximum(starts, prior_max)
    return jnp.sum(jnp.maximum(ends - uncovered, 0.0), axis=-1)


def qrl_loss_components(distance_goal, distance_step, lambda_value, *, epsilon: float = 0.25):
    """QRL primal objective and one-step constraint diagnostics."""
    jax, jnp, _, _, _ = _jax_modules()
    # -E[phi(d)] where phi(d) = -softplus(beta * (500 - d)) / beta.
    distance_maximization_loss = jnp.mean(
        jax.nn.softplus((500.0 - distance_goal) * 0.01) / 0.01
    )
    one_step_violation = jnp.mean(jnp.square(jax.nn.relu(distance_step - 1.0)))
    constraint_gap = one_step_violation - epsilon**2
    total = distance_maximization_loss + lambda_value * constraint_gap
    return total, distance_maximization_loss, one_step_violation, constraint_gap


def make_train_step(model, *, batch_size: int = 4096, epsilon: float = 0.25,
                    learning_rate: float = 5e-4, lambda_learning_rate: float = 1e-2):
    """Create the compiled primal/dual QRL update for one training batch."""
    jax, jnp, _, _, optax = _jax_modules()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    primal_optimizer = optax.adam(learning_rate)
    dual_optimizer = optax.adam(lambda_learning_rate)

    @jax.jit
    def update(params, primal_state, raw_lambda, dual_state, key,
               states, next_state_index, train_rows):
        key, state_key, goal_key = jax.random.split(key, 3)
        sample_positions = jax.random.randint(
            state_key, (batch_size,), 0, train_rows.shape[0]
        )
        goal_positions = jax.random.randint(
            goal_key, (batch_size,), 0, train_rows.shape[0]
        )
        rows = train_rows[sample_positions]
        goal_rows = train_rows[goal_positions]
        current = states[rows]
        one_step_next = states[next_state_index[rows]]
        independent_goal = states[next_state_index[goal_rows]]
        left = jnp.concatenate((current, current), axis=0)
        right = jnp.concatenate((independent_goal, one_step_next), axis=0)
        dual_value = jax.lax.stop_gradient(jax.nn.softplus(raw_lambda))

        def primal_loss(parameters):
            distances = model.apply({"params": parameters}, left, right)
            distance_goal = distances[:batch_size]
            distance_step = distances[batch_size:]
            total, distance_loss, violation, constraint_gap = qrl_loss_components(
                distance_goal, distance_step, dual_value, epsilon=epsilon
            )
            return total, (
                distance_goal, distance_step, distance_loss, violation, constraint_gap
            )

        (total, (distance_goal, distance_step, distance_loss,
                 violation, constraint_gap)), grads = jax.value_and_grad(
            primal_loss, has_aux=True
        )(params)
        updates, primal_state = primal_optimizer.update(grads, primal_state, params)
        params = optax.apply_updates(params, updates)

        def dual_loss_fn(raw):
            lam = jax.nn.softplus(raw)
            return -lam * jax.lax.stop_gradient(constraint_gap)

        dual_grad = jax.grad(dual_loss_fn)(raw_lambda)
        dual_updates, dual_state = dual_optimizer.update(
            dual_grad, dual_state, raw_lambda
        )
        raw_lambda = optax.apply_updates(raw_lambda, dual_updates)
        metrics = {
            "loss": total,
            "distance_maximization_loss": distance_loss,
            "one_step_violation_squared": violation,
            "constraint_gap": constraint_gap,
            "lambda": jax.nn.softplus(raw_lambda),
            "distance_goal_mean": jnp.mean(distance_goal),
            "distance_step_mean": jnp.mean(distance_step),
        }
        return params, primal_state, raw_lambda, dual_state, key, metrics

    return update, primal_optimizer, dual_optimizer


def initialize_train_state(model, *, seed: int, input_dim: int,
                           primal_optimizer, dual_optimizer):
    jax, jnp, _, _, _ = _jax_modules()
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = model.init(
        init_key, jnp.zeros((2, input_dim), dtype=jnp.float32),
        jnp.zeros((2, input_dim), dtype=jnp.float32),
    )["params"]
    primal_state = primal_optimizer.init(params)
    # Inverse softplus so the requested initial positive dual variable is 0.01.
    initial_lambda = 0.01
    raw_lambda = jnp.asarray(np.log(np.expm1(initial_lambda)), dtype=jnp.float32)
    dual_state = dual_optimizer.init(raw_lambda)
    return params, primal_state, raw_lambda, dual_state, key


def make_distance_predictor(model, params, *, batch_size: int = 8192):
    """Build a fixed-shape JIT forward function and NumPy batch adapter."""
    jax, jnp, _, _, _ = _jax_modules()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    @jax.jit
    def predict_jit(source, destination):
        return model.apply({"params": params}, source, destination)

    def predict(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
        src = np.asarray(source, dtype=np.float32)
        dst = np.asarray(destination, dtype=np.float32)
        if src.ndim != 2 or src.shape != dst.shape:
            raise ValueError("source and destination must be aligned rank-2 arrays")
        out = np.empty(len(src), dtype=np.float32)
        for start in range(0, len(src), batch_size):
            end = min(start + batch_size, len(src))
            current_n = end - start
            source_batch = np.zeros((batch_size, src.shape[1]), dtype=np.float32)
            destination_batch = np.zeros_like(source_batch)
            source_batch[:current_n] = src[start:end]
            destination_batch[:current_n] = dst[start:end]
            values = np.asarray(predict_jit(jnp.asarray(source_batch), jnp.asarray(destination_batch)))
            out[start:end] = values[:current_n]
        return out

    return predict


def next_state_indices(terminal: np.ndarray) -> np.ndarray:
    """Map every transition row to its aligned next-state table index."""
    terminal = np.asarray(terminal, dtype=np.bool_)
    n = len(terminal)
    terminal_rows = np.flatnonzero(terminal)
    result = np.empty(n, dtype=np.int64)
    nonterminal_rows = np.flatnonzero(~terminal)
    result[nonterminal_rows] = nonterminal_rows + 1
    result[terminal_rows] = n + np.arange(len(terminal_rows), dtype=np.int64)
    return result


def provenance(root: str | Path) -> dict[str, object]:
    """Return the source commit and tracked/untracked worktree cleanliness."""
    root = Path(root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all",
         "--ignore-submodules=all"], cwd=root,
        check=True, text=True, capture_output=True,
    ).stdout
    qc_gitlink = subprocess.run(
        ["git", "ls-tree", "HEAD", "third_party/qc"], cwd=root,
        check=True, text=True, capture_output=True,
    ).stdout.strip().split()
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "qc_submodule_commit": qc_gitlink[2] if len(qc_gitlink) >= 3 else None,
    }


def save_quasimetric_checkpoint(
    directory: str | Path,
    params: Any,
    metadata: dict[str, Any],
    *,
    train_state: Any | None = None,
) -> Path:
    _, _, _, serialization, _ = _jax_modules()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    params_bytes = serialization.to_bytes(params)
    (path / "params.msgpack").write_bytes(params_bytes)
    record = dict(metadata)
    record["params_sha256"] = hashlib.sha256(params_bytes).hexdigest()
    (path / "metadata.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if train_state is not None:
        (path / "train_state.msgpack").write_bytes(serialization.to_bytes(train_state))
    return path


def load_quasimetric_checkpoint(
    directory: str | Path,
    *,
    input_dim: int,
    expected_dataset_id: str,
):
    _, jnp, _, serialization, _ = _jax_modules()
    path = Path(directory)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != expected_dataset_id:
        raise ValueError(
            "quasimetric checkpoint dataset ID does not match graph-build data: "
            f"checkpoint={metadata.get('dataset_id')!r}, expected={expected_dataset_id!r}"
        )
    if int(metadata.get("input_dim", -1)) != input_dim:
        raise ValueError("quasimetric checkpoint input dimension does not match graph data")
    model = make_quasimetric_model(
        encoder_widths=tuple(metadata["encoder_widths"]),
        projector_widths=tuple(metadata["projector_widths"]),
        n_components=int(metadata["n_components"]),
        component_dim=int(metadata["component_dim"]),
    )
    template = model.init(
        __import__("jax").random.PRNGKey(0),
        jnp.zeros((1, input_dim), dtype=jnp.float32),
        jnp.zeros((1, input_dim), dtype=jnp.float32),
    )["params"]
    params_bytes = (path / "params.msgpack").read_bytes()
    digest = hashlib.sha256(params_bytes).hexdigest()
    if metadata.get("params_sha256") != digest:
        raise ValueError("quasimetric checkpoint params SHA256 does not match metadata")
    params = serialization.from_bytes(template, params_bytes)
    return model, params, metadata
