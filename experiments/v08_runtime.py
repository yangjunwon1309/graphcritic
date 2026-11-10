"""JAX runtime for v0.8 IQE quasimetric training and inference.

Kept under ``experiments`` so importing the core graph package does not require
JAX, Flax, or Optax.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
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
    activation: str = "gelu",
    layernorm: bool = True,
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
    if activation not in {"relu", "gelu"}:
        raise ValueError("activation must be 'relu' or 'gelu'")

    class IQEQuasimetric(nn.Module):
        encoder_widths: tuple[int, ...]
        projector_widths: tuple[int, ...]
        n_components: int
        component_dim: int
        activation: str
        layernorm: bool

        @nn.compact
        def __call__(self, source, destination, *, return_diagnostics=False):
            if source.shape != destination.shape or source.ndim != 2:
                raise ValueError("quasimetric inputs must be aligned rank-2 arrays")
            joined = jnp.concatenate((source, destination), axis=0)
            preactivations = {}
            for layer, width in enumerate(self.encoder_widths[:-1]):
                name = f"encoder_{layer}"
                joined = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(),
                    name=name,
                )(joined)
                if self.layernorm:
                    joined = nn.LayerNorm(
                        scale_init=nn.initializers.ones,
                        bias_init=nn.initializers.zeros,
                        name=f"{name}_norm",
                    )(joined)
                if return_diagnostics:
                    preactivations[name] = joined
                joined = nn.relu(joined) if self.activation == "relu" else jax.nn.gelu(joined)
            joined = nn.Dense(
                self.encoder_widths[-1],
                kernel_init=nn.initializers.xavier_uniform(),
                name="encoder_out",
            )(joined)
            encoder_embedding = joined
            for layer, width in enumerate(self.projector_widths[:-1]):
                name = f"projector_{layer}"
                joined = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(),
                    name=name,
                )(joined)
                if self.layernorm:
                    joined = nn.LayerNorm(
                        scale_init=nn.initializers.ones,
                        bias_init=nn.initializers.zeros,
                        name=f"{name}_norm",
                    )(joined)
                if return_diagnostics:
                    preactivations[name] = joined
                joined = nn.relu(joined) if self.activation == "relu" else jax.nn.gelu(joined)
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
            raw_alpha = self.param(
                "raw_alpha", nn.initializers.constant(-1.0), ()
            )
            distance = iqe_distance_from_embeddings(
                latent_source, latent_destination, raw_alpha,
                n_components=self.n_components, component_dim=self.component_dim,
                jnp=jnp,
            )
            if return_diagnostics:
                return {
                    "distance": distance,
                    "preactivations": preactivations,
                    "encoder_embedding": encoder_embedding[:batch],
                    "projector_embedding": joined[:batch],
                }
            return distance

    # The nested module needs no closure over JAX; this makes its checkpoint
    # structure stable across the training and graph-build entry points.
    return IQEQuasimetric(
        encoder_widths=tuple(map(int, encoder_widths)),
        projector_widths=tuple(map(int, projector_widths)),
        n_components=int(n_components), component_dim=int(component_dim),
        activation=activation, layernorm=bool(layernorm),
    )


def make_quasimetric_embedder(model):
    """Create a backbone-only module with the model's unchanged parameter names."""
    jax, _, nn, _, _ = _jax_modules()

    class IQEEmbedder(nn.Module):
        encoder_widths: tuple[int, ...]
        projector_widths: tuple[int, ...]
        n_components: int
        component_dim: int
        activation: str
        layernorm: bool

        @nn.compact
        def __call__(self, inputs):
            if inputs.ndim != 2:
                raise ValueError("quasimetric inputs must be rank-2 arrays")
            values = inputs
            for layer, width in enumerate(self.encoder_widths[:-1]):
                name = f"encoder_{layer}"
                values = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(), name=name
                )(values)
                if self.layernorm:
                    values = nn.LayerNorm(
                        scale_init=nn.initializers.ones,
                        bias_init=nn.initializers.zeros,
                        name=f"{name}_norm",
                    )(values)
                values = nn.relu(values) if self.activation == "relu" else jax.nn.gelu(values)
            values = nn.Dense(
                self.encoder_widths[-1], kernel_init=nn.initializers.xavier_uniform(),
                name="encoder_out",
            )(values)
            for layer, width in enumerate(self.projector_widths[:-1]):
                name = f"projector_{layer}"
                values = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(), name=name
                )(values)
                if self.layernorm:
                    values = nn.LayerNorm(
                        scale_init=nn.initializers.ones,
                        bias_init=nn.initializers.zeros,
                        name=f"{name}_norm",
                    )(values)
                values = nn.relu(values) if self.activation == "relu" else jax.nn.gelu(values)
            values = nn.Dense(
                self.projector_widths[-1],
                kernel_init=nn.initializers.xavier_uniform(),
                name="projector_out",
            )(values)
            expected = self.n_components * self.component_dim
            if values.shape[-1] != expected:
                raise ValueError("projector output width does not match the IQE dimensions")
            return values

    return IQEEmbedder(
        encoder_widths=tuple(model.encoder_widths),
        projector_widths=tuple(model.projector_widths),
        n_components=int(model.n_components),
        component_dim=int(model.component_dim),
        activation=str(model.activation),
        layernorm=bool(model.layernorm),
    )


def iqe_distance_from_embeddings(source, destination, raw_alpha, *, n_components: int,
                                 component_dim: int, jnp):
    """Evaluate the checkpoint's IQE-maxmean head on flattened or shaped embeddings."""
    if source.ndim == 2:
        source = source.reshape((-1, n_components, component_dim))
    if destination.ndim == 2:
        destination = destination.reshape((-1, n_components, component_dim))
    if source.shape != destination.shape or source.ndim != 3:
        raise ValueError("IQE embeddings must be aligned (N,C,D) or flattened arrays")
    component_distance = _iqe_union_distance(source, destination, jnp)
    import jax
    alpha = jax.nn.sigmoid(raw_alpha)
    return alpha * jnp.max(component_distance, axis=-1) + (
        1.0 - alpha
    ) * jnp.mean(component_distance, axis=-1)


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


def qrl_loss_components(
    distance_goal, distance_step, lambda_value, *,
    epsilon: float = 0.25, constraint_horizon: float = 1.0,
):
    """QRL objective with a configurable frame-step constraint threshold."""
    jax, jnp, _, _, _ = _jax_modules()
    # -E[phi(d)] where phi(d) = -softplus(beta * (500 - d)) / beta.
    distance_maximization_loss = jnp.mean(
        jax.nn.softplus((500.0 - distance_goal) * 0.01) / 0.01
    )
    # ``constraint_horizon`` may be a scalar horizon (the original v0.8
    # trainer) or one per sampled chunk (the v0.8.2 object event/motion cost).
    threshold = jnp.asarray(constraint_horizon, dtype=distance_step.dtype)
    violation = jnp.mean(jnp.square(jax.nn.relu(distance_step - threshold)))
    constraint_gap = violation - epsilon**2
    total = distance_maximization_loss + lambda_value * constraint_gap
    return total, distance_maximization_loss, violation, constraint_gap


def projected_dual_sgd(raw_lambda, constraint_gap, learning_rate, jnp):
    """Apply λ ← max(0, λ + η·gap) in direct, projected-lambda space."""
    return jnp.maximum(0.0, raw_lambda + learning_rate * constraint_gap)


def make_train_step(model, *, batch_size: int = 4096, epsilon: float = 0.25,
                    learning_rate: float = 5e-4, lambda_learning_rate: float = 1e-2,
                    grad_clip: float = 1.0, dual: str = "adam",
                    constraint_horizon: float = 1.0,
                    independent_goal_is_next_state: bool = True):
    """Create the compiled primal/dual QRL update for one training batch."""
    jax, jnp, _, _, optax = _jax_modules()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")
    if dual not in {"adam", "sgd"}:
        raise ValueError("dual must be 'adam' or 'sgd'")
    primal_optimizer = (
        optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(learning_rate))
        if grad_clip > 0 else optax.adam(learning_rate)
    )
    dual_optimizer = optax.adam(lambda_learning_rate) if dual == "adam" else None

    @jax.jit
    def update(params, primal_state, raw_lambda, dual_state, key,
               states, next_state_index, train_rows, constraint_rows=None,
               constraint_next_index=None, constraint_cost_values=None):
        key, state_key, goal_key, constraint_key = jax.random.split(key, 4)
        sample_positions = jax.random.randint(
            state_key, (batch_size,), 0, train_rows.shape[0]
        )
        goal_positions = jax.random.randint(
            goal_key, (batch_size,), 0, train_rows.shape[0]
        )
        rows = train_rows[sample_positions]
        goal_rows = train_rows[goal_positions]
        valid_constraint_rows = train_rows if constraint_rows is None else constraint_rows
        constraint_positions = jax.random.randint(
            constraint_key, (batch_size,), 0, valid_constraint_rows.shape[0]
        )
        constraint_source_rows = valid_constraint_rows[constraint_positions]
        current = states[rows]
        constraint_current = states[constraint_source_rows]
        constraint_mapping = (
            next_state_index if constraint_next_index is None else constraint_next_index
        )
        constraint_next = states[constraint_mapping[constraint_source_rows]]
        independent_goal = (
            states[next_state_index[goal_rows]]
            if independent_goal_is_next_state else states[goal_rows]
        )
        constraint_threshold = (
            constraint_horizon if constraint_cost_values is None
            else constraint_cost_values[constraint_source_rows]
        )
        left = jnp.concatenate((current, constraint_current), axis=0)
        right = jnp.concatenate((independent_goal, constraint_next), axis=0)
        dual_value = jax.lax.stop_gradient(
            jax.nn.softplus(raw_lambda) if dual == "adam" else raw_lambda
        )

        def primal_loss(parameters):
            distances = model.apply({"params": parameters}, left, right)
            distance_goal = distances[:batch_size]
            distance_step = distances[batch_size:]
            total, distance_loss, violation, constraint_gap = qrl_loss_components(
                distance_goal, distance_step, dual_value, epsilon=epsilon,
                constraint_horizon=constraint_threshold,
            )
            return total, (
                distance_goal, distance_step, distance_loss, violation, constraint_gap
            )

        (total, (distance_goal, distance_step, distance_loss,
                 violation, constraint_gap)), grads = jax.value_and_grad(
            primal_loss, has_aux=True
        )(params)
        gradient_norm_preclip = optax.global_norm(grads)
        updates, primal_state = primal_optimizer.update(grads, primal_state, params)
        params = optax.apply_updates(params, updates)

        if dual == "adam":
            def dual_loss_fn(raw):
                lam = jax.nn.softplus(raw)
                return -lam * jax.lax.stop_gradient(constraint_gap)

            dual_grad = jax.grad(dual_loss_fn)(raw_lambda)
            dual_updates, dual_state = dual_optimizer.update(
                dual_grad, dual_state, raw_lambda
            )
            raw_lambda = optax.apply_updates(raw_lambda, dual_updates)
        else:
            raw_lambda = projected_dual_sgd(
                raw_lambda, constraint_gap, lambda_learning_rate, jnp
            )
        metrics = {
            "loss": total,
            "distance_maximization_loss": distance_loss,
            # Keep the historic key for A1/A2 log compatibility; for windowed
            # runs this is the h-frame chunk constraint reported below.
            "one_step_violation_squared": violation,
            "constraint_violation_squared": violation,
            "constraint_horizon_steps": jnp.mean(jnp.asarray(constraint_threshold)),
            "constraint_gap": constraint_gap,
            "lambda": jax.nn.softplus(raw_lambda) if dual == "adam" else raw_lambda,
            "distance_goal_mean": jnp.mean(distance_goal),
            "distance_step_mean": jnp.mean(distance_step),
            "gradient_norm_preclip": gradient_norm_preclip,
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
    # Adam follows QRL's softplus parameterization. Projected SGD stores λ directly.
    initial_lambda = 0.01
    raw_lambda = jnp.asarray(
        np.log(np.expm1(initial_lambda)) if dual_optimizer is not None else initial_lambda,
        dtype=jnp.float32,
    )
    dual_state = (
        dual_optimizer.init(raw_lambda) if dual_optimizer is not None
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    return params, primal_state, raw_lambda, dual_state, key


def make_activation_diagnostic_fn(model):
    """Compile the fixed-shape held-out activation/embedding diagnostic."""
    jax, _, _, _, _ = _jax_modules()

    @jax.jit
    def diagnose(params, states):
        return model.apply(
            {"params": params}, states, states, return_diagnostics=True
        )

    return diagnose


def activation_diagnostics(diagnose, params, states: np.ndarray, *, activation: str):
    """Summarize dead hidden units and representation variance on fixed states."""
    if activation not in {"relu", "gelu"}:
        raise ValueError("activation must be 'relu' or 'gelu'")
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or len(states) == 0:
        raise ValueError("diagnostic states must be a non-empty rank-2 array")
    raw = diagnose(params, states)
    preactivations = raw["preactivations"]
    dead_threshold = 0.0 if activation == "relu" else -3.0
    dead_compare = np.less_equal if activation == "relu" else np.less
    dead = {}
    for name, values in preactivations.items():
        values = np.asarray(values)
        dead[name] = float(np.mean(np.all(dead_compare(values, dead_threshold), axis=0)))

    def mean_dim_variance(values):
        values = np.asarray(values, dtype=np.float32)
        return float(np.mean(np.var(values, axis=0, dtype=np.float64)))

    return {
        "n_diagnostic_states": int(len(states)),
        "dead_unit_frac_by_layer": dead,
        "encoder_output_mean_variance": mean_dim_variance(raw["encoder_embedding"]),
        "projector_output_mean_variance": mean_dim_variance(raw["projector_embedding"]),
    }


def collapse_alarm(diagnostics: dict[str, Any], *, last_projector_layer: str,
                   dead_fraction_limit: float = 0.95,
                   min_projector_variance: float = 1e-6) -> tuple[bool, str | None]:
    """Apply the v0.8 F3 collapse gates."""
    dead = float(diagnostics["dead_unit_frac_by_layer"][last_projector_layer])
    variance = float(diagnostics["projector_output_mean_variance"])
    if dead > dead_fraction_limit:
        return True, f"dead_unit_frac:{last_projector_layer}={dead:.8g}"
    if variance < min_projector_variance:
        return True, f"projector_output_mean_variance={variance:.8g}"
    return False, None


def make_distance_predictor(model, params, *, batch_size: int = 8192):
    """Predict distances through the same separately batched encoder as cache."""
    jax, jnp, _, _, _ = _jax_modules()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    embedder = make_quasimetric_embedder(model)
    @jax.jit
    def embed_jit(inputs):
        return embedder.apply({"params": params}, inputs)

    @jax.jit
    def distance_jit(source_embedding, destination_embedding):
        return iqe_distance_from_embeddings(
            source_embedding, destination_embedding, params["raw_alpha"],
            n_components=model.n_components,
            component_dim=model.component_dim, jnp=jnp,
        )

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
            source_embedding = embed_jit(jnp.asarray(source_batch))
            destination_embedding = embed_jit(jnp.asarray(destination_batch))
            values = np.asarray(distance_jit(source_embedding, destination_embedding))
            out[start:end] = values[:current_n]
        return out

    return predict


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def make_embedding_cache_predictor(
    model,
    params,
    node_features: np.ndarray,
    node_ids: np.ndarray,
    *,
    params_sha256: str,
    dataset_id: str,
    cache_path: str | Path,
    embed_batch_size: int = 8192,
    pair_batch_size: int = 4096,
):
    """Encode each candidate node once and expose cached node-pair distances.

    The on-disk cache is float32 and keyed by checkpoint hash, ordered node-ID
    hash, feature hash, and dataset ID. Returned pair predictions use a
    GPU-resident embedding table; no model half-precision conversion occurs.
    """
    jax, jnp, _, _, _ = _jax_modules()
    features = np.ascontiguousarray(node_features, dtype=np.float32)
    ids = np.ascontiguousarray(node_ids, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(ids) or len(ids) == 0:
        raise ValueError("node_features and node_ids must be non-empty and aligned")
    if np.any(ids < 0) or np.unique(ids).size != len(ids):
        raise ValueError("node_ids must be unique and non-negative")
    if ids.size > 1 and np.any(ids[1:] < ids[:-1]):
        raise ValueError("node_ids must be sorted for pair-index lookup")
    if embed_batch_size < 1 or pair_batch_size < 1:
        raise ValueError("embedding and pair batch sizes must be positive")
    if not params_sha256 or not dataset_id:
        raise ValueError("checkpoint hash and dataset ID are required for cache keys")
    try:
        device = jax.devices("gpu")[0]
    except (IndexError, RuntimeError) as exc:  # pragma: no cover - hardware-specific
        raise RuntimeError("v0.8 embedding cache requires an available JAX GPU") from exc

    embedder = make_quasimetric_embedder(model)
    embed_jit = jax.jit(
        lambda batch: embedder.apply({"params": params}, batch)
    )
    latent_dim = int(model.n_components * model.component_dim)
    embeddings_path = Path(cache_path)
    sidecar_path = embeddings_path.with_suffix(embeddings_path.suffix + ".json")
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    key = {
        "schema": "v0.8-embedding-cache-v1",
        "params_sha256": str(params_sha256),
        "node_ids_sha256": _array_sha256(ids),
        "node_features_sha256": _array_sha256(features),
        "dataset_id": str(dataset_id),
        "dtype": "float32",
        "shape": [int(len(ids)), latent_dim],
    }
    embedding_started = time.perf_counter()
    cache_hit = False
    if embeddings_path.is_file() and sidecar_path.is_file():
        try:
            saved_key = json.loads(sidecar_path.read_text(encoding="utf-8"))
            saved = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
            cache_hit = saved_key == key and saved.shape == tuple(key["shape"]) and saved.dtype == np.float32
        except (OSError, ValueError, json.JSONDecodeError):
            cache_hit = False
    if not cache_hit:
        temporary_path = embeddings_path.with_name(embeddings_path.name + ".tmp.npy")
        output = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float32, shape=tuple(key["shape"])
        )
        for start in range(0, len(features), embed_batch_size):
            end = min(start + embed_batch_size, len(features))
            count = end - start
            batch = np.zeros((embed_batch_size, features.shape[1]), dtype=np.float32)
            batch[:count] = features[start:end]
            encoded = np.asarray(jax.device_get(embed_jit(jax.device_put(batch, device))))
            output[start:end] = encoded[:count].reshape(count, latent_dim)
        output.flush()
        del output
        os.replace(temporary_path, embeddings_path)
        sidecar_tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")
        sidecar_tmp.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(sidecar_tmp, sidecar_path)
    host_embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if host_embeddings.shape != tuple(key["shape"]) or host_embeddings.dtype != np.float32:
        raise ValueError("embedding cache has an invalid array shape or dtype")
    embeddings = jax.device_put(jnp.asarray(host_embeddings), device)
    embedding_seconds = float(time.perf_counter() - embedding_started)

    distance_jit = jax.jit(lambda source, target: iqe_distance_from_embeddings(
        source, target, params["raw_alpha"], n_components=model.n_components,
        component_dim=model.component_dim, jnp=jnp,
    ))
    distance_seconds = [0.0]
    report = {
        "cache_path": str(embeddings_path),
        "cache_hit": bool(cache_hit),
        "node_count": int(len(ids)),
        "embedding_dim": latent_dim,
        "dtype": "float32",
        "embedding_bytes": int(len(ids) * latent_dim * np.dtype(np.float32).itemsize),
        "embedding_seconds": embedding_seconds,
        "pair_distance_seconds": 0.0,
    }

    def predict_pairs(pairs: np.ndarray) -> np.ndarray:
        pair_array = np.asarray(pairs, dtype=np.int64)
        if pair_array.ndim != 2 or pair_array.shape[1] != 2:
            raise ValueError("node pairs must have shape (N,2)")
        if not len(pair_array):
            return np.empty(0, dtype=np.float32)
        source_local = np.searchsorted(ids, pair_array[:, 0])
        target_local = np.searchsorted(ids, pair_array[:, 1])
        if (np.any(source_local >= len(ids)) or np.any(target_local >= len(ids))
                or np.any(ids[source_local] != pair_array[:, 0])
                or np.any(ids[target_local] != pair_array[:, 1])):
            raise ValueError("pair references a node absent from the embedding cache")
        result = np.empty(len(pair_array), dtype=np.float32)
        started = time.perf_counter()
        for start in range(0, len(pair_array), pair_batch_size):
            end = min(start + pair_batch_size, len(pair_array))
            src = embeddings[jnp.asarray(source_local[start:end], dtype=jnp.int32)]
            dst = embeddings[jnp.asarray(target_local[start:end], dtype=jnp.int32)]
            result[start:end] = np.asarray(jax.device_get(distance_jit(src, dst)), dtype=np.float32)
        distance_seconds[0] += time.perf_counter() - started
        report["pair_distance_seconds"] = float(distance_seconds[0])
        return result
    return predict_pairs, report


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
        # Checkpoints predating v0.8 §2.1 used ReLU with no LayerNorm.
        activation=str(metadata.get("activation", "relu")),
        layernorm=bool(metadata.get("layernorm", False)),
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
