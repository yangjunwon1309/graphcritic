"""JAX/Flax runtime for training and validating v0.7 bridge models.

This stays under ``experiments`` so importing the framework-independent
``graphcritic`` package does not import the heavyweight ML stack.
"""

from __future__ import annotations

from pathlib import Path
from functools import partial
import json
from typing import Any

import numpy as np

from graphcritic.bridges.check import BridgePlan


def _flax_modules():
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from flax import serialization
        import optax
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("v0.7 model training requires the bench JAX/Flax/Optax environment") from exc

    class MLP(nn.Module):
        output_dim: int
        width: int = 512
        depth: int = 4

        @nn.compact
        def __call__(self, x):
            for layer in range(self.depth):
                x = nn.Dense(self.width, name=f"dense_{layer}")(x)
                x = nn.LayerNorm(name=f"norm_{layer}")(x)
                x = nn.gelu(x)
            return nn.Dense(self.output_dim, name="out")(x)

    class TransitionMLP(nn.Module):
        n_cont: int
        n_bits: int
        width: int = 512
        depth: int = 4

        @nn.compact
        def __call__(self, x):
            for layer in range(self.depth):
                x = nn.Dense(self.width, name=f"dense_{layer}")(x)
                x = nn.LayerNorm(name=f"norm_{layer}")(x)
                x = nn.gelu(x)
            continuous = nn.Dense(self.n_cont, name="continuous_head")(x)
            buttons = nn.Dense(self.n_bits, name="button_logits")(x)
            return jnp.concatenate((continuous, buttons), axis=-1)

    return jax, jnp, nn, serialization, optax, MLP, TransitionMLP


def save_checkpoint(directory: str | Path, params: Any, step: int,
                    metadata: dict[str, Any]) -> Path:
    _, _, _, serialization, _, _, _ = _flax_modules()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "params.msgpack").write_bytes(serialization.to_bytes(params))
    (path / "metadata.json").write_text(
        json.dumps({**metadata, "step": int(step)}, indent=2, sort_keys=True) + "\n"
    )
    return path


def validate_checkpoint_dataset_ids(
    pi_metadata: dict[str, Any], f_metadata: dict[str, Any],
    expected_dataset_id: str,
) -> None:
    """Reject mixed or cross-dataset π/f checkpoints before graph building."""
    pi_id = pi_metadata.get("dataset_id")
    f_id = f_metadata.get("dataset_id")
    if not expected_dataset_id:
        raise ValueError("an expected dataset ID is required when loading v0.7 models")
    if pi_id != f_id:
        raise ValueError(f"π/f checkpoint dataset IDs differ: π={pi_id!r}, f={f_id!r}")
    if pi_id != expected_dataset_id:
        raise ValueError(
            "checkpoint dataset ID does not match graph-build data: "
            f"checkpoint={pi_id!r}, expected={expected_dataset_id!r}"
        )


def load_model_pair(
    directory: str | Path, n_cont: int, n_bits: int, *,
    expected_dataset_id: str | None = None,
):
    jax, jnp, _, serialization, _, MLP, TransitionMLP = _flax_modules()
    root = Path(directory)
    pi_metadata = json.loads((root / "pi" / "metadata.json").read_text(encoding="utf-8"))
    f_metadata = json.loads((root / "f" / "metadata.json").read_text(encoding="utf-8"))
    if expected_dataset_id is None:
        raise ValueError("expected_dataset_id is required for v0.7 model loading")
    validate_checkpoint_dataset_ids(pi_metadata, f_metadata, expected_dataset_id)
    for model_meta in (pi_metadata, f_metadata):
        if int(model_meta.get("n_cont", -1)) != n_cont or int(model_meta.get("n_bits", -1)) != n_bits:
            raise ValueError("checkpoint state dimensions do not match the graph-build dataset")
    pi_model = MLP(output_dim=25)
    f_model = TransitionMLP(n_cont=n_cont, n_bits=n_bits)
    key = jax.random.PRNGKey(0)
    state_dim = n_cont + n_bits
    pi_template = pi_model.init(
        key, jnp.zeros((1, 2 * state_dim + 26), dtype=jnp.float32)
    )["params"]
    f_template = f_model.init(
        key, jnp.zeros((1, state_dim + 25), dtype=jnp.float32)
    )["params"]
    pi_params = serialization.from_bytes(
        pi_template, (root / "pi" / "params.msgpack").read_bytes()
    )
    f_params = serialization.from_bytes(
        f_template, (root / "f" / "params.msgpack").read_bytes()
    )
    return (pi_model, pi_params), (f_model, f_params)


def make_bridge_evaluator(policy: tuple[Any, Any], transition: tuple[Any, Any]):
    """Create a single-JIT, batched π/f checker and action-plan retriever."""
    jax, jnp, _, _, _, _, _ = _flax_modules()
    pi_model, pi_params = policy
    f_model, f_params = transition

    def policy_action(state, target, noise):
        def flow_step(step, action):
            tau = jnp.full((action.shape[0], 1), step / 10.0, dtype=jnp.float32)
            model_input = jnp.concatenate((state, target, action, tau), axis=1)
            velocity = pi_model.apply({"params": pi_params}, model_input)
            return action + velocity / 10.0

        action = jax.lax.fori_loop(0, 10, flow_step, noise)
        return jnp.clip(action.reshape((-1, 5, 5)), -1.0, 1.0)

    @partial(jax.jit, static_argnums=(5,))
    def evaluate_batch(source, target, noise, epsilon_z, scale, n_bits: int):
        # noise: [chunk, candidate, sample, 25]
        n_candidates, state_dim = source.shape
        n_chunks, _, n_samples, _ = noise.shape
        start = jnp.broadcast_to(source[:, None, :], (n_candidates, n_samples, state_dim))
        goal = jnp.broadcast_to(target[:, None, :], (n_candidates, n_samples, state_dim))
        initial_length = jnp.zeros((n_candidates, n_samples), dtype=jnp.int32)
        initial_error = jnp.full((n_candidates, n_samples), jnp.inf, dtype=jnp.float32)

        def rollout(carry, step_input):
            trajectories, lengths, errors = carry
            chunk_index, noise_chunk = step_input
            flat_state = trajectories.reshape((n_candidates * n_samples, state_dim))
            flat_goal = goal.reshape((n_candidates * n_samples, state_dim))
            action = policy_action(
                flat_state, flat_goal,
                noise_chunk.reshape((n_candidates * n_samples, 25)),
            ).reshape((n_candidates * n_samples, 25))
            model_input = jnp.concatenate((flat_state, action), axis=1)
            prediction = f_model.apply({"params": f_params}, model_input)
            prediction = prediction.reshape((n_candidates, n_samples, state_dim))
            next_cont = flat_state[:, :state_dim - n_bits].reshape(
                (n_candidates, n_samples, state_dim - n_bits)
            ) + prediction[:, :, :state_dim - n_bits] * scale[None, None, :]
            next_bits = (prediction[:, :, state_dim - n_bits:] > 0.0).astype(jnp.float32)
            next_state = jnp.concatenate((next_cont, next_bits), axis=2)
            err = jnp.max(
                jnp.abs(next_cont - goal[:, :, :state_dim - n_bits])
                / epsilon_z[None, None, :], axis=2,
            )
            exact_bits = jnp.all(next_bits == goal[:, :, state_dim - n_bits:], axis=2)
            hit = (err <= 1.0) & exact_bits & (lengths == 0)
            lengths = jnp.where(hit, chunk_index + 1, lengths)
            errors = jnp.where(hit, err, errors)
            return (next_state, lengths, errors), None

        chunk_ids = jnp.arange(n_chunks, dtype=jnp.int32)
        (_, lengths, errors), _ = jax.lax.scan(
            rollout, (start, initial_length, initial_error),
            (chunk_ids, noise),
        )
        valid_lengths = jnp.where(lengths > 0, lengths, n_chunks + 1)
        sample_ids = jnp.argmin(valid_lengths, axis=1)
        best_lengths = jnp.take_along_axis(valid_lengths, sample_ids[:, None], axis=1)[:, 0]
        best_errors = jnp.take_along_axis(errors, sample_ids[:, None], axis=1)[:, 0]
        best_lengths = jnp.where(best_lengths <= n_chunks, best_lengths, 0)
        return best_lengths, sample_ids, best_errors

    @partial(jax.jit, static_argnums=(4,))
    def retrieve_selected_actions(source, target, noise, scale, n_bits: int):
        # Only selected successful candidate/sample plans are sent to this pass.
        n_candidates, state_dim = source.shape

        def rollout(state, step_input):
            _, noise_chunk = step_input
            action = policy_action(state, target, noise_chunk)
            flat_action = action.reshape((n_candidates, 25))
            prediction = f_model.apply(
                {"params": f_params}, jnp.concatenate((state, flat_action), axis=1)
            )
            n_cont = state_dim - n_bits
            next_cont = state[:, :n_cont] + prediction[:, :n_cont] * scale[None, :]
            next_bits = (prediction[:, n_cont:] > 0.0).astype(jnp.float32)
            return jnp.concatenate((next_cont, next_bits), axis=1), action

        chunk_ids = jnp.arange(noise.shape[1], dtype=jnp.int32)
        _, actions = jax.lax.scan(
            rollout, source, (chunk_ids, jnp.swapaxes(noise, 0, 1))
        )
        return jnp.swapaxes(actions, 0, 1)

    def evaluate(
        source_states, target_states, source_ids, target_ids, *,
        epsilon_z, state_std, delta_std, n_cont, n_bits, h=5,
        max_chunks=4, n_samples=64, seed=0, action_low=-1.0,
        action_high=1.0, candidate_batch=1024, noise=None,
    ):
        if h != 5 or action_low != -1.0 or action_high != 1.0:
            raise ValueError("the compiled v0.7 checker requires h=5 and actions in [-1, 1]")
        if max_chunks < 1 or n_samples < 1 or candidate_batch < 1:
            raise ValueError("max_chunks, n_samples, and candidate_batch must be positive")
        src = np.asarray(source_states, dtype=np.float32)
        dst = np.asarray(target_states, dtype=np.float32)
        sid = np.asarray(source_ids, dtype=np.int64)
        tid = np.asarray(target_ids, dtype=np.int64)
        eps = np.asarray(epsilon_z, dtype=np.float32)
        state_scale = np.asarray(state_std, dtype=np.float32)
        delta_scale = np.asarray(delta_std, dtype=np.float32)
        if src.shape != dst.shape or sid.shape != (len(src),) or tid.shape != sid.shape:
            raise ValueError("source/target states and candidate IDs must align")
        if src.ndim != 2 or src.shape[1] != n_cont + n_bits:
            raise ValueError("candidate state dimension does not match n_cont + n_bits")
        if eps.shape != (n_cont,) or state_scale.shape != (n_cont,) or delta_scale.shape != (n_cont,):
            raise ValueError("endpoint and model scales must match continuous dimensions")
        if np.any(eps <= 0) or np.any(state_scale <= 0) or np.any(delta_scale <= 0):
            raise ValueError("endpoint and model scales must be positive")
        if noise is not None:
            fixed_noise = np.asarray(noise, dtype=np.float32)
            expected_noise_shape = (max_chunks, len(src), n_samples, h * 5)
            if fixed_noise.shape != expected_noise_shape:
                raise ValueError(f"noise must have shape {expected_noise_shape}")
        else:
            fixed_noise = None

        results: list[BridgePlan | None] = [None] * len(src)
        rng = np.random.default_rng(seed)
        scale = delta_scale / state_scale
        for base in range(0, len(src), candidate_batch):
            end = min(len(src), base + candidate_batch)
            b = end - base
            if fixed_noise is None:
                batch_noise = np.stack([
                    rng.standard_normal((b, n_samples, h * 5), dtype=np.float32)
                    for _ in range(max_chunks)
                ])
            else:
                batch_noise = fixed_noise[:, base:end]
            lengths, sample_ids, errors = evaluate_batch(
                jnp.asarray(src[base:end]), jnp.asarray(dst[base:end]),
                jnp.asarray(batch_noise), jnp.asarray(eps), jnp.asarray(scale), int(n_bits),
            )
            lengths = np.asarray(lengths, dtype=np.int32)
            sample_ids = np.asarray(sample_ids, dtype=np.int32)
            errors = np.asarray(errors, dtype=np.float32)
            accepted_local = np.flatnonzero(lengths > 0)
            if not accepted_local.size:
                continue
            chosen_noise = np.stack([
                batch_noise[:, local_id, sample_id, :]
                for local_id, sample_id in zip(
                    accepted_local, sample_ids[accepted_local], strict=True
                )
            ], axis=0)
            accepted_count = len(accepted_local)
            # Keep the retrieval shape fixed across batches. JAX otherwise
            # recompiles once for every distinct number of accepted candidates.
            padded_source = np.zeros((candidate_batch, src.shape[1]), dtype=np.float32)
            padded_target = np.zeros_like(padded_source)
            padded_noise = np.zeros(
                (candidate_batch, max_chunks, h * 5), dtype=np.float32
            )
            padded_source[:accepted_count] = src[base:end][accepted_local]
            padded_target[:accepted_count] = dst[base:end][accepted_local]
            padded_noise[:accepted_count] = chosen_noise
            actions = np.asarray(retrieve_selected_actions(
                jnp.asarray(padded_source), jnp.asarray(padded_target),
                jnp.asarray(padded_noise), jnp.asarray(scale), int(n_bits),
            )[:accepted_count], dtype=np.float32)
            for plan_index, local_id in enumerate(accepted_local):
                length = int(lengths[local_id])
                results[base + int(local_id)] = BridgePlan(
                    source=int(sid[base + int(local_id)]),
                    target=int(tid[base + int(local_id)]),
                    chunks=actions[plan_index, :length].copy(),
                    length_chunks=length,
                    endpoint_error=float(errors[local_id]),
                    sample_index=int(sample_ids[local_id]),
                )
        return results

    return evaluate


def make_and_evaluate(
    source_states: np.ndarray,
    target_states: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    policy: tuple[Any, Any],
    transition: tuple[Any, Any],
    epsilon_z: np.ndarray,
    state_std: np.ndarray,
    delta_std: np.ndarray,
    n_cont: int,
    n_bits: int,
    h: int = 5,
    max_chunks: int = 4,
    n_samples: int = 64,
    seed: int = 0,
    action_low: float = -1.0,
    action_high: float = 1.0,
    candidate_batch: int = 32,
    noise: np.ndarray | None = None,
) -> list[BridgePlan | None]:
    """JIT-batched model calls and endpoint checks."""
    evaluator = make_bridge_evaluator(policy, transition)
    return evaluator(
        source_states, target_states, source_ids, target_ids,
        epsilon_z=epsilon_z, state_std=state_std, delta_std=delta_std,
        n_cont=n_cont, n_bits=n_bits, h=h, max_chunks=max_chunks,
        n_samples=n_samples, seed=seed, action_low=action_low,
        action_high=action_high, candidate_batch=candidate_batch, noise=noise,
    )
