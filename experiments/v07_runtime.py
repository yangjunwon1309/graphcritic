"""JAX/Flax runtime for training and validating v0.7 bridge models.

This stays under ``experiments`` so importing the framework-independent
``graphcritic`` package does not import the heavyweight ML stack.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from graphcritic.bridges.check import BridgePlan, make_and_evaluate as _make_and_evaluate


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


def load_model_pair(directory: str | Path, n_cont: int, n_bits: int):
    jax, jnp, _, serialization, _, MLP, TransitionMLP = _flax_modules()
    root = Path(directory)
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
    """Create a reusable checker closure with compiled policy and f calls."""
    jax, jnp, _, _, _, _, _ = _flax_modules()
    pi_model, pi_params = policy
    f_model, f_params = transition

    @jax.jit
    def pi_apply(x):
        return pi_model.apply({"params": pi_params}, x)

    @jax.jit
    def f_apply(x):
        return f_model.apply({"params": f_params}, x)

    def pi_predict(x):
        return pi_apply(jnp.asarray(x, dtype=jnp.float32))

    def f_predict(x):
        return f_apply(jnp.asarray(x, dtype=jnp.float32))

    def evaluate(source_states, target_states, source_ids, target_ids, **kwargs):
        return _make_and_evaluate(
            source_states, target_states, source_ids, target_ids,
            policy=(pi_predict, None), transition=(f_predict, None), **kwargs,
        )

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
) -> list[BridgePlan | None]:
    """JIT model calls, then run the framework-independent endpoint checker."""
    evaluator = make_bridge_evaluator(policy, transition)
    return evaluator(
        source_states, target_states, source_ids, target_ids,
        epsilon_z=epsilon_z, state_std=state_std, delta_std=delta_std,
        n_cont=n_cont, n_bits=n_bits, h=h, max_chunks=max_chunks,
        n_samples=n_samples, seed=seed, action_low=action_low,
        action_high=action_high, candidate_batch=candidate_batch,
    )
