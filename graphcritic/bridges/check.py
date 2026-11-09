"""Framework-independent records for validated action-chunk bridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BridgePlan:
    source: int
    target: int
    chunks: np.ndarray
    length_chunks: int
    endpoint_error: float
    sample_index: int

    @property
    def n_steps(self) -> int:
        return int(self.length_chunks * self.chunks.shape[1])


def _model_apply(model: Any, params: Any, inputs: np.ndarray) -> np.ndarray:
    """Call either a Flax-style model or an injected predictor function."""
    if hasattr(model, "apply"):
        return np.asarray(model.apply({"params": params}, inputs))
    if params is None:
        return np.asarray(model(inputs))
    return np.asarray(model(params, inputs))


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
    """Sample policy chunks and retain plans whose modeled endpoint is valid.

    Inputs use ``[z-normalized continuous coordinates, exact binary bits]``.
    At each five-action boundary, the policy is conditioned on its latest
    predicted state and the fixed target. The earliest valid endpoint wins.
    """
    pi_model, pi_params = policy
    f_model, f_params = transition
    src = np.asarray(source_states, dtype=np.float32)
    dst = np.asarray(target_states, dtype=np.float32)
    sid = np.asarray(source_ids, dtype=np.int64)
    tid = np.asarray(target_ids, dtype=np.int64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != n_cont + n_bits:
        raise ValueError("source/target states must be aligned schema vectors")
    if sid.shape != (len(src),) or tid.shape != sid.shape:
        raise ValueError("candidate IDs must align with source/target states")
    if n_samples < 1 or max_chunks < 1 or h < 1 or candidate_batch < 1:
        raise ValueError("n_samples, max_chunks, h, and candidate_batch must be positive")
    epsilon_z = np.asarray(epsilon_z, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)
    delta_std = np.asarray(delta_std, dtype=np.float32)
    if epsilon_z.shape != (n_cont,) or state_std.shape != (n_cont,) or delta_std.shape != (n_cont,):
        raise ValueError("state scales and endpoint tolerances must match continuous dimensions")
    if np.any(epsilon_z <= 0) or np.any(state_std <= 0) or np.any(delta_std <= 0):
        raise ValueError("state scales and endpoint tolerances must be positive")
    count = len(src)
    results: list[BridgePlan | None] = [None] * count
    if count == 0:
        return results
    fixed_noise = None if noise is None else np.asarray(noise, dtype=np.float32)
    if fixed_noise is not None and fixed_noise.shape != (
        max_chunks, count, n_samples, h * 5
    ):
        raise ValueError("fixed noise must have shape (max_chunks, candidates, samples, h * 5)")

    scale = delta_std / state_std
    rng = np.random.default_rng(seed)
    for base in range(0, count, candidate_batch):
        end = min(count, base + candidate_batch)
        b = end - base
        start0 = src[base:end]
        target = dst[base:end]
        start = np.repeat(start0[:, None, :], n_samples, axis=1).reshape(b * n_samples, -1)
        goal = np.repeat(target[:, None, :], n_samples, axis=1).reshape(b * n_samples, -1)
        trajectories = start.copy()
        chunks_kept = np.zeros((b * n_samples, max_chunks, h, 5), dtype=np.float32)
        found_length = np.zeros(b * n_samples, dtype=np.int8)
        found_error = np.full(b * n_samples, np.inf, dtype=np.float32)
        for length in range(1, max_chunks + 1):
            if fixed_noise is None:
                action = rng.standard_normal((b * n_samples, h * 5), dtype=np.float32)
            else:
                action = fixed_noise[length - 1, base:end].reshape(b * n_samples, h * 5).copy()
            for flow_step in range(10):
                tau = np.full((b * n_samples, 1), flow_step / 10.0, dtype=np.float32)
                flow_input = np.concatenate((trajectories, goal, action, tau), axis=1)
                velocity = _model_apply(pi_model, pi_params, flow_input)
                if velocity.shape != action.shape:
                    raise ValueError("policy must return one 25D velocity vector per sample")
                action += velocity / 10.0
            action = np.clip(action.reshape(b * n_samples, h, 5), action_low, action_high)
            model_input = np.concatenate((trajectories, action.reshape(b * n_samples, h * 5)), axis=1)
            pred = _model_apply(f_model, f_params, model_input)
            if pred.shape != (b * n_samples, n_cont + n_bits):
                raise ValueError("transition model output width must equal continuous plus binary state dims")
            pred_cont = trajectories[:, :n_cont] + pred[:, :n_cont] * scale
            pred_bits = (pred[:, n_cont:] > 0.0).astype(np.float32)
            trajectories = np.concatenate((pred_cont, pred_bits), axis=1)
            chunks_kept[:, length - 1] = action
            error = np.max(
                np.abs(trajectories[:, :n_cont] - goal[:, :n_cont]) / epsilon_z, axis=1
            )
            exact_bits = np.all(trajectories[:, n_cont:] == goal[:, n_cont:], axis=1)
            hit = (error <= 1.0) & exact_bits & (found_length == 0)
            found_length[hit] = length
            found_error[hit] = error[hit]
        for local in range(b):
            sample_ids = np.arange(local * n_samples, (local + 1) * n_samples)
            valid = sample_ids[found_length[sample_ids] > 0]
            if not valid.size:
                continue
            chosen = int(valid[np.argmin(found_length[valid])])
            length = int(found_length[chosen])
            results[base + local] = BridgePlan(
                source=int(sid[base + local]), target=int(tid[base + local]),
                chunks=chunks_kept[chosen, :length].copy(), length_chunks=length,
                endpoint_error=float(found_error[chosen]), sample_index=chosen % n_samples,
            )
    return results

