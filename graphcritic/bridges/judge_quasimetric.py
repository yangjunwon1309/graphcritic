"""Quasimetric bridge judge with framework-injected batched inference."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from graphcritic.bridges.judge import JudgeResult


def intersect_quasimetric_judges(
    d_theta: np.ndarray,
    *,
    max_distance_steps: float = 20.0,
    m_psi: np.ndarray | None = None,
    e_phi: np.ndarray | None = None,
    max_events: int | None = None,
    oracle_press_count: np.ndarray | None = None,
    oracle_max_press_count: int | None = None,
) -> dict[str, np.ndarray | dict[str, int]]:
    """Apply the v0.8.2 AND rule and explain rejected candidate combinations.

    ``d_theta`` remains the sole bridge cost/distance. Optional object scores
    are lower-bound acceptance checks only. The oracle replaces the event
    judge when requested and is intended for diagnostic graphs only.
    """
    theta = np.asarray(d_theta, dtype=np.float32)
    if theta.ndim != 1 or np.any(~np.isfinite(theta)) or np.any(theta < -1e-6):
        raise ValueError("d_theta must be a finite non-negative vector")
    if not np.isfinite(max_distance_steps) or max_distance_steps <= 0:
        raise ValueError("max_distance_steps must be positive and finite")
    n = len(theta)
    fail_theta = np.maximum(theta, 0.0) > float(max_distance_steps)

    def _optional_scores(name: str, values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.zeros(n, dtype=np.bool_)
        score = np.asarray(values, dtype=np.float32)
        if score.shape != (n,) or np.any(~np.isfinite(score)) or np.any(score < -1e-6):
            raise ValueError(f"{name} must be a finite non-negative vector aligned with d_theta")
        return score

    fail_motion = np.zeros(n, dtype=np.bool_)
    if m_psi is not None:
        motion = _optional_scores("m_psi", m_psi)
        fail_motion = motion > float(max_distance_steps)

    fail_event = np.zeros(n, dtype=np.bool_)
    if oracle_press_count is not None:
        counts = np.asarray(oracle_press_count, dtype=np.int64)
        if counts.shape != (n,) or oracle_max_press_count is None or oracle_max_press_count < 0:
            raise ValueError("oracle press counts and a non-negative maximum must align")
        fail_event = (counts < 0) | (counts > int(oracle_max_press_count))
    elif e_phi is not None:
        if max_events is None or max_events < 0:
            raise ValueError("max_events must be non-negative when e_phi is enabled")
        events = _optional_scores("e_phi", e_phi)
        fail_event = events > (float(max_events) + 0.5)
    elif max_events is not None:
        raise ValueError("max_events requires e_phi")
    if oracle_press_count is not None and e_phi is not None:
        raise ValueError("oracle event filtering replaces e_phi; they cannot both be active")

    fails = np.stack((fail_theta, fail_motion, fail_event), axis=1)
    combination_names = {
        "d_theta_only": (1, 0, 0),
        "m_psi_only": (0, 1, 0),
        "e_phi_or_oracle_only": (0, 0, 1),
        "d_theta_and_m_psi": (1, 1, 0),
        "d_theta_and_e_phi_or_oracle": (1, 0, 1),
        "m_psi_and_e_phi_or_oracle": (0, 1, 1),
        "all_three": (1, 1, 1),
    }
    categories = {
        name: int(np.count_nonzero(np.all(fails == np.asarray(pattern, dtype=np.bool_), axis=1)))
        for name, pattern in combination_names.items()
    }
    return {
        "accept": ~np.any(fails, axis=1),
        "fail_d_theta": fail_theta,
        "fail_m_psi": fail_motion,
        "fail_e_phi_or_oracle": fail_event,
        "categories": categories,
        "individual_rejections": {
            "d_theta": int(np.count_nonzero(fail_theta)),
            "m_psi": int(np.count_nonzero(fail_motion)),
            "e_phi_or_oracle": int(np.count_nonzero(fail_event)),
        },
    }


class QuasimetricJudge:
    """Accept a candidate iff its learned directed time distance is <= K."""

    name = "quasimetric"

    def __init__(
        self,
        model_dataset_id: str,
        predict_distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        max_distance_steps: float = 20.0,
        batch_size: int = 4096,
        predict_node_pairs: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if not model_dataset_id:
            raise ValueError("model_dataset_id must be non-empty")
        if not np.isfinite(max_distance_steps) or max_distance_steps <= 0:
            raise ValueError("max_distance_steps must be positive and finite")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_dataset_id = model_dataset_id
        self._predict_distance = predict_distance
        self.max_distance_steps = float(max_distance_steps)
        self.batch_size = int(batch_size)
        self._predict_node_pairs = predict_node_pairs

    def _result(self, distances: np.ndarray) -> JudgeResult:
        values = np.asarray(distances, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("distance predictor must return one scalar per pair")
        if np.any(~np.isfinite(values)) or np.any(values < -1e-6):
            raise ValueError("quasimetric predictor returned an invalid distance")
        values = np.maximum(values, 0.0)
        accept = values <= self.max_distance_steps
        costs = np.where(accept, values, np.nan).astype(np.float32)
        result = JudgeResult(accept, costs, {"distance_steps": values})
        result.validate(len(values))
        return result

    def judge_node_pairs(self, pairs: np.ndarray) -> JudgeResult:
        """Judge graph-node IDs using a precomputed embedding table."""
        if self._predict_node_pairs is None:
            raise RuntimeError("this judge has no node-embedding cache configured")
        node_pairs = np.asarray(pairs, dtype=np.int64)
        if node_pairs.ndim != 2 or node_pairs.shape[1] != 2:
            raise ValueError("node pairs must have shape (N,2)")
        if not len(node_pairs):
            return self._result(np.empty(0, dtype=np.float32))
        distances = np.empty(len(node_pairs), dtype=np.float32)
        for start in range(0, len(node_pairs), self.batch_size):
            end = min(start + self.batch_size, len(node_pairs))
            values = np.asarray(self._predict_node_pairs(node_pairs[start:end]), dtype=np.float32)
            if values.shape != (end - start,):
                raise ValueError("node-pair predictor must return one scalar per pair")
            distances[start:end] = values
        return self._result(distances)

    def judge(self, src: np.ndarray, dst: np.ndarray) -> JudgeResult:
        source = np.asarray(src, dtype=np.float32)
        target = np.asarray(dst, dtype=np.float32)
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("source and destination must be aligned rank-2 arrays")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("source and destination states must be finite")

        distances = np.empty(len(source), dtype=np.float32)
        if len(source) == 0:
            return JudgeResult(
                accept=np.empty(0, dtype=np.bool_),
                cost_steps=np.empty(0, dtype=np.float32),
                extras={"distance_steps": np.empty(0, dtype=np.float32)},
            )
        for start in range(0, len(source), self.batch_size):
            end = min(start + self.batch_size, len(source))
            batch_distances = np.asarray(
                self._predict_distance(source[start:end], target[start:end]),
                dtype=np.float32,
            )
            if batch_distances.shape != (end - start,):
                raise ValueError("distance predictor must return one scalar per pair")
            distances[start:end] = batch_distances
        # Clamp tiny negative floating-point noise to zero. IQE is non-negative
        # by construction; materially negative outputs are rejected above.
        return self._result(distances)

