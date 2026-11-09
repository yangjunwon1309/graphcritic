"""Quasimetric bridge judge with framework-injected batched inference."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from graphcritic.bridges.judge import JudgeResult


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
        if np.any(~np.isfinite(distances)) or np.any(distances < -1e-6):
            raise ValueError("quasimetric predictor returned an invalid distance")
        # Clamp tiny negative floating-point noise to zero. IQE is non-negative
        # by construction; materially negative outputs are rejected above.
        distances = np.maximum(distances, 0.0)
        accept = distances <= self.max_distance_steps
        costs = np.where(accept, distances, np.nan).astype(np.float32)
        result = JudgeResult(accept, costs, {"distance_steps": distances})
        result.validate(len(source))
        return result

