"""Framework-independent adapter for the v0.7 policy/transition judge."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from graphcritic.bridges.check import BridgePlan
from graphcritic.bridges.judge import JudgeResult


class FlowJudge:
    """Adapt the existing batched π/f evaluator to the common Judge API."""

    name = "flow"

    def __init__(
        self,
        model_dataset_id: str,
        evaluate_plans: Callable[[np.ndarray, np.ndarray], list[BridgePlan | None]],
        *,
        horizon_steps: int = 5,
        max_chunks: int = 4,
    ) -> None:
        if not model_dataset_id:
            raise ValueError("model_dataset_id must be non-empty")
        if horizon_steps < 1 or max_chunks < 1:
            raise ValueError("horizon_steps and max_chunks must be positive")
        self.model_dataset_id = model_dataset_id
        self._evaluate_plans = evaluate_plans
        self.horizon_steps = int(horizon_steps)
        self.max_chunks = int(max_chunks)

    def judge(self, src: np.ndarray, dst: np.ndarray) -> JudgeResult:
        source = np.asarray(src, dtype=np.float32)
        target = np.asarray(dst, dtype=np.float32)
        if source.ndim != 2 or source.shape != target.shape:
            raise ValueError("source and destination must be aligned rank-2 arrays")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("source and destination states must be finite")
        plans = self._evaluate_plans(source, target)
        if len(plans) != len(source):
            raise ValueError("flow evaluator returned a different number of plans than candidates")

        accept = np.fromiter((plan is not None for plan in plans), dtype=np.bool_, count=len(plans))
        cost = np.full(len(plans), np.nan, dtype=np.float32)
        chunk_count = np.zeros(len(plans), dtype=np.uint8)
        sample_index = np.full(len(plans), -1, dtype=np.int16)
        endpoint_error = np.full(len(plans), np.nan, dtype=np.float32)
        action_chunks = np.zeros(
            (len(plans), self.max_chunks, self.horizon_steps, 5), dtype=np.float32
        )
        for i, plan in enumerate(plans):
            if plan is None:
                continue
            if plan.chunks.shape != (plan.length_chunks, self.horizon_steps, 5):
                raise ValueError("flow plan action shape disagrees with judge settings")
            cost[i] = float(plan.length_chunks * self.horizon_steps)
            chunk_count[i] = int(plan.length_chunks)
            sample_index[i] = int(plan.sample_index)
            endpoint_error[i] = float(plan.endpoint_error)
            action_chunks[i, :plan.length_chunks] = plan.chunks
        result = JudgeResult(accept, cost, {
            "chunk_count": chunk_count,
            "sample_index": sample_index,
            "endpoint_error": endpoint_error,
            "action_chunks": action_chunks,
        })
        result.validate(len(source))
        return result

