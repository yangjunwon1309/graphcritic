"""Small, framework-independent bridge judge interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class JudgeResult:
    """Per-pair decisions and predicted travel costs from a bridge judge.

    ``cost_steps`` is required to be finite and non-negative only for accepted
    pairs. Rejected entries may be NaN. Judge-specific diagnostics are stored
    in ``extras`` with the candidate batch on their leading axis.
    """

    accept: np.ndarray
    cost_steps: np.ndarray
    extras: dict[str, np.ndarray]

    def validate(self, n_pairs: int) -> None:
        accept = np.asarray(self.accept)
        costs = np.asarray(self.cost_steps, dtype=np.float64)
        if accept.shape != (n_pairs,) or accept.dtype != np.bool_:
            raise ValueError("judge accept must be a boolean vector aligned with candidates")
        if costs.shape != (n_pairs,):
            raise ValueError("judge cost_steps must align with candidates")
        if np.any(~np.isfinite(costs[accept])) or np.any(costs[accept] < 0):
            raise ValueError("accepted judge costs must be finite and non-negative")
        for name, values in self.extras.items():
            if np.asarray(values).ndim == 0 or np.asarray(values).shape[0] != n_pairs:
                raise ValueError(f"judge extra {name!r} must align with candidates")


@runtime_checkable
class Judge(Protocol):
    """Judge contract shared by flow and quasimetric bridge builders."""

    name: str
    model_dataset_id: str

    def judge(self, src: np.ndarray, dst: np.ndarray) -> JudgeResult:
        """Evaluate aligned source/destination state vectors."""

