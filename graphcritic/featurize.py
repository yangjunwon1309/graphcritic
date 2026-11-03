"""Feature preparation for graph-state clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import FeatureConfig
from graphcritic.types import SchemaError, TransitionSet


@dataclass(frozen=True)
class Standardizer:
    """A fitted state-feature transform; representatives remain unscaled."""

    mean: NDArray[np.float32]
    std: NDArray[np.float32]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.ndim != 1 or std.shape != mean.shape or mean.size == 0:
            raise SchemaError("standardizer mean and std must be equally sized vectors")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 1e-6):
            raise SchemaError("standardizer statistics must be finite and std >= 1e-6")
        object.__setattr__(self, "mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "std", np.ascontiguousarray(std))

    @classmethod
    def fit(cls, trans: TransitionSet, cfg: FeatureConfig) -> Standardizer:
        """Fit population statistics over the distinct-state occurrence table."""
        trans.validate()
        x, _, _ = state_table(trans)
        if x.shape[0] == 0:
            raise SchemaError("cannot fit a standardizer on an empty transition set")
        if not cfg.standardize:
            return cls(np.zeros(x.shape[1], dtype=np.float32), np.ones(x.shape[1], dtype=np.float32))
        mean64 = np.mean(x, axis=0, dtype=np.float64)
        std64 = np.std(x, axis=0, dtype=np.float64)
        std64 = np.maximum(std64, 1e-6)
        return cls(mean64.astype(np.float32), std64.astype(np.float32))

    def transform(self, x: NDArray) -> NDArray[np.float32]:
        values = np.asarray(x)
        if values.ndim == 0 or values.shape[-1] != self.mean.size:
            raise SchemaError("features have an incompatible final dimension")
        if not np.all(np.isfinite(values)):
            raise SchemaError("features must be finite")
        return np.ascontiguousarray((values.astype(np.float32) - self.mean) / self.std, dtype=np.float32)

    def as_stats(self) -> dict[str, NDArray[np.float32]]:
        return {"mean": self.mean.copy(), "std": self.std.copy()}


def state_table(
    trans: TransitionSet,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.bool_]]:
    """Return current states and one terminal next-state occurrence per episode.

    State IDs ``0..N-1`` refer to current states. Terminal next states follow
    in episode order, at IDs ``N..N+n_episodes-1``. The explicit
    ``success_cur`` label is authoritative for current-state occurrences.
    """
    trans.validate()
    n_rows = trans.n_rows
    final_rows = np.flatnonzero(trans.terminal).astype(np.int64, copy=False)
    x = np.concatenate((trans.feat, trans.next_feat[final_rows]), axis=0).astype(np.float32, copy=False)
    ref_cur = np.column_stack(
        (np.arange(n_rows, dtype=np.int64), np.zeros(n_rows, dtype=np.int64))
    )
    ref_final = np.column_stack(
        (final_rows, np.ones(final_rows.size, dtype=np.int64))
    )
    ref = np.concatenate((ref_cur, ref_final), axis=0)
    success = np.concatenate((trans.success_cur, trans.success[final_rows])).astype(np.bool_, copy=False)
    return np.ascontiguousarray(x), np.ascontiguousarray(ref), np.ascontiguousarray(success)
