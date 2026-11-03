"""Graph edge-weight policies for hop, time, and LCB costs."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import GraphConfig
from graphcritic.stats import edge_neglog_lcb
from graphcritic.types import EdgeSet


def edge_weights(
    edges: EdgeSet,
    cfg: GraphConfig,
) -> NDArray[np.float64]:
    edges.validate()
    c = step_cost(cfg.gamma)
    if cfg.weight_policy == "hop":
        weights = np.ones(edges.n_edges, dtype=np.float64)
    elif cfg.weight_policy == "time":
        weights = c * edges.kappa.astype(np.float64)
    elif cfg.weight_policy == "lcb":
        weights = edge_neglog_lcb(edges, cfg.pessimism) + c * edges.kappa.astype(np.float64)
    else:  # GraphConfig validates this, but protect deserialized/mutated objects.
        raise ValueError(f"unknown edge weight policy {cfg.weight_policy!r}")

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("edge weights must be finite and strictly positive")
    return np.asarray(weights, dtype=np.float64)


def step_cost(gamma: float) -> float:
    """Return the time cost per environment step, ``-log(gamma)``."""
    if not np.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between zero and one")
    return float(-np.log(float(gamma)))
