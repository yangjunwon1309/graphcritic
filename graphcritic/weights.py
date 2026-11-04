"""Graph edge-weight policies for hop, time, and LCB costs."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaincinv

from graphcritic.config import GraphConfig
from graphcritic.stats import edge_neglog_lcb
from graphcritic.types import EdgeSet


def edge_weights(
    edges: EdgeSet,
    cfg: GraphConfig,
    *,
    d_step: float | None = None,
    meta: dict | None = None,
) -> NDArray[np.float64]:
    edges.validate()
    c = step_cost(cfg.gamma)
    if cfg.graph_kind == "state":
        weights = c * edges.kappa.astype(np.float64)
        bridge = edges.kind == 1
        if np.any(bridge):
            policy = cfg.bridge.cost.policy
            if policy == "prior":
                radius = (
                    cfg.bridge.radius * float(d_step)
                    if cfg.bridge.radius_unit == "step" and d_step is not None
                    else cfg.bridge.radius
                )
                if radius <= 0 or not np.isfinite(radius):
                    raise ValueError("state-graph step-radius policy requires a valid d_step")
                lambda_bridge = c * (
                    cfg.bridge.cost.lambda0_steps
                    + cfg.bridge.cost.lambda1_steps * np.square(edges.dist[bridge] / radius)
                )
            elif policy == "data_calibrated":
                n_bins = int((meta or {}).get("bridge_n_bins", cfg.bridge.cost.n_bins))
                level = cfg.bridge.cost.delta / max(n_bins, 1)
                ell = betaincinv(
                    cfg.bridge.cost.prior[0] + edges.n_pos[bridge],
                    cfg.bridge.cost.prior[1] + edges.n_neg[bridge],
                    level,
                )
                lambda_bridge = -np.log(np.maximum(ell, 1e-300))
            elif policy == "sim_verified":
                if np.any(edges.n_pos[bridge] < 1):
                    raise ValueError("sim_verified graph contains bridges without a successful rollout")
                lambda_bridge = np.zeros(np.count_nonzero(bridge), dtype=np.float64)
            else:
                raise ValueError(f"unknown bridge cost policy {policy!r}")
            weights[bridge] += lambda_bridge
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("state-graph edge weights must be finite and positive")
        return weights

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
