"""Statistical primitives for conservative edge-success estimates."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betaincinv

from graphcritic.config import PessimismConfig
from graphcritic.types import EdgeSet, SchemaError


def lcb_level(cfg: PessimismConfig, n_edges: int) -> float:
    """Return the common lower-tail probability used for all graph edges."""
    if isinstance(n_edges, bool) or not isinstance(n_edges, (int, np.integer)) or n_edges < 0:
        raise SchemaError("n_edges must be a non-negative integer")
    if cfg.mode == "union":
        return float(cfg.delta / max(int(n_edges), 1))
    if cfg.mode == "fixed":
        # PessimismConfig validates lam for fixed mode.
        assert cfg.lam is not None
        return float(math.exp(-cfg.lam))
    raise SchemaError(f"unknown pessimism mode {cfg.mode!r}")


def beta_lcb(
    n_pos: ArrayLike,
    n_neg: ArrayLike,
    prior: tuple[float, float],
    level: float,
) -> NDArray[np.float64]:
    """Lower quantile of Beta(a0+n_pos, b0+n_neg) at ``level``."""
    pos = np.asarray(n_pos)
    neg = np.asarray(n_neg)
    if pos.shape != neg.shape:
        raise SchemaError("n_pos and n_neg must have the same shape")
    if (not np.issubdtype(pos.dtype, np.number) or not np.issubdtype(neg.dtype, np.number)
            or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(neg))
            or np.any(pos < 0) or np.any(neg < 0)
            or np.any(pos != np.floor(pos)) or np.any(neg != np.floor(neg))):
        raise SchemaError("occurrence counts must be finite non-negative integers")
    if len(prior) != 2 or any(not np.isfinite(v) or v <= 0 for v in prior):
        raise SchemaError("Beta prior must contain two positive finite values")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise SchemaError("level must be strictly between zero and one")

    alpha = float(prior[0]) + pos.astype(np.float64, copy=False)
    beta = float(prior[1]) + neg.astype(np.float64, copy=False)
    result = np.asarray(betaincinv(alpha, beta, float(level)), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise SchemaError("Beta lower quantile calculation produced non-finite values")
    return result


def edge_neglog_lcb(edges: EdgeSet, cfg: PessimismConfig) -> NDArray[np.float64]:
    """Return ``-log(ell_e)`` using data or bridge priors by edge kind."""
    edges.validate()
    level = lcb_level(cfg, edges.n_edges)
    ell = np.empty(edges.n_edges, dtype=np.float64)
    data = edges.kind == 0
    bridge = edges.kind == 1
    if np.any(data):
        ell[data] = beta_lcb(edges.n_pos[data], edges.n_neg[data], cfg.prior_data, level)
    if np.any(bridge):
        ell[bridge] = beta_lcb(edges.n_pos[bridge], edges.n_neg[bridge], cfg.prior_bridge, level)
    ell = np.maximum(ell, 1e-300)
    costs = -np.log(ell)
    if not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise SchemaError("edge confidence costs must be finite and non-negative")
    return costs
