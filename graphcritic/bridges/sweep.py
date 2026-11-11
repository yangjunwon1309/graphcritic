"""Summaries for bridge-budget layered shortest-path sweeps."""

from __future__ import annotations

import numpy as np


def summarize_budget_sweep(cost_steps: np.ndarray, *, tolerance: float = 0.005) -> dict:
    """Summarize reachability and finite path-cost quantiles for every budget."""
    costs = np.asarray(cost_steps, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[1] < 1:
        raise ValueError("cost_steps must be a non-empty (budget, node) matrix")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    reachable = np.isfinite(costs)
    fractions = np.mean(reachable, axis=1)
    r_sat = float(fractions[-1])
    threshold = r_sat - float(tolerance)
    eligible = np.flatnonzero(fractions >= threshold)
    b_star = int(eligible[0]) if len(eligible) else int(len(fractions) - 1)
    rows = []
    for budget, layer in enumerate(costs):
        finite = layer[np.isfinite(layer)]
        rows.append({
            "b": int(budget),
            "reachability": float(fractions[budget]),
            "reachable_nodes": int(len(finite)),
            "cost_p50": float(np.quantile(finite, 0.5)) if len(finite) else None,
            "cost_p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
        })
    return {
        "max_b": int(len(fractions) - 1),
        "saturation_tolerance": float(tolerance),
        "r_sat": r_sat,
        "b_star": b_star,
        "budgets": rows,
    }
