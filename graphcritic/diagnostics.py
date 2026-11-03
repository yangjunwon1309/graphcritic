"""Graph monotonicity, coverage, connectivity, and rank diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from graphcritic.types import GraphArtifact, TargetSet, TransitionSet
from graphcritic.weights import step_cost


def _python_number(value: Any) -> int | float | None:
    if value is None:
        return None
    scalar = np.asarray(value).item()
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    scalar = float(scalar)
    return scalar if np.isfinite(scalar) else None


def report(
    trans: TransitionSet,
    artifact: GraphArtifact,
    targets: TargetSet | None = None,
) -> dict[str, Any]:
    """Summarize graph coverage, success-cost ordering, and target statistics."""
    trans.validate()
    artifact.validate(trans)
    if targets is not None:
        targets.validate(trans)
        if targets.graph_hash != artifact.content_hash:
            raise ValueError("targets were generated from a different graph artifact")
        if targets.dataset_id != artifact.meta["dataset_id"]:
            raise ValueError("target and graph dataset_id values differ")

    n_rows = trans.n_rows
    n_states = n_rows + int(np.count_nonzero(trans.terminal))
    n_nodes = artifact.nodes.n_nodes
    n_edges = artifact.edges.n_edges
    current_cost = artifact.cost[artifact.nodes.assign_cur]
    reachable_frac = float(np.mean(np.isfinite(current_cost))) if n_rows else 0.0

    episode_starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
    episode_ends = np.r_[episode_starts[1:], n_rows]
    success_episodes = 0
    monotone_n = 0
    monotone_ok = 0
    rank_cost: list[float] = []
    rank_remaining: list[float] = []
    for start, end in zip(episode_starts, episode_ends, strict=True):
        rows = np.arange(start, end, dtype=np.int64)
        success_rows = np.flatnonzero(trans.success[start:end])
        if success_rows.size == 0:
            continue
        success_episodes += 1
        first_success = int(start + success_rows[0])
        prefix_rows = rows[rows <= first_success]
        if prefix_rows.size == 0:
            continue
        cc = artifact.cost[artifact.nodes.assign_cur[prefix_rows]]
        cn = artifact.cost[artifact.nodes.assign_next[prefix_rows]]
        finite = np.isfinite(cc) & np.isfinite(cn)
        if np.any(finite):
            monotone_n += int(np.count_nonzero(finite))
            monotone_ok += int(np.count_nonzero(cn[finite] <= cc[finite] + 1e-9))
        rank_cost.extend(cc[np.isfinite(cc)].astype(np.float64).tolist())
        rank_remaining.extend((first_success - prefix_rows + 1)[np.isfinite(cc)].astype(np.float64).tolist())

    corr: float | None = None
    if len(rank_cost) >= 2:
        with np.errstate(invalid="ignore"):
            corr_result = spearmanr(rank_cost, rank_remaining).statistic
        corr = _python_number(corr_result)

    if n_edges:
        kappas = artifact.edges.kappa
        kappa_stats: dict[str, int | float] = {
            "mean": float(np.mean(kappas)),
            "p50": float(np.percentile(kappas, 50)),
            "p95": float(np.percentile(kappas, 95)),
            "max": int(np.max(kappas)),
        }
    else:
        kappa_stats = {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0}

    if n_nodes:
        adjacency = csr_matrix(
            (np.ones(n_edges, dtype=np.int8), (artifact.edges.src, artifact.edges.dst)),
            shape=(n_nodes, n_nodes),
        )
        n_components = int(connected_components(adjacency, directed=True, connection="weak", return_labels=False))
    else:
        n_components = 0

    result: dict[str, Any] = {
        "n_rows": int(n_rows),
        "n_states": int(n_states),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "merge_ratio": float(n_states / n_nodes) if n_nodes else 0.0,
        "n_goal_nodes": int(np.count_nonzero(artifact.nodes.is_goal)),
        "n_success_episodes": int(success_episodes),
        "reachable_frac": reachable_frac,
        "monotonicity": float(monotone_ok / monotone_n) if monotone_n else 0.0,
        "rank_corr": corr,
        "kappa": kappa_stats,
        "n_components": n_components,
    }
    if targets is not None:
        valid_y = targets.y[targets.valid].astype(np.float64)
        if targets.unit == "nats":
            cap_value = step_cost(artifact.config.gamma) * (
                targets.h + float(artifact.meta["c_max_steps"])
            )
        else:
            cap_value = targets.h + float(artifact.meta["c_max_steps"])
        clipped = valid_y >= cap_value - 1e-6
        if valid_y.size:
            targets_report: dict[str, int | float] = {
                "valid_frac": float(np.mean(targets.valid)),
                "y_mean": float(np.mean(valid_y)),
                "y_p50": float(np.percentile(valid_y, 50)),
                "y_p95": float(np.percentile(valid_y, 95)),
                "y_max": float(np.max(valid_y)),
                "frac_at_cmax": float(np.mean(clipped)),
            }
        else:
            targets_report = {
                "valid_frac": float(np.mean(targets.valid)) if n_rows else 0.0,
                "y_mean": 0.0,
                "y_p50": 0.0,
                "y_p95": 0.0,
                "y_max": 0.0,
                "frac_at_cmax": 0.0,
            }
        result["targets"] = targets_report
    return result
