"""Graph coverage, stitching, path, and target-provenance diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import betaincinv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from graphcritic.types import GraphArtifact, TargetSet, TransitionSet
from graphcritic.weights import step_cost


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    scalar = np.asarray(value).item()
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    scalar = float(scalar)
    return scalar if np.isfinite(scalar) else None


def _optimal_path_bridge_counts(artifact: GraphArtifact) -> NDArray[np.int64]:
    """Count bridges on each saved shortest-path successor chain."""
    n = artifact.nodes.n_nodes
    counts = np.zeros(n, dtype=np.int64)
    reachable = np.flatnonzero(np.isfinite(artifact.cost) & ~artifact.nodes.is_goal)
    if not reachable.size:
        return counts

    order = np.lexsort((artifact.weights, artifact.edges.dst, artifact.edges.src))
    src, dst, weight = artifact.edges.src[order], artifact.edges.dst[order], artifact.weights[order]
    kind = artifact.edges.kind[order]
    first = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
    src, dst, weight, kind = src[first], dst[first], weight[first], kind[first]
    pair_key = src * np.int64(n) + dst
    for node in reachable[np.argsort(artifact.cost[reachable], kind="stable")]:
        nxt = int(artifact.succ[node])
        pair = int(node) * n + nxt
        pos = int(np.searchsorted(pair_key, pair))
        if pos < pair_key.size and pair_key[pos] == pair:
            # The pair arrays keep the cheapest parallel edge. Since sort order
            # is (src, dst, weight), this is the edge selected by Dijkstra.
            counts[node] = counts[nxt] + int(kind[pos] == 1)
    return counts


def report(
    trans: TransitionSet,
    artifact: GraphArtifact,
    targets: TargetSet | None = None,
) -> dict[str, Any]:
    """Report graph usefulness, explicit bridge use, and target provenance."""
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
    reachable_rows = np.isfinite(current_cost)
    reachable_frac = float(np.mean(reachable_rows)) if n_rows else 0.0
    episode_starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
    episode_ends = np.r_[episode_starts[1:], n_rows]

    monotone_n = monotone_ok = 0
    rank_cost: list[float] = []
    rank_remaining: list[float] = []
    successful_episodes: list[tuple[NDArray[np.int64], int]] = []
    failed_episode_rows: list[NDArray[np.int64]] = []
    for start, end in zip(episode_starts, episode_ends, strict=True):
        rows = np.arange(start, end, dtype=np.int64)
        has_success = bool(np.any(trans.success_cur[start:end]) or np.any(trans.success[start:end]))
        if not has_success:
            failed_episode_rows.append(rows)
            continue
        event_steps = []
        row_success_cur = np.flatnonzero(trans.success_cur[start:end])
        row_success = np.flatnonzero(trans.success[start:end])
        if row_success_cur.size:
            event_steps.append(int(start + row_success_cur[0]))
        if row_success.size:
            event_steps.append(int(start + row_success[0] + 1))
        event_step = min(event_steps) if event_steps else int(end)
        prefix = rows[rows <= min(event_step, end - 1)]
        successful_episodes.append((prefix, event_step))
        cc = artifact.cost[artifact.nodes.assign_cur[prefix]]
        cn = artifact.cost[artifact.nodes.assign_next[prefix]]
        finite = np.isfinite(cc) & np.isfinite(cn)
        if np.any(finite):
            monotone_n += int(np.count_nonzero(finite))
            monotone_ok += int(np.count_nonzero(cn[finite] <= cc[finite] + 1e-9))
            rank_cost.extend(cc[finite].astype(np.float64).tolist())
            rank_remaining.extend((event_step - prefix[finite]).astype(np.float64).tolist())

    rank_corr = None
    if len(rank_cost) >= 2:
        with np.errstate(invalid="ignore"):
            rank_corr = _number(spearmanr(rank_cost, rank_remaining).statistic)

    adjacency = csr_matrix(
        (np.ones(n_edges, dtype=np.int8), (artifact.edges.src, artifact.edges.dst)),
        shape=(n_nodes, n_nodes),
    )
    n_components = int(connected_components(adjacency, directed=True, connection="weak", return_labels=False)) if n_nodes else 0
    path_switches = _optimal_path_bridge_counts(artifact)
    row_switches = path_switches[artifact.nodes.assign_cur]
    finite_switches = row_switches[reachable_rows]
    bridge_use_frac = float(np.mean(finite_switches > 0)) if finite_switches.size else 0.0

    stitch_diffs: list[float] = []
    c = step_cost(artifact.config.gamma)
    for rows, event_step in successful_episodes:
        remaining = (event_step - rows).astype(np.float64)
        graph_steps = current_cost[rows] / c
        finite = np.isfinite(graph_steps)
        stitch_diffs.extend((remaining[finite] - graph_steps[finite]).tolist())
    stitch_values = np.asarray(stitch_diffs, dtype=np.float64)
    bridge_mask = artifact.edges.kind == 1
    bridge_type = np.asarray(artifact.edges.bridge_type, dtype=np.int8)
    d_step = float(artifact.meta.get("d_step", 1.0))
    bridge_dist = artifact.edges.dist[bridge_mask].astype(np.float64) / max(d_step, 1e-12)
    bridge_bins = [dict(item) for item in artifact.meta.get("bridge_bin_evidence", [])]
    if artifact.config.bridge.cost.policy == "data_calibrated":
        n_bins = max(1, int(artifact.meta.get("bridge_n_bins", len(bridge_bins) or 1)))
        level = artifact.config.bridge.cost.delta / n_bins
        for item in bridge_bins:
            if "ell_lcb" not in item:
                ell = float(betaincinv(
                    artifact.config.bridge.cost.prior[0] + item["n_pos"],
                    artifact.config.bridge.cost.prior[1] + item["n_neg"],
                    level,
                ))
                item["ell_lcb"] = ell
                item["lambda_steps"] = float(-np.log(max(ell, 1e-300)) / c)
    q = lambda x, p: _number(np.percentile(x, p)) if len(x) else 0.0

    graph_result: dict[str, Any] = {
        "graph_kind": artifact.config.graph_kind,
        "n_rows": int(n_rows),
        "n_states": int(n_states),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "n_temporal_edges": int(np.count_nonzero(artifact.edges.kind == 0)),
        "n_bridge_edges": int(np.count_nonzero(bridge_mask)),
        "bridge_counts": (
            {
                "R": int(np.count_nonzero(bridge_mask & (bridge_type == 0))),
                "Graph Critic learned": int(np.count_nonzero(bridge_mask & (bridge_type == 1))),
                "legacy_unclassified": int(np.count_nonzero(bridge_mask & (bridge_type == -1))),
            }
            if artifact.meta.get("bridge_v07") else
            {
                "R": int(np.count_nonzero(bridge_mask & (bridge_type == 0))),
                "M": int(np.count_nonzero(bridge_mask & (bridge_type == 1))),
                "legacy_unclassified": int(np.count_nonzero(bridge_mask & (bridge_type == -1))),
            }
        ),
        "merge_ratio": float(n_states / max(1, n_nodes)),
        "n_goal_nodes": int(np.count_nonzero(artifact.nodes.is_goal)),
        "n_success_episodes": int(len(successful_episodes)),
        "n_failed_episodes": int(len(failed_episode_rows)),
        "reachable_frac": reachable_frac,
        "reachable_frac_failed_eps": (
            float(np.mean(np.isfinite(artifact.cost[artifact.nodes.assign_cur[np.concatenate(failed_episode_rows)]])))
            if failed_episode_rows and sum(map(len, failed_episode_rows)) else 0.0
        ),
        "bridge_use_frac": bridge_use_frac,
        "switches": {"p50": q(finite_switches, 50), "p95": q(finite_switches, 95), "max": int(np.max(finite_switches)) if finite_switches.size else 0},
        "stitch_gain": {
            "frac_gt_1_step": float(np.mean(stitch_values > 1.0)) if stitch_values.size else 0.0,
            "median_steps": q(stitch_values, 50),
        },
        "bridge_d_step_units": {
            "count": int(bridge_dist.size), "p50": q(bridge_dist, 50), "p95": q(bridge_dist, 95),
            "max": q(bridge_dist, 100),
        },
        "bridge_calibration": {
            "policy": artifact.meta.get("bridge_cost_policy"),
            "tau_model": artifact.meta.get("bridge_tau_model"),
            "tau_quantile": artifact.meta.get("bridge_tau_quantile"),
            "tau_horizon": artifact.meta.get("bridge_tau_horizon"),
            "tau_split_seed": artifact.meta.get("bridge_tau_split_seed"),
            "tau_far_exclusion_radius": artifact.meta.get("bridge_tau_far_exclusion_radius"),
            "tau_calibrated_quantile": artifact.meta.get("bridge_tau_calibrated_quantile"),
            "tau_p50": artifact.meta.get("bridge_tau_p50"),
            "tau_p90": artifact.meta.get("bridge_tau_p90"),
            "tau_max": artifact.meta.get("bridge_tau_max"),
            "tau_histogram": artifact.meta.get("bridge_tau_histogram", {}),
            "f_h_p10": artifact.meta.get("bridge_f_h_p10"),
            "f_h_p50": artifact.meta.get("bridge_f_h_p50"),
            "f_h_p90": artifact.meta.get("bridge_f_h_p90"),
            "tau_model_sha256": artifact.meta.get("tau_model_sha256"),
            "candidate_count": artifact.meta.get("bridge_candidates", 0),
            "retained_count": int(np.count_nonzero(bridge_mask)),
            "goal_edge_count": artifact.meta.get("bridge_goal_edge_count", 0),
            "goal_mode_edge_counts": artifact.meta.get("bridge_goal_mode_edge_counts", {}),
            "pruned_over_lambda_cap": artifact.meta.get("bridge_pruned_over_lambda_cap", 0),
            "lambda_cap_steps": artifact.config.bridge.cost.lambda_cap_steps,
            "radius_abs": artifact.meta.get("bridge_radius_abs"),
            "knn_backend": artifact.meta.get("knn_backend", []),
            "distance_bin_edges": artifact.meta.get("bridge_bin_edges", []),
            "distance_bins": bridge_bins,
            "passive_diff_epsilon_units": artifact.meta.get("bridge_passive_diff_epsilon_units"),
            "nn_passive_cross": artifact.meta.get("nn_passive_cross"),
            "kinematic_violation": artifact.meta.get("bridge_tau_kinematic_violation"),
            "passive_violation": artifact.meta.get("bridge_tau_passive_violation"),
            "tau_by_type": {
                label: {
                    "n": int(np.count_nonzero(bridge_mask & (bridge_type == code))),
                    "p50": q(artifact.edges.kappa[bridge_mask & (bridge_type == code)], 50),
                    "p95": q(artifact.edges.kappa[bridge_mask & (bridge_type == code)], 95),
                }
                for label, code in (("R", 0), ("M", 1))
            },
            "puzzle_press_calibration": artifact.meta.get("bridge_puzzle_press_calibration"),
            "puzzle_tau_real": artifact.meta.get("bridge_puzzle_tau_real"),
            "puzzle_bucket_sizes": artifact.meta.get("bridge_puzzle_bucket_sizes"),
            "puzzle_candidate_search": artifact.meta.get("bridge_candidate_search"),
            "puzzle_free_state_fraction": artifact.meta.get("bridge_free_state_fraction"),
            "puzzle_tau_max_steps": artifact.meta.get("bridge_tau_max_steps"),
            "puzzle_success_rho": artifact.meta.get("bridge_success_rho"),
            "puzzle_lambda_rho_steps": artifact.meta.get("bridge_lambda_rho_steps"),
        },
        "modes": {
            "count": artifact.meta.get("n_mode_values", 1),
            "segment_count": artifact.meta.get("segment_count", 0),
            "segment_length_p50": artifact.meta.get("segment_length_p50", 0.0),
            "segment_length_p95": artifact.meta.get("segment_length_p95", 0.0),
            "segment_length_max": artifact.meta.get("segment_length_max", 0),
            "goal_mode_distribution": artifact.meta.get("goal_mode_distribution", {}),
        },
        "d_step": d_step,
        "build_seconds": artifact.meta.get("build_seconds", 0.0),
        "peak_rss_mb": artifact.meta.get("peak_rss_mb", 0.0),
        "monotonicity_success_eps": float(monotone_ok / monotone_n) if monotone_n else 0.0,
        "rank_corr_success_eps": rank_corr,
        # Keep old names for existing dashboards; their success-only scope is explicit above.
        "monotonicity": float(monotone_ok / monotone_n) if monotone_n else 0.0,
        "rank_corr": rank_corr,
        "path_exec_rate": artifact.meta.get("path_exec_rate"),
        "path_exec_switch_le_3": artifact.meta.get("path_exec_switch_le_3"),
        "path_exec_by_switches": artifact.meta.get("path_exec_by_switches", {}),
        "path_exec_n": artifact.meta.get("path_exec_n", 0),
        "kappa": {
            "mean": float(np.mean(artifact.edges.kappa)) if n_edges else 0.0,
            "p50": q(artifact.edges.kappa, 50), "p95": q(artifact.edges.kappa, 95),
            "max": int(np.max(artifact.edges.kappa)) if n_edges else 0,
        },
        "n_components": n_components,
    }
    if targets is not None:
        valid = targets.valid
        source = targets.source
        valid_count = int(np.count_nonzero(valid))
        mix = {
            str(code): float(np.count_nonzero(valid & (source == code)) / max(1, valid_count))
            for code in (0, 1, 2, 4)
        }
        graph_target = valid & (source == 1)
        cap_pool = valid & (source != 0)
        valid_y = targets.y[valid].astype(np.float64)
        cap_value = targets.h + float(artifact.meta["c_max_steps"])
        if targets.unit == "nats":
            cap_value *= c
        graph_y = targets.y[graph_target].astype(np.float64)
        graph_result["targets"] = {
            "valid_frac": float(np.mean(valid)) if n_rows else 0.0,
            "y_mean": float(np.mean(valid_y)) if valid_y.size else 0.0,
            "y_p50": q(valid_y, 50), "y_p95": q(valid_y, 95),
            "y_max": float(np.max(valid_y)) if valid_y.size else 0.0,
            "frac_at_cmax": (
                float(np.mean(valid_y >= cap_value - 1e-6)) if valid_y.size else 0.0
            ) if targets.kind == "cost" else None,
            "target_mix": mix,
            "y_graph_q": {"p5": q(graph_y, 5), "p50": q(graph_y, 50), "p95": q(graph_y, 95)},
            "valid_non_success_cap_frac": (
                float(np.count_nonzero(valid & (source == 2)) / np.count_nonzero(cap_pool))
                if np.count_nonzero(cap_pool) else 0.0
            ),
        }
        if targets.kind == "regret":
            train_rows = valid & targets.loss_mask
            regret = targets.y[train_rows].astype(np.float64)
            max_value = targets.regret_max * (c if targets.unit == "nats" else 1.0)
            starts = np.arange(max(0, n_rows - targets.h + 1), dtype=np.int64)
            if starts.size:
                endpoint_rows = starts + targets.h - 1
                start_cost = artifact.cost[artifact.nodes.assign_cur[starts]] / c
                end_cost = artifact.cost[artifact.nodes.assign_next[endpoint_rows]] / c
                cmax = float(artifact.meta["c_max_steps"])
                start_reachable = np.isfinite(start_cost)
                end_reachable = np.isfinite(end_cost)
                start_finite = start_reachable & (start_cost < cmax)
                audited = valid[starts]
                impossible = audited & ~start_reachable & end_reachable
                success_steps = np.zeros(starts.size, dtype=np.float64)
                window = np.stack(
                    [trans.success[offset : offset + starts.size] for offset in range(targets.h)],
                    axis=1,
                )
                has_success = np.any(window, axis=1)
                success_steps[has_success] = np.argmax(window[has_success], axis=1) + 1
                success_steps[trans.success_cur[starts]] = 0.0
                success_row = (source[starts] == 0) & audited
                success_violation = (success_row & ~start_finite) | (
                    success_row & (start_cost > success_steps + 1e-8)
                )
                finite_edge = (source[starts] == 1) & audited
                temporal_violation = (
                    finite_edge & start_reachable & end_reachable
                    & (start_cost > targets.h + end_cost + 1e-8)
                )
                property_violations = int(np.count_nonzero(impossible | success_violation | temporal_violation))
            else:
                property_violations = 0
            graph_result["targets"].update({
                "kind": "regret",
                "regret_loss_mask_frac": float(np.mean(targets.loss_mask[valid])) if valid_count else 0.0,
                "regret_excluded_both_capped": int(np.count_nonzero(valid & (source == 4))),
                "regret_p50": q(regret, 50),
                "regret_p90": q(regret, 90),
                "regret_zero_frac": float(np.mean(regret <= 1e-6)) if regret.size else 0.0,
                "regret_rmax_frac": float(np.mean(regret >= max_value - 1e-5)) if regret.size else 0.0,
                "regret_max": float(max_value),
                "property_1_violation_count": property_violations,
            })
        else:
            graph_result["targets"].update({"kind": "cost", "loss_mask_frac": float(np.mean(targets.loss_mask[valid])) if valid_count else 0.0})
    return graph_result
