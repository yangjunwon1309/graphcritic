"""Offline lower-bound audit of retained v0.8 puzzle bridges.

This diagnostic checks whether stored quasimetric distances are shorter than
the minimum button-board transition time supported by the offline data. It
does not simulate or execute bridges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.modes import state_episode_time
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.object_state import (
    frame_object_features_from_checkpoint,
    load_or_build_object_pair_index,
)
from graphcritic.bridges.windows import puzzle_state_windows, puzzle_window_candidate_nodes
from graphcritic.puzzle import minimum_press_solution, toggle_matrix
from graphcritic.types import GraphArtifact, SchemaError, TransitionSet


_BOARD_LAYOUTS = {9: (3, 3), 16: (4, 4), 20: (4, 5)}
_NP_GROUPS = ("0", "1", "2", "3", "≥4", "∞")


def _candidate_only_mode(args) -> str:
    """Validate candidate-only arguments without opening optional checkpoints."""
    if args.distance_checkpoint is not None:
        if args.object_event_checkpoint is not None or args.reference_checkpoint is None:
            raise SystemExit(
                "--distance-checkpoint requires --reference-checkpoint and excludes --object-event-checkpoint"
            )
        return "distance"
    if args.object_event_checkpoint is None:
        raise SystemExit("--candidates-only requires --object-event-checkpoint or --distance-checkpoint")
    return "object_event"


def _distance_group_report(press_counts: np.ndarray, dplus: np.ndarray,
                           dtheta: np.ndarray) -> dict[str, dict[str, Any]]:
    """Summarize d+ and reference d_theta for the v1.0 press-count bins."""
    counts = np.asarray(press_counts, dtype=np.int64)
    plus = np.asarray(dplus, dtype=np.float32)
    theta = np.asarray(dtheta, dtype=np.float32)
    if counts.shape != plus.shape or counts.shape != theta.shape:
        raise ValueError("press counts and distance arrays must align")
    masks = {
        "0": counts == 0,
        "1": counts == 1,
        "2": counts == 2,
        "3": counts == 3,
        "≥4": counts >= 4,
        "∞": counts < 0,
    }
    report = {}
    for label, mask in masks.items():
        plus_values = plus[mask]
        theta_values = theta[mask]
        report[label] = {
            "count": int(np.count_nonzero(mask)),
            "d_plus_p10": float(np.quantile(plus_values, 0.1)) if len(plus_values) else None,
            "d_plus_p50": float(np.quantile(plus_values, 0.5)) if len(plus_values) else None,
            "d_plus_p90": float(np.quantile(plus_values, 0.9)) if len(plus_values) else None,
            "d_theta_p50": float(np.quantile(theta_values, 0.5)) if len(theta_values) else None,
            "fraction_d_plus_le_20": float(np.mean(plus_values <= 20.0)) if len(plus_values) else None,
        }
    return report


def _c0_optimistic_mask(pairs: np.ndarray, c0: np.ndarray, *, minimum_gain: float = 10.0) -> np.ndarray:
    """Match build_judged_graph's finite-destination, C0 gain filter."""
    pairs = np.asarray(pairs, dtype=np.int64)
    costs = np.asarray(c0, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("candidate pairs must have shape (N,2)")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= len(costs)):
        raise ValueError("candidate pair node is outside the G0 cost vector")
    with np.errstate(invalid="ignore"):
        gain = costs[pairs[:, 0]] - costs[pairs[:, 1]]
    return np.isfinite(costs[pairs[:, 1]]) & (gain >= float(minimum_gain))


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _board_change_intervals(
    trans: TransitionSet, button_states: np.ndarray
) -> np.ndarray:
    """Return gaps between successive board changes, within each episode."""
    episode, times = state_episode_time(trans)
    boards = np.asarray(button_states)
    if boards.ndim != 2 or len(boards) != len(episode):
        raise SchemaError("button_states must align with the full transition state table")
    gaps: list[np.ndarray] = []
    for episode_id in np.unique(episode):
        state_ids = np.flatnonzero(episode == episode_id)
        state_ids = state_ids[np.argsort(times[state_ids], kind="stable")]
        if state_ids.size < 3:
            continue
        changed = np.any(boards[state_ids[1:]] != boards[state_ids[:-1]], axis=1)
        change_times = times[state_ids[1:]][changed]
        if change_times.size > 1:
            gaps.append(np.diff(change_times).astype(np.float64))
    if not gaps:
        raise SchemaError(
            "cannot estimate button interval: need consecutive board changes in the data"
        )
    intervals = np.concatenate(gaps)
    if np.any(intervals <= 0):
        raise SchemaError("board-change intervals must be positive")
    return intervals


def _minimum_press_count(
    difference: np.ndarray,
    matrix: np.ndarray,
    memo: dict[int, int | None] | None = None,
) -> int | None:
    """Return the exact minimum GF(2) solution weight, or None if unreachable."""
    rhs = np.asarray(difference, dtype=np.uint8).reshape(-1)
    key = sum(int(bit) << i for i, bit in enumerate(rhs))
    if memo is not None and key in memo:
        return memo[key]
    try:
        solution = minimum_press_solution(matrix, rhs)
    except SchemaError:
        count = None
    else:
        count = int(np.count_nonzero(solution))
    if memo is not None:
        memo[key] = count
    return count


def _distance_checkpoint_pairs(
    checkpoint: Path,
    *,
    trans,
    dataset_id: str,
    states: np.ndarray,
    node_ids: np.ndarray,
    pairs: np.ndarray,
    cache_dir: Path,
    label: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Evaluate a saved state quasimetric on cached candidate pairs."""
    from experiments.v08_runtime import (
        load_quasimetric_checkpoint,
        make_embedding_cache_predictor,
    )

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != dataset_id or int(metadata.get("window", -1)) != 5:
        raise ValueError(f"{label} checkpoint dataset or window does not match the audit")
    params_bytes = (checkpoint / "params.msgpack").read_bytes()
    checkpoint_sha = hashlib.sha256(params_bytes).hexdigest()
    if checkpoint_sha != metadata.get("params_sha256"):
        raise ValueError(f"{label} checkpoint parameter hash mismatch")
    model, params, metadata = load_quasimetric_checkpoint(
        checkpoint, input_dim=states.shape[1], expected_dataset_id=dataset_id,
    )
    cache_path = cache_dir / (
        f"embeddings-{checkpoint_sha[:16]}-{_array_sha256(node_ids)[:16]}.npy"
    )
    predict, cache_report = make_embedding_cache_predictor(
        model, params, states[node_ids], node_ids,
        params_sha256=checkpoint_sha, dataset_id=dataset_id,
        cache_path=cache_path, pair_batch_size=4096,
    )
    distances = predict(pairs)
    del predict, params, model
    try:
        import gc
        import jax
        gc.collect()
        jax.clear_caches()
    except Exception:  # pragma: no cover - best-effort accelerator cleanup
        pass
    return distances, metadata, cache_report


def _candidate_distance_audit(args, trans, trans_meta, sim_state, boards) -> dict[str, Any]:
    """Run the v1.0 full-candidate P2/P3 distance audit."""
    if args.reference_checkpoint is None:
        raise SystemExit("--distance-checkpoint requires --reference-checkpoint")
    if args.object_event_checkpoint is not None:
        raise SystemExit("choose --distance-checkpoint or --object-event-checkpoint, not both")
    with np.load(args.candidate_cache, allow_pickle=False) as archive:
        candidate_pairs = np.asarray(archive["pairs"], dtype=np.int64)
        candidate_nodes = np.asarray(archive["node_ids"], dtype=np.int64)
        if str(archive["dataset_id"].item()) != trans_meta["dataset_id"]:
            raise SchemaError("candidate cache dataset ID does not match the audit transitions")
        if int(archive["window"].item()) != 5 or int(archive["k"].item()) != 32:
            raise SchemaError("distance audit requires the v0.8 window-5, k=32 candidate cache")
    model_data = prepare_puzzle_model_data(trans, sim_state, seed=0)
    frame = np.concatenate((model_data.state_cont, model_data.state_bits), axis=1).astype(np.float32)
    state_windows = puzzle_state_windows(
        frame, trans, model_data.terminal_state_index, window=5,
    )
    terminal_rows = np.flatnonzero(trans.terminal)
    state_goals = np.concatenate((trans.success_cur, trans.success[terminal_rows]))
    expected_nodes, _, _ = puzzle_window_candidate_nodes(trans, state_goals, window=5)
    if not np.array_equal(expected_nodes, candidate_nodes):
        raise SchemaError("candidate cache node order differs from the v0.8 window-5 rule")
    state_episodes = np.concatenate((trans.episode_id, trans.episode_id[terminal_rows]))
    if np.any(state_episodes[candidate_pairs[:, 0]] == state_episodes[candidate_pairs[:, 1]]):
        raise SchemaError("candidate distance audit requires cross-episode pairs")
    pairs, matched_idx, unmatched_idx, pair_index_path, pair_index_hit = load_or_build_object_pair_index(
        args.candidate_cache, boards, dataset_id=trans_meta["dataset_id"],
    )
    if not np.array_equal(pairs, candidate_pairs):
        raise SchemaError("object match index cache is keyed to a different pair array")

    distances, distance_meta, distance_cache = _distance_checkpoint_pairs(
        args.distance_checkpoint,
        trans=trans, dataset_id=trans_meta["dataset_id"],
        states=state_windows, node_ids=candidate_nodes,
        pairs=pairs, cache_dir=args.out.parent, label="dplus",
    )

    # The candidate audit command is the machine-readable S2 gate report.  The
    # separate eval_quasimetric_chunk.py command in RUN_v1.0 remains useful for
    # its richer diagnostics, while this computes the exact held-out d+ > 6.25
    # criterion so all four S2 gates can be summarized together.
    from experiments.train_quasimetric import _targets_at_offset
    from experiments.v08_runtime import (
        load_quasimetric_checkpoint,
        make_distance_predictor,
    )

    heldout_data = prepare_puzzle_model_data(trans, sim_state, seed=0)
    chunk_rows, chunk_endpoints = _targets_at_offset(
        heldout_data.heldout_rows, heldout_data, 5,
    )
    chunk_model, chunk_params, _ = load_quasimetric_checkpoint(
        args.distance_checkpoint,
        input_dim=state_windows.shape[1],
        expected_dataset_id=trans_meta["dataset_id"],
    )
    chunk_predict = make_distance_predictor(chunk_model, chunk_params, batch_size=8192)
    chunk_distances = chunk_predict(
        state_windows[chunk_rows], state_windows[chunk_endpoints]
    )
    chunk_violation_rate = (
        float(np.mean(chunk_distances > 6.25)) if len(chunk_distances) else None
    )
    del chunk_predict, chunk_params, chunk_model
    try:
        import gc
        import jax
        gc.collect()
        jax.clear_caches()
    except Exception:  # pragma: no cover - best-effort accelerator cleanup
        pass

    counts = np.empty(len(pairs), dtype=np.int16)
    memo: dict[int, int | None] = {}
    toggle = toggle_matrix(*_BOARD_LAYOUTS[boards.shape[1]])
    for index, (source, destination) in enumerate(pairs):
        counts[index] = -1 if (count := _minimum_press_count(
            boards[int(source)] ^ boards[int(destination)], toggle, memo
        )) is None else count

    reference_distances, reference_meta, reference_cache = _distance_checkpoint_pairs(
        args.reference_checkpoint,
        trans=trans, dataset_id=trans_meta["dataset_id"],
        states=state_windows, node_ids=candidate_nodes,
        pairs=pairs, cache_dir=args.out.parent, label="dtheta-reference",
    )
    del reference_meta
    by_group = _distance_group_report(counts, distances, reference_distances)
    np_ge2_mask = (counts >= 2) | (counts < 0)
    np_ge2_values = distances[np_ge2_mask]
    np_ge2_accept = float(np.mean(np_ge2_values <= 20.0)) if len(np_ge2_values) else None
    acceptance_by_press_class = {
        "n_p_0": by_group["0"]["fraction_d_plus_le_20"],
        "n_p_1": by_group["1"]["fraction_d_plus_le_20"],
        "n_p_ge_2_including_infinity": np_ge2_accept,
    }

    # The command intentionally has no extra G0 argument. Resolve the runbook's
    # sibling D3/g0/graph.npz from the shared candidate-cache location.
    g0_path = args.candidate_cache.parent.parent / "g0" / "graph.npz"
    if not g0_path.is_file():
        raise FileNotFoundError(f"cannot locate the G0 graph beside candidate cache: {g0_path}")
    g0_graph = load_graph(g0_path, trans=trans)
    if len(g0_graph.cost) != len(g0_graph.nodes.is_goal):
        raise SchemaError("G0 costs must align with its node table")
    if np.any(g0_graph.edges.kind != 0):
        raise SchemaError("optimistic audit expects the temporal-only G0 graph")
    from graphcritic.bridges.insert import layered_reverse_dijkstra
    from graphcritic.weights import step_cost
    temporal = g0_graph.edges.kind == 0
    base_c = step_cost(g0_graph.config.gamma)
    c0_solution = layered_reverse_dijkstra(
        g0_graph.nodes.n_nodes,
        np.flatnonzero(g0_graph.nodes.is_goal),
        g0_graph.edges.src[temporal].astype(np.int64),
        g0_graph.edges.dst[temporal].astype(np.int64),
        (g0_graph.weights[temporal] / base_c).astype(np.float64),
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64), b_max=0,
    )
    c0_steps = c0_solution.cost_steps[0]
    optimistic_mask = _c0_optimistic_mask(pairs, c0_steps, minimum_gain=10.0)
    optimistic_report = _distance_group_report(
        counts[optimistic_mask], distances[optimistic_mask],
        reference_distances[optimistic_mask],
    )
    optimistic_counts = counts[optimistic_mask]
    optimistic_distances = distances[optimistic_mask]
    optimistic_ge2 = (optimistic_counts >= 2) | (optimistic_counts < 0)
    optimistic_acceptance = {
        "n_p_0": optimistic_report["0"]["fraction_d_plus_le_20"],
        "n_p_1": optimistic_report["1"]["fraction_d_plus_le_20"],
        "n_p_ge_2_including_infinity": (
            float(np.mean(optimistic_distances[optimistic_ge2] <= 20.0))
            if np.any(optimistic_ge2) else None
        ),
    }

    # Candidate pairs are cross-episode; retaining pairs whose endpoints both
    # belong to held-out episodes gives an episode-disjoint P2 audit.
    heldout_state_mask = np.isin(state_episodes, model_data.heldout_episodes)
    heldout_pair_mask = heldout_state_mask[pairs[:, 0]] & heldout_state_mask[pairs[:, 1]]
    matched_mask = np.zeros(len(pairs), dtype=np.bool_)
    matched_mask[matched_idx] = True
    heldout_match_indices = np.flatnonzero(heldout_pair_mask & matched_mask)
    heldout_match_pairs = pairs[heldout_match_indices]
    reference_forward = reference_reverse = np.empty(0, dtype=np.float32)
    dplus_reverse = np.empty(0, dtype=np.float32)
    if len(heldout_match_pairs):
        reference_reverse, _, reference_reverse_cache = _distance_checkpoint_pairs(
            args.reference_checkpoint,
            trans=trans, dataset_id=trans_meta["dataset_id"],
            states=state_windows, node_ids=candidate_nodes,
            pairs=heldout_match_pairs[:, ::-1], cache_dir=args.out.parent,
            label="dtheta-reference-reverse",
        )
        reference_forward = reference_distances[heldout_match_indices]
        dplus_reverse, _, dplus_reverse_cache = _distance_checkpoint_pairs(
            args.distance_checkpoint,
            trans=trans, dataset_id=trans_meta["dataset_id"],
            states=state_windows, node_ids=candidate_nodes,
            pairs=heldout_match_pairs[:, ::-1], cache_dir=args.out.parent,
            label="dplus-reverse",
        )
        distance_cache["reverse_pair_embedding_cache"] = dplus_reverse_cache
        reference_cache["reverse_pair_embedding_cache"] = reference_reverse_cache
    if len(heldout_match_pairs):
        dplus_directed = np.concatenate((distances[heldout_match_indices], dplus_reverse))
        dref_directed = np.concatenate((reference_forward, reference_reverse))
        p2_violation_rate = float(np.mean(dplus_directed > dref_directed + 1.25))
    else:
        p2_violation_rate = None

    d0 = by_group["0"]["fraction_d_plus_le_20"]
    d1 = by_group["1"]["d_plus_p50"]
    dge2 = float(np.median(np.concatenate((
        distances[counts >= 2], distances[counts < 0]
    )))) if np.any(np_ge2_mask) else None
    gates = {
        "chunk_violation_passed": bool(
            chunk_violation_rate is not None and chunk_violation_rate <= 0.05
        ),
        "p2_heldout_violation_at_most_5_percent": bool(p2_violation_rate is not None and p2_violation_rate <= 0.05),
        "median_order_np0_lt_np1_lt_np_ge2": bool(
            by_group["0"]["d_plus_p50"] is not None and d1 is not None and dge2 is not None
            and by_group["0"]["d_plus_p50"] < d1 < dge2
        ),
        "np_ge2_accept_rate_at_most_1_percent": bool(
            np_ge2_accept is not None and np_ge2_accept <= 0.01
        ),
    }
    gates["s2_all_passed"] = bool(all(gates.values()))
    gate_values = {
        "chunk_violation_rate": chunk_violation_rate,
        "p2_heldout_violation_rate_both_directions": p2_violation_rate,
        "median_d_plus_n_p0": by_group["0"]["d_plus_p50"],
        "median_d_plus_n_p1": by_group["1"]["d_plus_p50"],
        "median_d_plus_n_p_ge2_including_infinity": dge2,
        "n_p_ge2_fraction_d_plus_le_20": np_ge2_accept,
    }
    return {
        "diagnostic": "v1.0_dplus_candidate_audit",
        "dataset_id": trans_meta["dataset_id"],
        "candidate_pair_count": int(len(pairs)),
        "candidate_node_count": int(len(candidate_nodes)),
        "object_match_index_cache": str(pair_index_path),
        "object_match_index_cache_hit": bool(pair_index_hit),
        "matched_pair_count": int(len(matched_idx)),
        "unmatched_pair_count": int(len(unmatched_idx)),
        "distance_checkpoint": str(args.distance_checkpoint),
        "distance_checkpoint_sha256": distance_meta["params_sha256"],
        "distance_embedding_cache": distance_cache,
        "reference_checkpoint": str(args.reference_checkpoint),
        "g0_graph": str(g0_path),
        "c0_optimistic_minimum_gain_steps": 10.0,
        "c0_optimistic_candidate_count": int(np.count_nonzero(optimistic_mask)),
        "reference_embedding_cache": reference_cache,
        "heldout_matched_pair_count": int(len(heldout_match_pairs)),
        "heldout_p2_violation_rate_both_directions": p2_violation_rate,
        "heldout_chunk_constraint": {
            "pair_count": int(len(chunk_distances)),
            "threshold": 6.25,
            "violation_rate": chunk_violation_rate,
        },
        "press_count_groups": by_group,
        "acceptance_fraction_by_press_class": acceptance_by_press_class,
        "c0_optimistic_press_count_groups": optimistic_report,
        "c0_optimistic_acceptance_fraction_by_press_class": optimistic_acceptance,
        "np0_d_le_20_recall": d0,
        "np1_d_le_20_fraction": by_group["1"]["fraction_d_plus_le_20"],
        "np_ge2_d_le_20_false_accept_rate": np_ge2_accept,
        "gate_values": gate_values,
        "gates": gates,
    }


def _node_state_indices(graph: GraphArtifact, trans: TransitionSet) -> np.ndarray:
    """Map each graph node's representative reference to a state-table row."""
    rows = trans.n_rows
    terminal_rows = np.flatnonzero(trans.terminal)
    next_state = np.empty(rows, dtype=np.int64)
    nonterminal_rows = np.flatnonzero(~trans.terminal)
    next_state[nonterminal_rows] = nonterminal_rows + 1
    next_state[terminal_rows] = rows + np.arange(terminal_rows.size, dtype=np.int64)

    refs = graph.nodes.rep_ref
    source_rows = refs[:, 0]
    roles = refs[:, 1]
    if np.any(source_rows < 0) or np.any(source_rows >= rows):
        raise SchemaError("puzzle board audit requires graph nodes mapped to transition rows")
    if np.any(~np.isin(roles, (0, 1))):
        raise SchemaError("graph node representative roles must be current or next state")
    result = source_rows.copy()
    next_mask = roles == 1
    result[next_mask] = next_state[source_rows[next_mask]]
    return result


def _np_group_masks(press_counts: np.ndarray) -> dict[str, np.ndarray]:
    finite = press_counts >= 0
    return {
        "0": finite & (press_counts == 0),
        "1": finite & (press_counts == 1),
        "2": finite & (press_counts == 2),
        "3": finite & (press_counts == 3),
        "≥4": finite & (press_counts >= 4),
        "∞": ~finite,
    }


def _candidate_event_report(press_counts: np.ndarray, event_scores: np.ndarray) -> dict:
    counts = np.asarray(press_counts, dtype=np.int64)
    scores = np.asarray(event_scores, dtype=np.float32)
    if counts.shape != scores.shape:
        raise ValueError("candidate press counts and event scores must align")
    finite = counts >= 0
    groups = {
        "0": finite & (counts == 0),
        "1": finite & (counts == 1),
        "≥2": finite & (counts >= 2),
        "∞": ~finite,
    }
    by_group = {
        label: {
            "count": int(np.count_nonzero(mask)),
            **(_quantiles(scores[mask]) or {"p10": None, "p50": None, "p90": None}),
        }
        for label, mask in groups.items()
    }
    predicted = scores <= 1.5
    actual = finite & (counts <= 1)
    false_positive = predicted & ~actual
    hard_negative = ~actual
    hard_negative_count = int(np.count_nonzero(hard_negative))
    false_positive_count = int(np.count_nonzero(false_positive & hard_negative))
    median0 = by_group["0"]["p50"]
    median1 = by_group["1"]["p50"]
    median2 = by_group["≥2"]["p50"]
    ordered_medians = bool(
        median0 is not None and median1 is not None and median2 is not None
        and median0 < median1 < median2
    )
    fpr = false_positive_count / hard_negative_count if hard_negative_count else None
    return {
        "candidate_pair_count": int(len(scores)),
        "e_phi_by_minimum_press_count": by_group,
        "threshold_1_5_vs_n_p_le_1": {
            "true_positive": int(np.count_nonzero(predicted & actual)),
            "false_positive": int(np.count_nonzero(false_positive)),
            "false_negative": int(np.count_nonzero(~predicted & actual)),
            "true_negative": int(np.count_nonzero(~predicted & ~actual)),
        },
        "false_positive_rate_for_n_p_ge_2": fpr,
        "median_order_0_lt_1_lt_ge2": ordered_medians,
        "d2_5_gate_passed": bool(ordered_medians and fpr is not None and fpr <= 0.01),
        "median_gate_note": "ordering is a machine-readable proxy for the spec's approximate 0, 1, >=2 separation",
    }


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if not len(values):
        return None
    return {
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def analyze_bridge_arrays(
    bridge_src: np.ndarray,
    bridge_dst: np.ndarray,
    distance_steps: np.ndarray,
    *,
    node_state_indices: np.ndarray,
    button_states: np.ndarray,
    episode: np.ndarray,
    times: np.ndarray,
    min_button_interval: float,
    toggle: np.ndarray,
    event_scores: np.ndarray | None = None,
) -> dict:
    """Summarize GF(2) button effort and distance lower-bound violations."""
    src = np.asarray(bridge_src, dtype=np.int64)
    dst = np.asarray(bridge_dst, dtype=np.int64)
    distance = np.asarray(distance_steps, dtype=np.float64)
    if src.ndim != 1 or dst.shape != src.shape or distance.shape != src.shape:
        raise ValueError("bridge endpoints and distances must be aligned vectors")
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise SchemaError("bridge judge distances must be finite and non-negative")
    node_states = np.asarray(node_state_indices, dtype=np.int64)
    boards = np.asarray(button_states)
    if (node_states.ndim != 1 or boards.ndim != 2 or len(episode) != len(times)
            or len(boards) != len(episode)):
        raise ValueError("state-table mappings, boards, episodes, and times must align")
    if boards.shape[1] != toggle.shape[1] or toggle.shape[0] != toggle.shape[1]:
        raise ValueError("button-board width must match the square toggle matrix")
    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= len(node_states)) or np.any(dst >= len(node_states)):
        raise SchemaError("bridge endpoint lies outside the graph node table")
    if event_scores is not None:
        event_scores = np.asarray(event_scores, dtype=np.float32)
        if event_scores.shape != src.shape or np.any(~np.isfinite(event_scores)) or np.any(event_scores < -1e-6):
            raise SchemaError("event scores must be finite non-negative values aligned with bridges")
    source_state = node_states[src]
    destination_state = node_states[dst]
    if (np.any(source_state < 0) or np.any(destination_state < 0)
            or np.any(source_state >= len(boards)) or np.any(destination_state >= len(boards))):
        raise SchemaError("graph node maps outside the puzzle sidecar state table")

    memo: dict[int, int | None] = {}
    press_counts = np.empty(len(src), dtype=np.int64)
    for index, (source, destination) in enumerate(zip(source_state, destination_state, strict=True)):
        result = _minimum_press_count(boards[source] ^ boards[destination], toggle, memo)
        press_counts[index] = -1 if result is None else result

    lower_bound = np.full(len(src), np.inf, dtype=np.float64)
    lower_bound[press_counts == 0] = 0.0
    positive = press_counts > 0
    lower_bound[positive] = (press_counts[positive] - 1) * min_button_interval + 1.0
    impossible = (press_counts < 0) | (distance < lower_bound)
    margin = distance - lower_bound
    groups = _np_group_masks(press_counts)
    count_total = len(src)
    distribution = {}
    distance_by_group = {}
    impossible_by_group = {}
    for label in _NP_GROUPS:
        mask = groups[label]
        count = int(np.count_nonzero(mask))
        distribution[label] = {
            "count": count,
            "fraction": float(count / count_total) if count_total else 0.0,
        }
        distance_by_group[label] = {
            "count": count,
            **(_quantiles(distance[mask]) or {"p10": None, "p50": None, "p90": None}),
        }
        impossible_count = int(np.count_nonzero(impossible & mask))
        impossible_by_group[label] = {
            "count": impossible_count,
            "fraction_within_group": (
                float(impossible_count / count) if count else None
            ),
            "fraction_of_all_bridges": (
                float(impossible_count / count_total) if count_total else 0.0
            ),
        }

    worst_order = sorted(
        np.flatnonzero(impossible).tolist(),
        key=lambda i: (float(margin[i]), int(src[i]), int(dst[i])),
    )[:20]
    worst = []
    for index in worst_order:
        state = int(source_state[index])
        p_count = int(press_counts[index])
        worst.append({
            "s": int(src[index]),
            "s_prime": int(dst[index]),
            "episode": int(episode[state]),
            "t": int(times[state]),
            "n_p": "infinity" if p_count < 0 else p_count,
            "d_theta": float(distance[index]),
            "L": "infinity" if p_count < 0 else float(lower_bound[index]),
            "d_theta_minus_L": "-infinity" if p_count < 0 else float(margin[index]),
        })

    impossible_count = int(np.count_nonzero(impossible))
    result = {
        "bridge_count": int(count_total),
        "n_p_distribution": distribution,
        "d_theta_by_n_p": distance_by_group,
        "impossible_bridge_count": impossible_count,
        "impossible_bridge_fraction": (
            float(impossible_count / count_total) if count_total else 0.0
        ),
        "impossible_by_n_p": impossible_by_group,
        "worst_20_impossible": worst,
    }
    if event_scores is not None:
        predicted_one_event = event_scores <= 1.5
        actual_at_most_one = (press_counts >= 0) & (press_counts <= 1)
        result["e_phi_by_n_p"] = {
            label: {
                "count": int(np.count_nonzero(mask)),
                **(_quantiles(event_scores[mask]) or {"p10": None, "p50": None, "p90": None}),
            }
            for label, mask in groups.items()
        }
        result["e_phi_le_1_5_vs_n_p_le_1"] = {
            "true_positive": int(np.count_nonzero(predicted_one_event & actual_at_most_one)),
            "false_positive": int(np.count_nonzero(predicted_one_event & ~actual_at_most_one)),
            "false_negative": int(np.count_nonzero(~predicted_one_event & actual_at_most_one)),
            "true_negative": int(np.count_nonzero(~predicted_one_event & ~actual_at_most_one)),
        }
    return result


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, nargs="*", default=[])
    parser.add_argument("--candidates-only", action="store_true",
                        help="audit a candidate cache without reading graph bridges")
    parser.add_argument("--candidate-cache", type=Path,
                        help="shared v0.8 window-5 pair cache for --candidates-only")
    parser.add_argument("--distance-checkpoint", type=Path,
                        help="state quasimetric checkpoint to audit on every candidate pair")
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="fixed d_R checkpoint used for the held-out P2 comparison")
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output JSON report")
    parser.add_argument("--object-event-checkpoint", type=Path,
                        help="optional v0.8.2 e_phi checkpoint for retained-edge comparison")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.candidates_only:
        if args.graph:
            raise SystemExit("--candidates-only cannot be combined with --graph")
        if args.candidate_cache is None:
            raise SystemExit("--candidates-only requires --candidate-cache")
        _candidate_only_mode(args)
    elif args.distance_checkpoint is not None or args.reference_checkpoint is not None:
        raise SystemExit("--distance-checkpoint and --reference-checkpoint require --candidates-only")
    elif not args.graph:
        raise SystemExit("provide --graph or use --candidates-only")
    trans, trans_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    sim_state.validate(trans.n_rows + int(np.count_nonzero(trans.terminal)))
    if sim_state.button_states is None:
        raise SchemaError("puzzle board audit requires sim_state.button_states")
    raw_boards = np.asarray(sim_state.button_states)
    if np.any((raw_boards != 0) & (raw_boards != 1)):
        raise SchemaError("puzzle button_states must be binary")
    boards = raw_boards.astype(np.uint8, copy=False)
    if boards.shape[1] not in _BOARD_LAYOUTS:
        raise SchemaError("board audit supports 3x3, 4x4, or 4x5 puzzle layouts")
    toggle = toggle_matrix(*_BOARD_LAYOUTS[boards.shape[1]])
    episode, times = state_episode_time(trans)
    intervals = _board_change_intervals(trans, boards)
    min_button_interval = float(np.quantile(intervals, 0.01))
    interval_report = {
        "sample_count": int(len(intervals)),
        "minimum_frames": float(np.min(intervals)),
        "p1_frames": min_button_interval,
        "p5_frames": float(np.quantile(intervals, 0.05)),
    }

    if args.candidates_only:
        mode = _candidate_only_mode(args)
        from experiments.v08_runtime import (
            load_quasimetric_checkpoint,
            make_embedding_cache_predictor,
        )

        if mode == "distance":
            result = _candidate_distance_audit(
                args, trans, trans_meta, sim_state, boards,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
            return 0

        checkpoint = args.object_event_checkpoint
        event_meta = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if (event_meta.get("model_kind") != "object-quasimetric-v0.8.2"
                or event_meta.get("object_cost") != "events"):
            raise ValueError("--object-event-checkpoint must name a v0.8.2 event-count model")
        if event_meta.get("object_label_version") != "contact-segment-position-q999-v2":
            raise ValueError("event checkpoint was not trained with the corrected contact labels")
        if event_meta.get("dataset_id") != trans_meta["dataset_id"] or int(event_meta.get("window", -1)) != 5:
            raise ValueError("event checkpoint dataset ID or window does not match the inputs")
        with np.load(args.candidate_cache, allow_pickle=False) as cache:
            required = {"pairs", "dataset_id", "window"}
            if not required.issubset(cache.files):
                raise ValueError("candidate cache is missing pairs, dataset_id, or window metadata")
            if str(cache["dataset_id"].item()) != trans_meta["dataset_id"]:
                raise ValueError("candidate cache dataset ID does not match the input data")
            if int(cache["window"].item()) != 5:
                raise ValueError("D2.5 requires the shared five-frame candidate cache")
            pairs = np.asarray(cache["pairs"], dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("candidate cache pairs must have shape (N,2)")
        if pairs.size and (np.any(pairs < 0) or np.any(pairs >= len(boards))):
            raise SchemaError("candidate pair references a state outside the sidecar table")
        if not len(pairs):
            raise ValueError("candidate cache has no candidate pairs")
        features = frame_object_features_from_checkpoint(
            trans, sim_state,
            continuous_mean=np.asarray(event_meta["continuous_mean"], dtype=np.float32),
            continuous_std=np.asarray(event_meta["continuous_std"], dtype=np.float32),
            window=5,
        )
        node_ids = np.unique(pairs)
        model, params, event_meta = load_quasimetric_checkpoint(
            checkpoint, input_dim=features.shape[1],
            expected_dataset_id=trans_meta["dataset_id"],
        )
        cache_path = args.out.parent / (
            f"{args.out.stem}-event-embeddings-{event_meta['params_sha256'][:16]}.npy"
        )
        predict, cache_report = make_embedding_cache_predictor(
            model, params, features[node_ids], node_ids,
            params_sha256=str(event_meta["params_sha256"]),
            dataset_id=trans_meta["dataset_id"], cache_path=cache_path,
        )
        event_scores = predict(pairs)
        press_counts = np.empty(len(pairs), dtype=np.int16)
        memo: dict[int, int | None] = {}
        for index, (source, destination) in enumerate(pairs):
            count = _minimum_press_count(
                boards[int(source)] ^ boards[int(destination)], toggle, memo
            )
            press_counts[index] = -1 if count is None else count
        result = {
            "diagnostic": "v0.8.2_d2_5_event_candidate_check",
            "dataset_id": trans_meta["dataset_id"],
            "candidate_cache": str(args.candidate_cache),
            "candidate_pair_sha256": _array_sha256(pairs),
            "event_checkpoint": str(checkpoint),
            "event_checkpoint_sha256": event_meta.get("params_sha256"),
            "event_embedding_cache": cache_report,
            "button_interval_frames": interval_report,
            **_candidate_event_report(press_counts, event_scores),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0

    loaded_graphs = []
    for graph_path in args.graph:
        graph = load_graph(graph_path, trans=trans)
        if graph.meta.get("bridge_judge") != "quasimetric":
            raise SchemaError(f"{graph_path} is not a quasimetric-judged puzzle graph")
        node_states = _node_state_indices(graph, trans)
        bridge_ids = np.flatnonzero(graph.edges.kind == 1)
        loaded_graphs.append((graph_path, graph, node_states, bridge_ids))

    event_predict = None
    event_meta = None
    event_cache_report = None
    if args.object_event_checkpoint is not None:
        from experiments.v08_runtime import (
            load_quasimetric_checkpoint,
            make_embedding_cache_predictor,
        )

        checkpoint = args.object_event_checkpoint
        event_meta = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if event_meta.get("model_kind") != "object-quasimetric-v0.8.2" or event_meta.get("object_cost") != "events":
            raise ValueError("--object-event-checkpoint must name a v0.8.2 event-count model")
        if event_meta.get("object_label_version") != "contact-segment-position-q999-v2":
            raise ValueError("event checkpoint was not trained with the corrected contact labels")
        if event_meta.get("dataset_id") is None:
            raise ValueError("event checkpoint lacks a dataset ID")
        if int(event_meta.get("window", -1)) != 5:
            raise ValueError("event checkpoint must use a five-frame window")
        expected_dataset_ids = {graph.meta.get("dataset_id") for _, graph, _, _ in loaded_graphs}
        if expected_dataset_ids != {event_meta["dataset_id"]} or event_meta["dataset_id"] != trans_meta["dataset_id"]:
            raise ValueError("event checkpoint dataset ID does not match the graph inputs")
        node_ids = np.unique(np.concatenate([
            np.concatenate((graph.edges.src[bridge_ids], graph.edges.dst[bridge_ids]))
            for _, graph, _, bridge_ids in loaded_graphs if len(bridge_ids)
        ])) if any(len(item[3]) for item in loaded_graphs) else np.empty(0, dtype=np.int64)
        if len(node_ids):
            features = frame_object_features_from_checkpoint(
                trans, sim_state,
                continuous_mean=np.asarray(event_meta["continuous_mean"], dtype=np.float32),
                continuous_std=np.asarray(event_meta["continuous_std"], dtype=np.float32),
                window=5,
            )
            if any(len(node_states) != len(features) for _, _, node_states, _ in loaded_graphs):
                raise SchemaError("event model node table does not align with the puzzle state table")
            reference_node_states = loaded_graphs[0][2]
            if any(not np.array_equal(node_states, reference_node_states)
                   for _, _, node_states, _ in loaded_graphs[1:]):
                raise SchemaError("graphs have different node-to-state mappings")
            model, params, event_meta = load_quasimetric_checkpoint(
                checkpoint, input_dim=features.shape[1],
                expected_dataset_id=event_meta["dataset_id"],
            )
            cache_path = args.out.parent / (
                f"{args.out.stem}-event-embeddings-{event_meta['params_sha256'][:16]}.npy"
            )
            event_predict, event_cache_report = make_embedding_cache_predictor(
                model, params, features[reference_node_states[node_ids]], node_ids,
                params_sha256=str(event_meta["params_sha256"]),
                dataset_id=str(event_meta["dataset_id"]), cache_path=cache_path,
            )

    graph_reports = []
    for graph_path, graph, node_states, bridge_ids in loaded_graphs:
        bridge_src = graph.edges.src[bridge_ids]
        bridge_dst = graph.edges.dst[bridge_ids]
        event_scores = (
            event_predict(np.column_stack((bridge_src, bridge_dst)))
            if event_predict is not None and len(bridge_ids) else
            (np.empty(0, dtype=np.float32) if event_predict is not None else None)
        )
        report = analyze_bridge_arrays(
            bridge_src, bridge_dst,
            graph.edges.dist[bridge_ids], node_state_indices=node_states,
            button_states=boards, episode=episode, times=times,
            min_button_interval=min_button_interval, toggle=toggle,
            event_scores=event_scores,
        )
        graph_reports.append({
            "graph": str(graph_path),
            "graph_hash": graph.content_hash,
            "bridge_lambda_steps": graph.meta.get("bridge_lambda_steps", 0.0),
            "object_event_checkpoint_sha256": event_meta.get("params_sha256") if event_meta else None,
            "button_interval_frames": interval_report,
            **report,
        })

    result = {
        "diagnostic": "temporary_puzzle_board_lower_bound_v0.8.2",
        "transitions": str(args.transitions),
        "sim_state": str(args.sim_state),
        "object_event_checkpoint": str(args.object_event_checkpoint) if args.object_event_checkpoint else None,
        "object_event_embedding_cache": event_cache_report,
        "limitations": [
            "This is an offline lower-bound check, not a simulator reachability test.",
            "n_p <= 1 bridges generally cannot be ruled out by this timing bound.",
            "A bridge that passes the check is not thereby certified executable.",
        ],
        "graphs": graph_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
