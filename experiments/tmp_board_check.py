"""Offline lower-bound audit of retained v0.8 puzzle bridges.

This diagnostic checks whether stored quasimetric distances are shorter than
the minimum button-board transition time supported by the offline data. It
does not simulate or execute bridges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.modes import state_episode_time
from graphcritic.puzzle import minimum_press_solution, toggle_matrix
from graphcritic.types import GraphArtifact, SchemaError, TransitionSet


_BOARD_LAYOUTS = {9: (3, 3), 16: (4, 4), 20: (4, 5)}
_NP_GROUPS = ("0", "1", "2", "3", "≥4", "∞")


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
    return {
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


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, nargs="+", required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    trans, _ = load_transitions(args.transitions)
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

    graph_reports = []
    for graph_path in args.graph:
        graph = load_graph(graph_path, trans=trans)
        if graph.meta.get("bridge_judge") != "quasimetric":
            raise SchemaError(f"{graph_path} is not a quasimetric-judged puzzle graph")
        node_states = _node_state_indices(graph, trans)
        bridge_ids = np.flatnonzero(graph.edges.kind == 1)
        report = analyze_bridge_arrays(
            graph.edges.src[bridge_ids], graph.edges.dst[bridge_ids],
            graph.edges.dist[bridge_ids], node_state_indices=node_states,
            button_states=boards, episode=episode, times=times,
            min_button_interval=min_button_interval, toggle=toggle,
        )
        graph_reports.append({
            "graph": str(graph_path),
            "graph_hash": graph.content_hash,
            "bridge_lambda_steps": graph.meta.get("bridge_lambda_steps", 0.0),
            "button_interval_frames": interval_report,
            **report,
        })

    result = {
        "diagnostic": "temporary_puzzle_board_lower_bound_v0.8.1",
        "transitions": str(args.transitions),
        "sim_state": str(args.sim_state),
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
