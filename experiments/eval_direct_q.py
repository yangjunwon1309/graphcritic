"""Evaluate v0.9 direct Q against held-out five-step graph costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import spearmanr

from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.direct_q import load_direct_q_scorer
from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.state_graph import state_episode_time
from graphcritic.weights import step_cost


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be positive")
    started = time.perf_counter()
    trans, trans_meta = load_transitions(args.transitions)
    sim = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim, seed=0)
    frame = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    windows = puzzle_state_windows(frame, trans, data.terminal_state_index, window=5)
    graph = load_graph(json.loads(args.config.read_text())["graph"], trans=trans)
    costs = np.asarray(graph.cost / step_cost(graph.config.gamma), dtype=np.float32)
    episode, times = state_episode_time(trans)
    candidates = np.asarray(data.heldout_rows, dtype=np.int64)
    endpoints = candidates + 5
    valid = (times[candidates] % 5 == 4) & (endpoints <= data.episode_end_rows[candidates])
    candidates = candidates[valid]
    endpoints = endpoints[valid]
    finite_y = np.isfinite(costs[endpoints])
    candidates, endpoints = candidates[finite_y], endpoints[finite_y]
    rng = np.random.default_rng(args.seed)
    if len(candidates) > args.n:
        selected = rng.choice(len(candidates), args.n, replace=False)
        candidates, endpoints = candidates[selected], endpoints[selected]
    if not len(candidates):
        raise ValueError("no held-out boundary rows with finite five-step endpoint graph costs")

    scorer = load_direct_q_scorer(args.config)
    config = json.loads(args.config.read_text())
    action_parameterization = str(config.get("action_parameterization", "free"))
    if action_parameterization not in {"free", "residual"}:
        raise ValueError("direct-Q config has an unsupported action parameterization")
    q_values = np.full(len(candidates), np.inf, dtype=np.float64)
    y_values = 5.0 + costs[endpoints].astype(np.float64)
    local_distance = np.full(len(candidates), np.nan, dtype=np.float64)
    alternative_distance = np.full(len(candidates), np.nan, dtype=np.float64)
    fallback_reasons: dict[str, int] = {}
    n_sizes, decision_seconds = [], []
    fallback_count = 0
    recorded_chunks = np.stack([trans.action[row : row + 5] for row in candidates])
    boards = data.state_bits[candidates].astype(np.uint8)
    excluded = trans.episode_id[candidates]
    batch_size = 32
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        q, diagnostics, d5 = scorer.score_batch(
            windows[candidates[start:end]], recorded_chunks[start:end], boards[start:end],
            exclude_episodes=excluded[start:end], target_node_ids=endpoints[start:end],
        )
        q_values[start:end] = q
        if d5 is not None:
            local_distance[start:end] = d5
        for diag in diagnostics:
            n_sizes.append(int(diag["neighborhood_size"]))
            decision_seconds.append(float(diag["decision_seconds"]))
            if diag["fallback"]:
                fallback_count += 1
                reason = str(diag["fallback_reason"] or "unknown")
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    if len(candidates) > 1:
        alternative_chunks = np.roll(recorded_chunks, 1, axis=0)
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            _, _, alt_d = scorer.score_batch(
                windows[candidates[start:end]], alternative_chunks[start:end],
                boards[start:end], exclude_episodes=excluded[start:end],
                target_node_ids=endpoints[start:end],
            )
            if alt_d is not None:
                alternative_distance[start:end] = alt_d

    finite = np.isfinite(q_values)
    abs_error = np.abs(q_values - y_values)
    median_value = float(np.median(abs_error))
    median_abs_error = median_value if np.isfinite(median_value) else None
    rho = spearmanr(q_values[finite], y_values[finite]).statistic if np.count_nonzero(finite) >= 2 else np.nan
    g1_values = local_distance[np.isfinite(local_distance)]
    sensitivity_mask = np.isfinite(local_distance) & np.isfinite(alternative_distance)
    sensitivity_delta = alternative_distance[sensitivity_mask] - local_distance[sensitivity_mask]
    result = {
        "spec_version": config.get("spec_version", "v0.9"),
        "action_parameterization": action_parameterization,
        "dataset_id": trans_meta["dataset_id"],
        "graph_hash": graph.content_hash,
        "n_requested": int(args.n),
        "n_sampled": int(len(candidates)),
        "n_finite_q": int(np.count_nonzero(finite)),
        "G1": {
            "n": int(len(g1_values)),
            "ensemble_distance_median_to_x_t_plus_5": float(np.median(g1_values)) if len(g1_values) else None,
            "fraction_above_h_plus_epsilon_6_25": float(np.mean(g1_values > 6.25)) if len(g1_values) else None,
            "gate_passed": bool(len(g1_values) and 4.0 <= np.median(g1_values) <= 6.25 and np.mean(g1_values > 6.25) <= 0.05),
        },
        "G2": {
            "median_absolute_error_steps": median_abs_error,
            "finite_error_count": int(np.count_nonzero(np.isfinite(abs_error))),
            "gate_passed": bool(median_abs_error is not None and median_abs_error <= 5.0),
        },
        "G3": {
            "n": int(np.count_nonzero(sensitivity_mask)),
            "alternative_action_distance_greater_fraction": (
                float(np.mean(alternative_distance[sensitivity_mask] > local_distance[sensitivity_mask]))
                if np.any(sensitivity_mask) else None
            ),
            "alternative_minus_recorded_distance_median": (
                float(np.median(sensitivity_delta)) if len(sensitivity_delta) else None
            ),
            "alternative_action": "recorded_chunk_from_previous_heldout_row",
        },
        "q_over_y_plus_5_fraction": float(np.mean(q_values > y_values + 5.0)),
        "q_under_y_minus_5_fraction": float(np.mean(q_values < y_values - 5.0)),
        "spearman_q_y": float(rho) if np.isfinite(rho) else None,
        "fallback_count": int(fallback_count),
        "fallback_fraction": float(fallback_count / len(candidates)),
        "fallback_reasons": fallback_reasons,
        "mean_neighborhood_size": float(np.mean(n_sizes)) if n_sizes else None,
        "mean_decision_seconds": float(np.mean(decision_seconds)) if decision_seconds else None,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

