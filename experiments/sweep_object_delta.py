#!/usr/bin/env python3
"""Run the v1.8 cube-double δp sweep, G2 audit, and task graph/target builds."""

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

from graphcritic.bridges.cube_graph import (
    DELTA_POS_CANDIDATES_M,
    build_cube_candidate_cache,
    build_cube_judged_graph,
    compute_grasp_mismatch,
    load_cube_context,
    score_cube_candidate_cache,
    summarize_delta_sweep,
    task_budget_sweep,
)
from graphcritic.bridges.object_state import object_distance


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-double-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-budget", type=int, default=32)
    parser.add_argument("--bridge-filter", choices=("c0", "none"), default="none")
    parser.add_argument("--embedding-pair-batch-size", type=int, default=4096)
    parser.add_argument("--no-build-graphs", action="store_true")
    args = parser.parse_args(argv)
    if args.max_budget < 1 or args.embedding_pair_batch_size < 1:
        parser.error("max-budget and embedding-pair-batch-size must be positive")

    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_paths: dict[float, Path] = {}
    score_paths: dict[float, Path] = {}
    stages: dict[str, float] = {}
    for delta in DELTA_POS_CANDIDATES_M:
        key = f"{delta * 100:.1f}cm"
        candidate = args.out / "candidates" / f"delta_{delta:.3f}.npz"
        score = args.out / "scores" / f"delta_{delta:.3f}.npz"
        start = time.perf_counter()
        build_report = build_cube_candidate_cache(
            args.cube_double_dataset, args.checkpoint, candidate,
            delta_pos=delta, delta_rot_deg=15.0,
        )
        stages[f"candidate_{key}_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        score_report = score_cube_candidate_cache(
            args.cube_double_dataset, args.checkpoint, candidate, score,
            pair_batch_size=args.embedding_pair_batch_size,
        )
        stages[f"score_{key}_seconds"] = time.perf_counter() - start
        candidate_paths[delta] = candidate
        score_paths[delta] = score
        print(json.dumps({"delta": delta, "candidate": build_report, "score": score_report}, sort_keys=True), flush=True)

    sweep_started = time.perf_counter()
    ctx = load_cube_context(args.cube_double_dataset)
    sweep, selected = summarize_delta_sweep(
        ctx, candidate_paths, score_paths, max_budget=args.max_budget,
        bridge_filter=args.bridge_filter,
    )
    stages["delta_cost_sweep_seconds"] = time.perf_counter() - sweep_started
    sweep["elapsed_seconds"] = dict(stages)
    sweep["total_elapsed_seconds"] = time.perf_counter() - started
    _json_write(args.out / "delta_sweep.json", sweep)
    print(json.dumps({"selected_delta_pos_m": selected, "sweep_file": str(args.out / 'delta_sweep.json')}, sort_keys=True), flush=True)
    if args.no_build_graphs:
        return 0

    # Audit the complete accepted set for every task before any graph/target is
    # emitted. A failed G2 gate therefore stops before S5.
    candidate_path = candidate_paths[selected]
    score_path = score_paths[selected]
    with np.load(candidate_path, allow_pickle=False) as archive:
        pairs = np.asarray(archive["pairs"], dtype=np.int64)
        cached_pos = np.asarray(archive["object_distance_pos_m"], dtype=np.float64)
        cached_rot = np.asarray(archive["object_distance_rot_rad"], dtype=np.float64)
    with np.load(score_path, allow_pickle=False) as archive:
        distances = np.asarray(archive["distance_steps"], dtype=np.float32)
    g2_task: dict[str, dict] = {}
    mismatch_weighted = 0
    accepted_total = 0
    dpos_violations = 0
    drot_violations = 0
    accepted_pos: list[np.ndarray] = []
    accepted_rot: list[np.ndarray] = []
    for task in range(1, 6):
        result = task_budget_sweep(
            ctx, task, pairs, distances, max_budget=args.max_budget,
            bridge_filter=args.bridge_filter,
        )
        accepted_ids = result["accepted_candidate_indices"]
        accepted_pairs = pairs[accepted_ids]
        mismatch = compute_grasp_mismatch(ctx, accepted_pairs)
        mismatch_weighted += int(round(mismatch * len(accepted_ids)))
        accepted_total += len(accepted_ids)
        dpos_violations += int(np.count_nonzero(cached_pos[accepted_ids] > selected + 1e-9))
        drot_violations += int(np.count_nonzero(cached_rot[accepted_ids] > np.deg2rad(15.0) + 1e-9))
        accepted_pos.append(cached_pos[accepted_ids])
        accepted_rot.append(cached_rot[accepted_ids])
        g2_task[str(task)] = {
            "accepted_bridges": int(len(accepted_ids)),
            "grasp_mismatch_fraction": mismatch,
            "dpos_violation_count": int(np.count_nonzero(cached_pos[accepted_ids] > selected + 1e-9)),
            "drot_violation_count": int(np.count_nonzero(cached_rot[accepted_ids] > np.deg2rad(15.0) + 1e-9)),
            "dpos_p50_cm": float(np.median(cached_pos[accepted_ids]) * 100) if len(accepted_ids) else None,
            "dpos_p90_cm": float(np.quantile(cached_pos[accepted_ids], 0.9) * 100) if len(accepted_ids) else None,
            "drot_p50_deg": float(np.median(cached_rot[accepted_ids]) * 180 / np.pi) if len(accepted_ids) else None,
            "drot_p90_deg": float(np.quantile(cached_rot[accepted_ids], 0.9) * 180 / np.pi) if len(accepted_ids) else None,
        }
        del result
    aggregate_mismatch = mismatch_weighted / accepted_total if accepted_total else 0.0
    g2 = {
        "accepted_bridge_task_occurrences": int(accepted_total),
        "grasp_mismatch_count": int(mismatch_weighted),
        "grasp_mismatch_fraction": float(aggregate_mismatch),
        "dpos_violation_count": int(dpos_violations),
        "drot_violation_count": int(drot_violations),
        "pass": bool(aggregate_mismatch <= 0.05 and dpos_violations == 0 and drot_violations == 0),
        "per_task": g2_task,
        "definition": "grasp mismatch = contact OR elevated cube state differs; elevated means cube center z > 0.04 m",
    }
    sweep["g2"] = g2
    _json_write(args.out / "delta_sweep.json", sweep)
    _json_write(args.out / "g2.json", g2)
    if not g2["pass"]:
        raise SystemExit(f"G2 failed; stopped before task graph builds: {g2}")

    graphs_started = time.perf_counter()
    selected_row = sweep["rows"][f"{selected:.3f}"]
    graph_results = {}
    for task in range(1, 6):
        b_star = int(selected_row["per_task"][str(task)]["b_star"])
        result = build_cube_judged_graph(
            args.cube_double_dataset, candidate_path, score_path,
            args.out / "graphs" / f"task{task}", task=task, b_max=b_star,
            target_kind="discounted", gamma=0.99, delta_pos=selected,
            delta_rot_deg=15.0, max_budget_for_sweep=args.max_budget,
            bridge_filter=args.bridge_filter,
        )
        graph_results[str(task)] = result
        print(json.dumps({"task": task, "graph_hash": result["graph_hash"], "b_star": b_star,
                          "target_mean": result["target_mean"]}, sort_keys=True), flush=True)
    stages["task_graph_and_target_builds_seconds"] = time.perf_counter() - graphs_started
    sweep["graph_build_seconds"] = stages["task_graph_and_target_builds_seconds"]
    sweep["task_graphs"] = {
        task: {
            key: graph_results[task][key] for key in (
                "graph_hash", "n_candidate_pairs", "n_optimistic_candidates",
                "n_accepted_bridges", "n_retained_bridges", "b_star",
                "graph_reachable_fraction", "target_valid_fraction",
                "target_loss_mask_fraction", "target_mean", "target_p50",
                "target_p90", "target_capped_loss_mask_fraction",
                "c_reduced_node_count", "c_reduced_node_fraction_of_all",
                "c_reduction_p50_steps", "c_reduction_p90_steps",
                "newly_reachable_node_count", "graph_bridge_count_distribution",
                "accepted_grasp_mismatch_fraction",
                "retained_object_distance_pos_violation_count",
            )
        }
        for task in graph_results
    }
    sweep["elapsed_seconds"] = dict(stages)
    sweep["total_elapsed_seconds"] = time.perf_counter() - started
    _json_write(args.out / "delta_sweep.json", sweep)
    _json_write(args.out / "task_graph_summary.json", sweep["task_graphs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
