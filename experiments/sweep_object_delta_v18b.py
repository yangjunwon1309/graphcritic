#!/usr/bin/env python3
"""Run the v1.8b cube-double δxy × Kfree sweep and gated graph build."""

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
    BRIDGE_LAMBDA_STEPS,
    DELTA_ROT_DEG,
    DELTA_XY_V18B_CANDIDATES_M,
    DELTA_Z_V18B_M,
    K_FREE_V18B_CANDIDATES,
    build_cube_candidate_cache,
    build_cube_judged_graph,
    cube_pair_structure,
    load_cube_context,
    score_cube_candidate_cache,
    task_budget_sweep,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _load_pairs_scores(candidate: Path, score: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(candidate, allow_pickle=False) as archive:
        pairs = np.asarray(archive["pairs"], dtype=np.int64)
        pair_hash = str(archive["dataset_hash"].item())
    with np.load(score, allow_pickle=False) as archive:
        score_pairs = np.asarray(archive["pairs"], dtype=np.int64)
        scores = np.asarray(archive["distance_steps"], dtype=np.float32)
        score_hash = str(archive["dataset_hash"].item())
    if pair_hash != score_hash or not np.array_equal(pairs, score_pairs):
        raise ValueError(f"candidate/score cache mismatch: {candidate}")
    return pairs, scores


def _row_for(
    ctx: dict,
    pairs: np.ndarray,
    scores: np.ndarray,
    *,
    bridge_filter: str,
    k_free: int,
    contact_sensitive_cap: bool = True,
) -> tuple[dict, dict[int, dict]]:
    tasks: dict[int, dict] = {}
    for task in range(1, 6):
        result = task_budget_sweep(
            ctx, task, pairs, scores, max_budget=32,
            bridge_filter=bridge_filter, k_free=k_free,
            contact_sensitive_cap=contact_sensitive_cap,
        )
        b32 = result["budget_sweep"][-1]
        selected_cost = result["budget_sweep"][result["b_star"]]
        tasks[task] = {
            "n_optimistic_candidates": result["n_optimistic_candidates"],
            "n_accepted_bridges": result["n_accepted_bridges"],
            "b_star": result["b_star"],
            "r_b32": b32["reachable_fraction"],
            "r_sat": result["r_sat"],
            "c_p50_steps": selected_cost["p50"],
            "c_p90_steps": selected_cost["p90"],
            "budget_sweep": result["budget_sweep"],
            "accepted_candidate_indices": result["accepted_candidate_indices"],
            "accepted_free_extended_distance_steps": result["accepted_free_extended_distance_steps"],
        }
    mean_r = float(np.mean([tasks[t]["r_b32"] for t in range(1, 6)]))
    first = tasks[1]
    free_extended = np.asarray(first["accepted_free_extended_distance_steps"], dtype=np.float64)
    row = {
        "candidate_pairs": int(len(pairs)),
        "accepted_pairs": int(first["n_accepted_bridges"]),
        "mean_reachability_b32": mean_r,
        "bridge_filter": bridge_filter,
        "k_free": int(k_free),
        "contact_sensitive_cap": bool(contact_sensitive_cap),
        "accepted_free_extended_dtheta_p50": float(np.median(free_extended)) if len(free_extended) else None,
        "accepted_free_extended_dtheta_p90": float(np.quantile(free_extended, .9)) if len(free_extended) else None,
        "accepted_free_dtheta_p50": None,
        "accepted_free_dtheta_p90": None,
        "per_task": {},
    }
    for task, value in tasks.items():
        row["per_task"][str(task)] = {key: val for key, val in value.items() if key not in {
            "accepted_candidate_indices", "accepted_free_extended_distance_steps"
        }}
    # Recompute all accepted, free endpoint distances for reporting, including
    # those below the original 20-step bound.
    accepted = np.asarray(first["accepted_candidate_indices"], dtype=np.int64)
    if len(accepted):
        contact = np.asarray(ctx["states"][:, 18] > .5, dtype=bool)
        both_free = ~contact[pairs[accepted, 0]] & ~contact[pairs[accepted, 1]]
        free_dist = np.asarray(scores[accepted][both_free], dtype=np.float64)
        if len(free_dist):
            row["accepted_free_dtheta_p50"] = float(np.median(free_dist))
            row["accepted_free_dtheta_p90"] = float(np.quantile(free_dist, .9))
    return row, tasks


def _g2_report(
    ctx: dict, pairs: np.ndarray, distances: np.ndarray, *, delta_xy: float,
    k_free: int, bridge_filter: str = "none",
) -> dict:
    per_task: dict[str, dict] = {}
    totals = {name: 0 for name in (
        "accepted_occurrences", "dxy_violation_count", "dz_violation_count",
        "drot_violation_count", "contact_mismatch_count", "support_mismatch_count",
        "legacy_grasp_mismatch_count",
    )}
    for task in range(1, 6):
        result = task_budget_sweep(
            ctx, task, pairs, distances, max_budget=32,
            bridge_filter=bridge_filter, k_free=k_free, contact_sensitive_cap=True,
        )
        accepted_indices = result["accepted_candidate_indices"]
        accepted_pairs = pairs[accepted_indices]
        structure = cube_pair_structure(ctx, accepted_pairs)
        counts = {
            "accepted_occurrences": int(len(accepted_pairs)),
            "dxy_violation_count": int(np.count_nonzero(structure["dxy_m"] > delta_xy + 1e-9)),
            "dz_violation_count": int(np.count_nonzero(structure["dz_m"] > DELTA_Z_V18B_M + 1e-9)),
            "drot_violation_count": int(np.count_nonzero(structure["drot_rad"] > np.deg2rad(DELTA_ROT_DEG) + 1e-9)),
            "contact_mismatch_count": int(np.count_nonzero(structure["contact_mismatch"])),
            "support_mismatch_count": int(np.count_nonzero(structure["support_mismatch"])),
            "legacy_grasp_mismatch_count": int(np.count_nonzero(structure["legacy_grasp_mismatch"])),
        }
        counts["support_mismatch_fraction"] = (
            counts["support_mismatch_count"] / counts["accepted_occurrences"]
            if counts["accepted_occurrences"] else 0.0
        )
        counts["legacy_grasp_mismatch_fraction"] = (
            counts["legacy_grasp_mismatch_count"] / counts["accepted_occurrences"]
            if counts["accepted_occurrences"] else 0.0
        )
        per_task[str(task)] = counts
        for key in totals:
            totals[key] += counts[key]
    total_support_rate = totals["support_mismatch_count"] / max(1, totals["accepted_occurrences"])
    total_legacy_rate = totals["legacy_grasp_mismatch_count"] / max(1, totals["accepted_occurrences"])
    structural_ok = all(totals[key] == 0 for key in (
        "dxy_violation_count", "dz_violation_count", "drot_violation_count", "contact_mismatch_count"
    ))
    return {
        "delta_xy_m": delta_xy,
        "delta_z_m": DELTA_Z_V18B_M,
        "delta_rot_deg": DELTA_ROT_DEG,
        "k_free": k_free,
        "totals": totals,
        "support_mismatch_fraction": total_support_rate,
        "legacy_grasp_mismatch_fraction": total_legacy_rate,
        "per_task": per_task,
        "structural_checks_pass": bool(structural_ok),
        "support_gate_pass": bool(total_support_rate <= .05),
        "pass": bool(structural_ok and total_support_rate <= .05),
        "support_definition": "table: z<0.03m; stacked: xy<0.02m and upper-center dz in [0.03,0.05]m; lifted: otherwise",
        "legacy_g2_definition": "contact OR any cube center z>0.04m differs between endpoints",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-double-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True,
                        help="read-only v1.8 bridge_sweep directory used for diagnostic rows")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embedding-pair-batch-size", type=int, default=4096)
    parser.add_argument("--bridge-filter", choices=("c0", "none"), default="none")
    args = parser.parse_args(argv)
    if args.embedding_pair_batch_size < 1:
        parser.error("embedding-pair-batch-size must be positive")
    start_all = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = load_cube_context(args.cube_double_dataset)
    row_table: dict[str, dict] = {}
    sweep_timings: dict[str, float] = {}
    cache_paths: dict[float, tuple[Path, Path]] = {}
    stage_timings: dict[str, float] = {}

    # Two fixed diagnostic rows use the already-recorded legacy candidate and
    # score caches. They are not regenerated or modified.
    legacy_candidate = args.legacy_dir / "candidates/delta_0.040.npz"
    legacy_score = args.legacy_dir / "scores/delta_0.040.npz"
    old_pairs, old_scores = _load_pairs_scores(legacy_candidate, legacy_score)
    for label, filt in (("v18_legacy_c0", "c0"), ("v18_legacy_none", "none")):
        tick = time.perf_counter()
        row, _ = _row_for(
            ctx, old_pairs, old_scores, bridge_filter=filt, k_free=20,
            contact_sensitive_cap=False,
        )
        row.update({"diagnostic": True, "match": "v1.8 isotropic 4cm; no contact; no z split"})
        row_table[label] = row
        sweep_timings[f"{label}_seconds"] = time.perf_counter() - tick

    for delta_xy in DELTA_XY_V18B_CANDIDATES_M:
        key = f"xy_{int(round(delta_xy * 100))}cm"
        candidate = args.out / "candidates" / f"{key}.npz"
        score = args.out / "scores" / f"{key}.npz"
        tick = time.perf_counter()
        candidate_report = build_cube_candidate_cache(
            args.cube_double_dataset, args.checkpoint, candidate,
            delta_xy=delta_xy, delta_z=DELTA_Z_V18B_M,
            require_contact_match=True, delta_rot_deg=DELTA_ROT_DEG,
        )
        stage_timings[f"candidate_{key}_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        score_report = score_cube_candidate_cache(
            args.cube_double_dataset, args.checkpoint, candidate, score,
            pair_batch_size=args.embedding_pair_batch_size,
        )
        stage_timings[f"score_{key}_seconds"] = time.perf_counter() - tick
        cache_paths[delta_xy] = (candidate, score)
        pairs, distances = _load_pairs_scores(candidate, score)
        for k_free in K_FREE_V18B_CANDIDATES:
            label = f"xy_{int(round(delta_xy * 100))}cm_kfree_{k_free}"
            tick = time.perf_counter()
            row, _ = _row_for(
                ctx, pairs, distances, bridge_filter=args.bridge_filter, k_free=k_free,
                contact_sensitive_cap=True,
            )
            row.update({
                "diagnostic": False,
                "delta_xy_m": delta_xy,
                "delta_z_m": DELTA_Z_V18B_M,
                "delta_rot_deg": DELTA_ROT_DEG,
                "match": "Dxy/Dz/Drot/contact",
                "candidate_cache": str(candidate),
                "score_cache": str(score),
                "candidate_cache_report": candidate_report,
                "score_cache_report": score_report,
            })
            row_table[label] = row
            sweep_timings[f"{label}_seconds"] = time.perf_counter() - tick
            print(json.dumps({"combo": label, "candidate_pairs": row["candidate_pairs"],
                              "accepted": row["accepted_pairs"],
                              "mean_r_b32": row["mean_reachability_b32"],
                              "seconds": sweep_timings[f"{label}_seconds"]}, sort_keys=True), flush=True)

    combo_labels = [f"xy_{int(round(d * 100))}cm_kfree_{k}"
                    for d in DELTA_XY_V18B_CANDIDATES_M for k in K_FREE_V18B_CANDIDATES]
    max_mean = max(row_table[name]["mean_reachability_b32"] for name in combo_labels)
    eligible = [name for name in combo_labels
                if row_table[name]["mean_reachability_b32"] >= max_mean - 0.005]
    selected_label = min(eligible, key=lambda name: (
        row_table[name]["k_free"], row_table[name]["delta_xy_m"]
    ))
    selected_row = row_table[selected_label]
    selected_mean = float(selected_row["mean_reachability_b32"])
    summary = {
        "spec_version": "v1.8b",
        "dataset_id_base": ctx["dataset_id_base"],
        "dataset_hash": ctx["dataset_hash"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_params_sha256": json.loads((args.checkpoint / "metadata.json").read_text())["params_sha256"],
        "candidate_knn": 32,
        "bridge_cost": "dtheta+10.5",
        "bridge_filter": args.bridge_filter,
        "rows": row_table,
        "selection": {
            "eligible_within_0_5pp_of_max": eligible,
            "maximum_mean_reachability_b32": max_mean,
            "selected": selected_label,
            "delta_xy_m": selected_row["delta_xy_m"],
            "k_free": selected_row["k_free"],
            "mean_reachability_b32": selected_mean,
            "per_task_b_star": {task: selected_row["per_task"][task]["b_star"] for task in map(str, range(1, 6))},
            "gate_passed": bool(selected_mean >= 0.5),
        },
        "stage_seconds": stage_timings,
        "combo_sweep_seconds": sweep_timings,
        "g2": None,
        "task_graphs": None,
    }
    _write(args.out / "sweep_v18b.json", summary)
    if selected_mean < 0.5:
        summary["stopped_at"] = "S3_prime_reachability_gate"
        _write(args.out / "sweep_v18b.json", summary)
        print(json.dumps({"stopped_at": summary["stopped_at"], "selection": summary["selection"]}, sort_keys=True))
        return 0

    selected_delta = float(selected_row["delta_xy_m"])
    selected_k = int(selected_row["k_free"])
    candidate, score = cache_paths[selected_delta]
    pairs, distances = _load_pairs_scores(candidate, score)
    g2_started = time.perf_counter()
    g2 = _g2_report(ctx, pairs, distances, delta_xy=selected_delta, k_free=selected_k,
                    bridge_filter=args.bridge_filter)
    g2["elapsed_seconds"] = time.perf_counter() - g2_started
    summary["g2"] = g2
    _write(args.out / "g2_v18b.json", g2)
    if not g2["pass"]:
        summary["stopped_at"] = "S4_prime_G2_gate"
        summary["stage_seconds"]["g2_seconds"] = g2["elapsed_seconds"]
        summary["total_elapsed_seconds"] = time.perf_counter() - start_all
        _write(args.out / "sweep_v18b.json", summary)
        print(json.dumps({"stopped_at": summary["stopped_at"], "g2": g2}, sort_keys=True))
        return 0

    graph_started = time.perf_counter()
    graph_results = {}
    for task in range(1, 6):
        b_star = int(selected_row["per_task"][str(task)]["b_star"])
        result = build_cube_judged_graph(
            args.cube_double_dataset, candidate, score,
            args.out / "graphs" / f"task{task}", task=task, b_max=b_star,
            target_kind="discounted", gamma=0.99, delta_xy=selected_delta,
            delta_z=DELTA_Z_V18B_M, require_contact_match=True,
            delta_rot_deg=DELTA_ROT_DEG, bridge_lambda_steps=BRIDGE_LAMBDA_STEPS,
            max_budget_for_sweep=32, bridge_filter=args.bridge_filter, k_free=selected_k,
        )
        graph_results[str(task)] = result
        print(json.dumps({"task": task, "graph_hash": result["graph_hash"],
                          "b_star": b_star, "reachable": result["graph_reachable_fraction"]}, sort_keys=True), flush=True)
    summary["stage_seconds"]["task_graph_target_builds_seconds"] = time.perf_counter() - graph_started
    summary["task_graphs"] = {
        task: {key: result[key] for key in (
            "graph_hash", "n_candidate_pairs", "n_optimistic_candidates", "n_accepted_bridges",
            "n_retained_bridges", "b_max", "r_sat", "graph_reachable_fraction",
            "target_valid_fraction", "target_loss_mask_fraction", "target_mean", "target_p50",
            "target_p90", "target_capped_loss_mask_fraction", "graph_bridge_count_distribution",
            "retained_object_distance_xy_violation_count", "retained_object_distance_z_violation_count",
            "retained_contact_mismatch_count",
        )} for task, result in graph_results.items()
    }
    summary["total_elapsed_seconds"] = time.perf_counter() - start_all
    _write(args.out / "sweep_v18b.json", summary)
    _write(args.out / "task_graph_summary.json", summary["task_graphs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
