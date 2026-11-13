#!/usr/bin/env python3
"""Check v1.8 cube data row alignment and threshold-bridge gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.bridges.cube_graph import (
    _graph_hash,
    compute_grasp_mismatch,
    load_cube_context,
)
from graphcritic.bridges.object_state import object_distance
from graphcritic.envs.ogbench_cube import OBS_DIM


def _qc_valid(terminals: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(terminals, dtype=bool)
    result = np.zeros(len(values), dtype=bool)
    n_starts = max(0, len(values) - horizon + 1)
    if n_starts:
        prefix = np.r_[0, np.cumsum(values, dtype=np.int64)]
        starts = np.arange(n_starts)
        result[:n_starts] = prefix[starts + horizon - 1] - prefix[starts] == 0
    return result


def _alignment(ctx: dict) -> dict:
    with np.load(ctx["path"], allow_pickle=False) as archive:
        obs = np.asarray(archive["observations"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)
        raw_terminal = np.asarray(archive["terminals"], dtype=bool)
    trans = ctx["transitions"]
    # Import the actual repository loaders by file path so the root checker
    # does not depend on either fork's top-level module names.
    qc_loader_path = ROOT / "third_party/qc/envs/ogbench_utils.py"
    dqc_loader_path = ROOT / "third_party/dqc/envs/env_utils.py"

    def _load_function(module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load OGBench row adapter {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_dataset

    qc_load_dataset = _load_function("v18_qc_ogbench_loader", qc_loader_path)
    dqc_load_dataset = _load_function("v18_dqc_ogbench_loader", dqc_loader_path)
    qc_data = qc_load_dataset(ctx["path"], compact_dataset=False, add_info=False)
    dqc_data = dqc_load_dataset(ctx["path"], compact_dataset=True, add_info=False)
    dqc_rows = np.flatnonzero(np.asarray(dqc_data["valids"]) > 0)
    qc_obs = np.asarray(qc_data["observations"], dtype=np.float32)
    qc_actions = np.asarray(qc_data["actions"], dtype=np.float32)
    # Match the repository's v0.8.2b movement detector: five-frame absolute
    # XYZ displacement, with each cube/axis threshold set to 5% of its p99.9
    # displacement. This is a movement-window rate, not a simulator estimate.
    from graphcritic.envs.ogbench_cube import decode_cube_xyz

    raw_ends = np.flatnonzero(raw_terminal)
    raw_starts = np.r_[0, raw_ends[:-1] + 1]
    movement_windows = []
    for start, end in zip(raw_starts, raw_ends, strict=True):
        xyz = decode_cube_xyz(obs[int(start) : int(end) + 1])
        if len(xyz) > 5:
            movement_windows.append(np.abs(xyz[5:] - xyz[:-5]))
    all_delta = np.concatenate(movement_windows, axis=0) if movement_windows else np.empty((0, 2, 3))
    if len(all_delta):
        q999 = np.quantile(all_delta, 0.999, axis=0)
        threshold = 0.05 * q999
        moving = np.any(all_delta > threshold[None, :, :], axis=2)
        moving_fraction_by_cube = np.mean(moving, axis=0)
        moving_fraction_any_cube = float(np.mean(np.any(moving, axis=1)))
    else:
        q999 = threshold = np.zeros((2, 3), dtype=np.float64)
        moving_fraction_by_cube = np.zeros(2, dtype=np.float64)
        moving_fraction_any_cube = 0.0
    checks = {
        "observation_dim": int(obs.shape[1]),
        "episode_count": int(len(raw_ends)),
        "state_frame_count": int(len(obs)),
        "observation_layout": {"robot_features": "[0,19)", "cube0_features": "[19,28)", "cube1_features": "[28,37)"},
        "transition_rows": int(trans.n_rows),
        "qc_rows": int(len(qc_obs)),
        "dqc_valid_transition_rows": int(len(dqc_rows)),
        "row_order_matches_qc": bool(np.array_equal(trans.feat, qc_obs)),
        "actions_match_qc": bool(np.array_equal(trans.action, qc_actions)),
        "row_order_matches_dqc": bool(np.array_equal(trans.row_idx, dqc_rows)),
        "dqc_observations_match": bool(np.array_equal(trans.feat, np.asarray(dqc_data["observations"])[dqc_rows])),
        "dqc_actions_match": bool(np.array_equal(trans.action, np.asarray(dqc_data["actions"])[dqc_rows])),
        "next_observations_match": bool(np.array_equal(trans.next_feat, np.asarray(qc_data["next_observations"]))),
        "terminal_rows_match_qc": bool(np.array_equal(trans.terminal, np.asarray(qc_data["terminals"]) > 0)),
        "terminal_rows_match_dqc": bool(np.array_equal(
            trans.terminal, np.asarray(dqc_data["terminals"])[dqc_rows] > 0
        )),
        "qc_valid_h5_matches": bool(np.array_equal(
            _qc_valid(trans.terminal, 5), _qc_valid(np.asarray(dqc_data["terminals"])[dqc_rows] > 0, 5)
        )),
        "all_task_goals_present_in_data": {
            str(task): int(np.count_nonzero(success))
            for task, success in ctx["state_success_by_task"].items()
        },
        "object_movement_v082b": {
            "method": "five-frame absolute XYZ displacement > 0.05 * per-cube-axis p99.9 displacement",
            "valid_five_frame_windows": int(len(all_delta)),
            "q999_abs_displacement_m_by_cube_xyz": q999.tolist(),
            "threshold_m_by_cube_xyz": threshold.tolist(),
            "moving_window_fraction_by_cube": moving_fraction_by_cube.tolist(),
            "moving_window_fraction_any_cube": moving_fraction_any_cube,
        },
    }
    checks["gate_passed"] = bool(
        obs.shape[1] == OBS_DIM
        and checks["row_order_matches_qc"]
        and checks["actions_match_qc"]
        and checks["row_order_matches_dqc"]
        and checks["dqc_observations_match"]
        and checks["dqc_actions_match"]
        and checks["next_observations_match"]
        and checks["terminal_rows_match_qc"]
        and checks["terminal_rows_match_dqc"]
        and checks["qc_valid_h5_matches"]
        and all(checks["all_task_goals_present_in_data"].values())
    )
    return checks


def _verify_graph(ctx: dict, graph_path: Path, candidate_path: Path) -> dict:
    graph_path = graph_path / "graph.npz" if graph_path.is_dir() else graph_path
    graph_json = graph_path.with_suffix(".json")
    result_json = graph_path.parent / "result.json"
    meta = json.loads(graph_json.read_text(encoding="utf-8"))
    arrays_file = np.load(graph_path, allow_pickle=False)
    arrays = {key: np.asarray(arrays_file[key]) for key in arrays_file.files}
    arrays_file.close()
    expected_hash = meta["graph_hash"]
    hash_meta = dict(meta)
    hash_meta.pop("graph_hash", None)
    actual_hash = _graph_hash(hash_meta, arrays)
    result = json.loads(result_json.read_text(encoding="utf-8"))
    with np.load(candidate_path, allow_pickle=False) as candidates:
        pairs = np.asarray(candidates["pairs"], dtype=np.int64)
        dpos = np.asarray(candidates["object_distance_pos_m"], dtype=np.float64)
        drot = np.asarray(candidates["object_distance_rot_rad"], dtype=np.float64)
    with np.load(graph_path.parent / "sweep.npz", allow_pickle=False) as sweep:
        accepted = np.asarray(sweep["accepted_candidate_indices"], dtype=np.int64)
        retained = np.asarray(sweep["retained_candidate_indices"], dtype=np.int64)
    delta_pos = float(meta["delta_pos_m"])
    delta_rot = np.deg2rad(float(meta["delta_rot_deg"]))
    accepted_pairs = pairs[accepted]
    retained_pairs = pairs[retained]
    fresh_pos = []
    fresh_rot = []
    objects = ctx["objects"]
    for selected in (accepted, retained):
        pos, rot = object_distance(objects[pairs[selected, 0]], objects[pairs[selected, 1]])
        fresh_pos.append(np.asarray(pos))
        fresh_rot.append(np.asarray(rot))
    dpos_violations = int(np.count_nonzero(fresh_pos[0] > delta_pos + 1e-9))
    drot_violations = int(np.count_nonzero(fresh_rot[0] > delta_rot + 1e-9))
    mismatch = compute_grasp_mismatch(ctx, accepted_pairs)
    passed = bool(
        actual_hash == expected_hash
        and dpos_violations == 0
        and drot_violations == 0
        and mismatch <= 0.05
    )
    return {
        "graph": str(graph_path.parent),
        "graph_hash_expected": expected_hash,
        "graph_hash_recomputed": actual_hash,
        "graph_hash_matches": actual_hash == expected_hash,
        "task": int(meta["task"]),
        "accepted_bridges": int(len(accepted)),
        "retained_bridges": int(len(retained)),
        "accepted_dpos_violation_count": dpos_violations,
        "accepted_drot_violation_count": drot_violations,
        "accepted_grasp_mismatch_fraction": mismatch,
        "accepted_dpos_p50_cm": float(np.median(fresh_pos[0]) * 100) if len(accepted) else None,
        "accepted_dpos_p90_cm": float(np.quantile(fresh_pos[0], 0.9) * 100) if len(accepted) else None,
        "accepted_drot_p50_deg": float(np.median(fresh_rot[0]) * 180 / np.pi) if len(accepted) else None,
        "accepted_drot_p90_deg": float(np.quantile(fresh_rot[0], 0.9) * 180 / np.pi) if len(accepted) else None,
        "retained_dpos_p50_cm": float(np.median(fresh_pos[1]) * 100) if len(retained) else None,
        "retained_dpos_p90_cm": float(np.quantile(fresh_pos[1], 0.9) * 100) if len(retained) else None,
        "retained_drot_p50_deg": float(np.median(fresh_rot[1]) * 180 / np.pi) if len(retained) else None,
        "retained_drot_p90_deg": float(np.quantile(fresh_rot[1], 0.9) * 180 / np.pi) if len(retained) else None,
        "c_reduced_node_fraction_of_all": result["c_reduced_node_fraction_of_all"],
        "c_reduction_p50_steps": result["c_reduction_p50_steps"],
        "c_reduction_p90_steps": result["c_reduction_p90_steps"],
        "pass": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-double-dataset", type=Path, required=True)
    parser.add_argument("--graph", type=Path, nargs="*", default=[])
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--alignment-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ctx = load_cube_context(args.cube_double_dataset)
    report = {
        "spec_version": "v1.8",
        "dataset_id_base": ctx["dataset_id_base"],
        "dataset_hash": ctx["dataset_hash"],
        "alignment": _alignment(ctx),
        "graphs": [],
    }
    if args.graph and not args.candidate_cache:
        parser.error("--graph requires --candidate-cache")
    for graph in args.graph:
        report["graphs"].append(_verify_graph(ctx, graph, args.candidate_cache))
    report["graph_gate_passed"] = all(item["pass"] for item in report["graphs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["alignment"]["gate_passed"] or (args.graph and not report["graph_gate_passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
