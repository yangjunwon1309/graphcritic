#!/usr/bin/env python3
"""Diagnose held-out v1.8 dθ chunk violations by joint speed and cube motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.train_quasimetric import _targets_at_offset
from experiments.v08_runtime import load_quasimetric_checkpoint, make_distance_predictor
from graphcritic.bridges.models import prepare_cube_robot_model_data
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.envs.ogbench_cube import decode_cube_xyz


def _quantile_report(values: np.ndarray, distances: np.ndarray, *, threshold: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "n": int(len(distances)),
        "joint_vel_norm_min": float(np.min(values)) if len(values) else None,
        "joint_vel_norm_max": float(np.max(values)) if len(values) else None,
        "violation_rate_d_gt_6_25": float(np.mean(distances > threshold)) if len(distances) else None,
        "distance_p50": float(np.quantile(distances, 0.50)) if len(distances) else None,
        "distance_p90": float(np.quantile(distances, 0.90)) if len(distances) else None,
    }


def _movement_threshold(dataset_path: Path) -> np.ndarray:
    with np.load(dataset_path, allow_pickle=False) as archive:
        observations = np.asarray(archive["observations"], dtype=np.float32)
        terminals = np.asarray(archive["terminals"], dtype=bool)
    ends = np.flatnonzero(terminals)
    starts = np.r_[0, ends[:-1] + 1]
    deltas = []
    for start, end in zip(starts, ends, strict=True):
        xyz = decode_cube_xyz(observations[int(start):int(end) + 1])
        if len(xyz) > 5:
            deltas.append(np.abs(xyz[5:] - xyz[:-5]))
    if not deltas:
        raise ValueError("dataset has no five-frame cube displacement windows")
    return 0.05 * np.quantile(np.concatenate(deltas, axis=0), 0.999, axis=0)


def diagnose(dataset_path: Path, checkpoint: Path, *, batch_size: int = 8192) -> dict:
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("window", -1)) != 5 or int(metadata.get("constraint_horizon", -1)) != 5:
        raise ValueError("v1.8 violation diagnostic requires a window-5, horizon-5 checkpoint")
    trans, transition_meta, data = prepare_cube_robot_model_data(str(dataset_path), seed=0)
    model, params, metadata = load_quasimetric_checkpoint(
        checkpoint, input_dim=95, expected_dataset_id=transition_meta["dataset_id"]
    )
    if metadata.get("input_part") != "robot" or int(metadata.get("frame_dim", -1)) != 19:
        raise ValueError("checkpoint must use robot-only 19D frames")
    states = puzzle_state_windows(
        data.state_cont, trans, data.terminal_state_index, window=5
    )
    raw_windows = puzzle_state_windows(
        data.state_raw, trans, data.terminal_state_index, window=5
    ).reshape(-1, 5, 19)
    rows, endpoints = _targets_at_offset(data.heldout_rows, data, 5)
    if not len(rows):
        raise ValueError("held-out split has no valid five-step pairs")
    predictor = make_distance_predictor(model, params, batch_size=batch_size)
    distances = predictor(states[rows], states[endpoints])

    source_joint_velocity = raw_windows[rows, :, 6:12]
    max_joint_speed = np.max(np.linalg.norm(source_joint_velocity, axis=2), axis=1)
    order = np.argsort(max_joint_speed, kind="stable")
    decile_indices = np.array_split(order, 10)
    deciles = []
    threshold_d = 6.25
    for index, ids in enumerate(decile_indices, start=1):
        deciles.append({
            "decile": index,
            "speed_p10": float(np.quantile(max_joint_speed[ids], 0.10)),
            "speed_p90": float(np.quantile(max_joint_speed[ids], 0.90)),
            **_quantile_report(max_joint_speed[ids], distances[ids], threshold=threshold_d),
        })

    move_threshold = _movement_threshold(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        observations = np.asarray(archive["observations"], dtype=np.float32)
    xyz = decode_cube_xyz(observations)
    raw_start = trans.row_idx[rows]
    delta = np.abs(xyz[raw_start + 5] - xyz[raw_start])
    moved = np.any(delta > move_threshold[None, :, :], axis=(1, 2))
    moving_stationary = {
        label: _quantile_report(max_joint_speed[mask], distances[mask], threshold=threshold_d)
        for label, mask in (("moving", moved), ("stationary", ~moved))
    }
    return {
        "dataset": str(dataset_path.resolve()),
        "dataset_id": transition_meta["dataset_id"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": int(metadata["step"]),
        "checkpoint_params_sha256": metadata.get("params_sha256"),
        "constraint_threshold": threshold_d,
        "pair_window_definition": "source model input's trailing 5 raw frames; max L2 norm of robot joint_vel observation indices [6,12)",
        "movement_definition": "cube endpoint displacement over five frames exceeds 0.05 times per-cube/per-axis p99.9 absolute displacement, matching S1 audit",
        "cube_movement_threshold_m_by_cube_xyz": move_threshold.tolist(),
        "heldout_pair_count": int(len(rows)),
        "distance_summary": _quantile_report(max_joint_speed, distances, threshold=threshold_d),
        "joint_velocity_deciles": deciles,
        "cube_moving_stationary": moving_stationary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-double-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    result = diagnose(args.cube_double_dataset, args.checkpoint, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "heldout_pair_count": result["heldout_pair_count"],
                      "violation_rate": result["distance_summary"]["violation_rate_d_gt_6_25"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
