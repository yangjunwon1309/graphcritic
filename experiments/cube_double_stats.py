"""Offline statistics for the standard OGBench cube-double-play-v0 dataset.

Cube positions come from the low-dimensional observation (which stores XYZ in
units scaled by 10 around [0.425, 0, 0]). No simulator is created or stepped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree


ENV_NAME = "cube-double-play-v0"
OBS_DIM = 37
ROBOT_SLICE = (0, 19)
CUBE_FEATURE_SLICES = ((19, 28), (28, 37))
CUBE_XYZ_INDICES = ((19, 20, 21), (28, 29, 30))
XYZ_CENTER = np.asarray([0.425, 0.0, 0.0], dtype=np.float64)
XYZ_SCALER = 10.0
GOAL_TOLERANCE_M = 0.04
MOVE_WINDOW = 5
MOVE_Q = 0.999
MOVE_EPS_FRACTION = 0.05
MERGE_GAP_FRAMES = 5
MIN_STATIONARY_FRAMES = 5


def task_goals_from_ogbench() -> dict[int, np.ndarray]:
    """Read task goals from OGBench's CubeEnv definition without making an env."""
    try:
        from ogbench.manipspace.envs.cube_env import CubeEnv
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("run with the bench environment containing OGBench") from exc
    definition = CubeEnv.__new__(CubeEnv)
    definition._env_type = "double"
    definition._reward_task_id = None
    definition.set_tasks()
    return {
        index + 1: np.asarray(info["goal_xyzs"], dtype=np.float64)
        for index, info in enumerate(definition.task_infos)
    }


def decode_cube_xyz(observations: np.ndarray) -> np.ndarray:
    """Decode shape (N, 37) observations to raw metric XYZ, shape (N, 2, 3)."""
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(f"expected observations with shape (N, {OBS_DIM}), got {obs.shape}")
    xyz_scaled = np.stack((obs[:, 19:22], obs[:, 28:31]), axis=1)
    return xyz_scaled / XYZ_SCALER + XYZ_CENTER


def _episode_state_sequences(dataset: dict[str, np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    obs = np.asarray(dataset["observations"])
    next_obs = np.asarray(dataset["next_observations"])
    terminals = np.asarray(dataset["terminals"]).astype(bool)
    if obs.ndim != 2 or next_obs.shape != obs.shape or terminals.shape != (len(obs),):
        raise ValueError("OGBench observations, next_observations, and terminals are not aligned")
    if obs.shape[1] != OBS_DIM:
        raise ValueError(f"cube-double low-dimensional observations must be {OBS_DIM}D, got {obs.shape[1]}")
    ends = np.flatnonzero(terminals)
    if not len(ends) or ends[-1] != len(obs) - 1:
        raise ValueError("expected each OGBench episode's last transition to be terminal")
    starts = np.r_[0, ends[:-1] + 1]
    sequences: list[np.ndarray] = []
    lengths: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        # Current observations plus the terminal next observation form the
        # contiguous physical state sequence for this episode.
        sequence = np.concatenate((obs[start : end + 1], next_obs[end : end + 1]), axis=0)
        sequences.append(sequence)
        lengths.append(int(end - start + 1))
    return sequences, np.asarray(lengths, dtype=np.int64)


def _group_active_chunk_starts(active: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    times = np.flatnonzero(np.asarray(active, dtype=bool))
    if not len(times):
        return []
    result: list[tuple[int, int]] = []
    start = previous = int(times[0])
    for value in times[1:]:
        current = int(value)
        if current - previous - 1 < merge_gap:
            previous = current
        else:
            result.append((start, previous))
            start = previous = current
    result.append((start, previous))
    return result


def _goal_success(positions: np.ndarray, goals: np.ndarray, tolerance: float) -> np.ndarray:
    """Assignment-invariant two-cube success for rows (..., 2, 3)."""
    d00 = np.linalg.norm(positions[..., 0, :] - goals[0], axis=-1)
    d01 = np.linalg.norm(positions[..., 0, :] - goals[1], axis=-1)
    d10 = np.linalg.norm(positions[..., 1, :] - goals[0], axis=-1)
    d11 = np.linalg.norm(positions[..., 1, :] - goals[1], axis=-1)
    direct = (d00 <= tolerance) & (d11 <= tolerance)
    swapped = (d01 <= tolerance) & (d10 <= tolerance)
    return direct | swapped


def _pair_match_metrics(
    positions: np.ndarray, episode_ids: np.ndarray, radius: float
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Find cross-episode layout pairs matching both or at least one cube.

    Cube ordering is treated as exchangeable. Returns unordered layout pairs
    matching both cubes, and (layout_i, layout_j, matched_cube_count) pairs
    matching at least one cube under the better of the two assignments.
    """
    n = len(positions)
    if n < 2:
        return [], []
    flat = positions.reshape(-1, 3)
    tree = cKDTree(flat)
    both_pairs: list[tuple[int, int]] = []
    one_or_more: list[tuple[int, int, int]] = []
    candidate_pairs: set[tuple[int, int]] = set()
    search_radius = float(radius) * (1.0 + 1e-12)
    for i in range(n):
        for cube in range(2):
            near_points = tree.query_ball_point(positions[i, cube], search_radius)
            for point_id in near_points:
                j = int(point_id // 2)
                if j > i and episode_ids[j] != episode_ids[i]:
                    candidate_pairs.add((i, j))
    for i, j in sorted(candidate_pairs):
        direct = np.linalg.norm(positions[i] - positions[j], axis=1) <= search_radius
        swapped = np.linalg.norm(positions[i] - positions[j, ::-1], axis=1) <= search_radius
        direct_count = int(np.count_nonzero(direct))
        swapped_count = int(np.count_nonzero(swapped))
        match_count = max(direct_count, swapped_count)
        if match_count:
            one_or_more.append((i, j, match_count))
        if match_count == 2:
            both_pairs.append((i, j))
    return both_pairs, one_or_more


def _backward_reachable(
    n_nodes: int,
    episode_nodes: list[list[int]],
    cross_pairs: list[tuple[int, int]],
    goals: np.ndarray,
) -> np.ndarray:
    reverse: list[list[int]] = [[] for _ in range(n_nodes)]
    for node_ids in episode_nodes:
        for src, dst in zip(node_ids[:-1], node_ids[1:]):
            reverse[dst].append(src)
    for first, second in cross_pairs:
        reverse[first].append(second)
        reverse[second].append(first)
    reached = np.zeros(n_nodes, dtype=bool)
    queue = [int(node) for node in np.flatnonzero(goals)]
    reached[queue] = True
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for pred in reverse[node]:
            if not reached[pred]:
                reached[pred] = True
                queue.append(pred)
    return reached


def analyze_cube_double(dataset: dict[str, np.ndarray], goals: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    """Compute movement, stable-layout, goal, matching, and connectivity stats."""
    goals = task_goals_from_ogbench() if goals is None else goals
    sequences, episode_lengths = _episode_state_sequences(dataset)
    per_episode_xyz = [decode_cube_xyz(sequence) for sequence in sequences]

    delta_blocks = [np.abs(xyz[MOVE_WINDOW:] - xyz[:-MOVE_WINDOW]) for xyz in per_episode_xyz if len(xyz) > MOVE_WINDOW]
    if not delta_blocks:
        raise ValueError("dataset has no valid five-frame displacement windows")
    all_delta = np.concatenate(delta_blocks, axis=0)
    q999 = np.quantile(all_delta, MOVE_Q, axis=0)
    epsilon = MOVE_EPS_FRACTION * q999

    episode_ids: list[int] = []
    layout_positions: list[np.ndarray] = []
    episode_layout_nodes: list[list[int]] = []
    transport_lengths: list[int] = []
    transport_speeds_cm_frame: list[float] = []
    per_cube_moving = np.zeros(2, dtype=np.int64)
    valid_chunk_count = 0

    for episode_id, (xyz, n_transitions) in enumerate(zip(per_episode_xyz, episode_lengths, strict=True)):
        del n_transitions
        chunk_delta = np.abs(xyz[MOVE_WINDOW:] - xyz[:-MOVE_WINDOW])
        moved_by_cube = np.any(chunk_delta > epsilon[None, :, :], axis=2)
        active = np.any(moved_by_cube, axis=1)
        valid_chunk_count += len(active)
        per_cube_moving += np.count_nonzero(moved_by_cube, axis=0)
        bouts = _group_active_chunk_starts(active, MERGE_GAP_FRAMES)

        # A five-frame displacement window beginning at t covers through t+5.
        # The inactive intervals between merged movement bouts are the stable
        # object layouts used as graph nodes.
        stable_ranges: list[tuple[int, int]] = []
        cursor = 0
        for start, last_active in bouts:
            covered_end = min(len(xyz) - 1, last_active + MOVE_WINDOW)
            stable_end = start - 1
            if stable_end - cursor + 1 >= MIN_STATIONARY_FRAMES:
                stable_ranges.append((cursor, stable_end))
            cursor = max(cursor, covered_end + 1)

            duration = last_active - start + MOVE_WINDOW
            if duration > 0:
                end_frame = min(last_active + MOVE_WINDOW, len(xyz) - 1)
                path_lengths = np.sum(np.linalg.norm(np.diff(xyz[start : end_frame + 1], axis=0), axis=2), axis=0)
                transport_lengths.append(int(duration))
                transport_speeds_cm_frame.append(float(np.sum(path_lengths) * 100.0 / duration))
        if len(xyz) - cursor >= MIN_STATIONARY_FRAMES:
            stable_ranges.append((cursor, len(xyz) - 1))

        nodes_for_episode: list[int] = []
        for start, end in stable_ranges:
            node_id = len(layout_positions)
            nodes_for_episode.append(node_id)
            episode_ids.append(episode_id)
            layout_positions.append(np.median(xyz[start : end + 1], axis=0))
        episode_layout_nodes.append(nodes_for_episode)

    layouts = np.asarray(layout_positions, dtype=np.float64).reshape(-1, 2, 3)
    layout_episode_ids = np.asarray(episode_ids, dtype=np.int64)
    stationary_counts: dict[str, Any] = {
        "layout_count": int(len(layouts)),
        "minimum_retained_stable_frames": MIN_STATIONARY_FRAMES,
        "movement_bout_gap_merge_frames": MERGE_GAP_FRAMES,
        "stationary_layout_definition": "stable frame interval between merged active five-frame displacement bouts; median XYZ representative",
    }

    # All-state task success counts include both current and terminal next
    # observations, and therefore count frames rather than transitions.
    frame_xyz = np.concatenate(per_episode_xyz, axis=0) if per_episode_xyz else np.empty((0, 2, 3))
    frame_offsets = np.cumsum([0] + [len(x) for x in per_episode_xyz[:-1]])
    task_goal_stats: dict[str, Any] = {}
    stacked_layout = (
        (np.linalg.norm(layouts[:, 0, :2] - layouts[:, 1, :2], axis=1) < 0.02)
        & (np.abs(np.abs(layouts[:, 0, 2] - layouts[:, 1, 2]) - 0.04) <= 0.01)
    ) if len(layouts) else np.zeros(0, dtype=bool)
    for task_id, task_goals in sorted(goals.items()):
        frame_success = _goal_success(frame_xyz, np.asarray(task_goals), GOAL_TOLERANCE_M)
        episode_success_count = 0
        for sequence, offset in zip(per_episode_xyz, frame_offsets, strict=True):
            episode_success_count += int(np.any(frame_success[offset : offset + len(sequence)]))
        layout_success = _goal_success(layouts, np.asarray(task_goals), GOAL_TOLERANCE_M)
        close_to_any_goal = np.min(
            np.linalg.norm(layouts[:, :, None, :] - np.asarray(task_goals)[None, None, :, :], axis=-1),
            axis=2,
        ) <= GOAL_TOLERANCE_M if len(layouts) else np.zeros((0, 2), dtype=bool)
        exactly_one_cube_close = np.count_nonzero(close_to_any_goal, axis=1) == 1
        task_goal_stats[str(task_id)] = {
            "goal_xyzs_m": np.asarray(task_goals, dtype=float).tolist(),
            "success_tolerance_m": GOAL_TOLERANCE_M,
            "successful_frame_count": int(np.count_nonzero(frame_success)),
            "episodes_with_successful_frame": int(episode_success_count),
            "successful_stationary_layout_count": int(np.count_nonzero(layout_success)),
            "stationary_layouts_with_exactly_one_cube_within_goal_tolerance": int(np.count_nonzero(exactly_one_cube_close)),
            "stationary_layouts_with_exactly_one_cube_within_goal_tolerance_fraction": (
                float(np.mean(exactly_one_cube_close)) if len(layouts) else 0.0
            ),
        }

    pair_stats: dict[str, Any] = {}
    connectivity: dict[str, Any] = {}
    for radius_cm in (1, 2, 4):
        radius = radius_cm / 100.0
        both_pairs, one_cube_pairs = _pair_match_metrics(layouts, layout_episode_ids, radius)
        n = len(layouts)
        pair_stats[str(radius_cm)] = {
            "radius_cm": radius_cm,
            "both_cubes_matching_pair_count": len(both_pairs),
            "both_cubes_mean_matching_partners_per_layout": (2.0 * len(both_pairs) / n if n else 0.0),
            "at_least_one_cube_matching_pair_count": len(one_cube_pairs),
            "at_least_one_cube_mean_matching_partners_per_layout": (2.0 * len(one_cube_pairs) / n if n else 0.0),
            "total_individual_cube_matches_over_layout_pairs": int(sum(count for _, _, count in one_cube_pairs)),
            "pair_definition": "unordered cube assignment; cross-episode only; each matched cube's 3D Euclidean error <= radius",
        }
        both_pair_ids = [(int(i), int(j)) for i, j in both_pairs]
        per_task: dict[str, Any] = {}
        for task_id, task_goals in sorted(goals.items()):
            goal_nodes = _goal_success(layouts, np.asarray(task_goals), GOAL_TOLERANCE_M)
            reached = _backward_reachable(n, episode_layout_nodes, both_pair_ids, goal_nodes)
            per_task[str(task_id)] = {
                "goal_layout_count": int(np.count_nonzero(goal_nodes)),
                "goal_reachable_layout_count": int(np.count_nonzero(reached)),
                "goal_reachable_fraction": float(np.mean(reached)) if n else 0.0,
            }
        connectivity[str(radius_cm)] = {
            "radius_cm": radius_cm,
            "node_count": n,
            "same_episode_consecutive_layout_edge_count": int(sum(max(0, len(nodes) - 1) for nodes in episode_layout_nodes)),
            "cross_episode_bidirectional_match_pair_count": len(both_pairs),
            "directed_edge_count_including_both_cross_match_directions": int(
                sum(max(0, len(nodes) - 1) for nodes in episode_layout_nodes) + 2 * len(both_pairs)
            ),
            "task_goal_reachability": per_task,
        }

    total_transitions = int(len(dataset["observations"]))
    transition_lengths = episode_lengths.astype(np.float64)
    frame_lengths = transition_lengths + 1.0
    moved_fraction = per_cube_moving / max(valid_chunk_count, 1)
    report: dict[str, Any] = {
        "dataset": {
            "name": ENV_NAME,
            "source_file": str(dataset.get("source_file", "")),
            "episode_count": int(len(episode_lengths)),
            "transition_count": total_transitions,
            "episode_length_transitions": {
                "min": int(np.min(transition_lengths)) if len(transition_lengths) else 0,
                "p10": float(np.quantile(transition_lengths, 0.10)) if len(transition_lengths) else 0.0,
                "p50": float(np.quantile(transition_lengths, 0.50)) if len(transition_lengths) else 0.0,
                "p90": float(np.quantile(transition_lengths, 0.90)) if len(transition_lengths) else 0.0,
                "max": int(np.max(transition_lengths)) if len(transition_lengths) else 0,
            },
            "episode_length_state_frames": {
                "min": int(np.min(frame_lengths)) if len(frame_lengths) else 0,
                "p10": float(np.quantile(frame_lengths, 0.10)) if len(frame_lengths) else 0.0,
                "p50": float(np.quantile(frame_lengths, 0.50)) if len(frame_lengths) else 0.0,
                "p90": float(np.quantile(frame_lengths, 0.90)) if len(frame_lengths) else 0.0,
                "max": int(np.max(frame_lengths)) if len(frame_lengths) else 0,
            },
            "observation_dim": int(np.asarray(dataset["observations"]).shape[1]),
            "observation_layout": {
                "robot_features_half_open": list(ROBOT_SLICE),
                "cube_0_features_half_open": list(CUBE_FEATURE_SLICES[0]),
                "cube_1_features_half_open": list(CUBE_FEATURE_SLICES[1]),
                "cube_xyz_indices": [list(index) for index in CUBE_XYZ_INDICES],
                "cube_xyz_decode": "raw_xyz_m = obs_xyz / 10 + [0.425, 0, 0]",
            },
        },
        "movement": {
            "window_frames": MOVE_WINDOW,
            "quantile": MOVE_Q,
            "threshold_fraction_of_quantile": MOVE_EPS_FRACTION,
            "q999_abs_displacement_m_by_cube_xyz": q999.tolist(),
            "movement_threshold_m_by_cube_xyz": epsilon.tolist(),
            "valid_five_frame_windows": int(valid_chunk_count),
            "moving_window_count_by_cube": per_cube_moving.tolist(),
            "moving_window_fraction_by_cube": moved_fraction.tolist(),
            "move_bout_count": len(transport_lengths),
            "move_bout_duration_frames_p10_p50_p90": (
                [float(np.quantile(transport_lengths, q)) for q in (0.10, 0.50, 0.90)]
                if transport_lengths else [None, None, None]
            ),
            "transport_path_speed_cm_per_frame_p50": (
                float(np.median(transport_speeds_cm_frame)) if transport_speeds_cm_frame else None
            ),
            "stationary_layouts": stationary_counts,
            "stacked_stationary_layout_count": int(np.count_nonzero(stacked_layout)),
            "stacked_stationary_layout_fraction": float(np.mean(stacked_layout)) if len(layouts) else 0.0,
            "stack_definition": "XY distance <2 cm and vertical separation within 1 cm of the 4 cm cube edge",
        },
        "task_goal_statistics": task_goal_stats,
        "cross_episode_object_state_matches": pair_stats,
        "stationary_layout_graph_connectivity": connectivity,
    }
    return report


def _load_standard_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset not found: {path}. Download it with `pixi run -e bench -- python -c "
            "'from ogbench.utils import download_datasets; download_datasets([\"cube-double-play-v0\"])'`"
        )
    from ogbench.utils import load_dataset

    dataset = load_dataset(str(path), compact_dataset=False, add_info=True)
    dataset["source_file"] = str(path)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("~/.ogbench/data/cube-double-play-v0.npz").expanduser(),
        help="standard OGBench training archive (default: ~/.ogbench/data/cube-double-play-v0.npz)",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    args = parser.parse_args()
    report = analyze_cube_double(_load_standard_dataset(args.dataset))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
