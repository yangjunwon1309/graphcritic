"""Puzzle object-state features and v0.8.2 contact labels.

The object judges see only physical object state: exact button bits, button
depth, and button velocity. Movement labels use position and discrete state;
velocity remains an input feature but is excluded from label generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from graphcritic.bridges.models import PuzzleModelData, prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.featurize import state_table
from graphcritic.puzzle import toggle_matrix
from graphcritic.state_schema import OGBenchPuzzleStateSchema
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


@dataclass(frozen=True)
class PuzzleObjectData:
    """Object features, five-frame labels, and contact-boundary pairs."""

    state_raw: np.ndarray
    frame_features: np.ndarray
    windows: np.ndarray
    moved: np.ndarray
    onset: np.ndarray
    chunk_endpoint: np.ndarray
    valid_chunk_rows: np.ndarray
    train_rows: np.ndarray
    heldout_rows: np.ndarray
    train_episodes: np.ndarray
    heldout_episodes: np.ndarray
    episode_end_rows: np.ndarray
    contact_start_rows: np.ndarray
    contact_end_rows: np.ndarray
    event_source_ids: np.ndarray
    event_target_ids: np.ndarray
    event_interval_indices: np.ndarray
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    epsilon_object: np.ndarray
    movement_report: dict[str, Any]
    onset_report: dict[str, Any]
    label_report: dict[str, Any]
    metadata: dict[str, Any]


def puzzle_object_state_raw(
    trans: TransitionSet, sim_state: SimStateSet
) -> np.ndarray:
    """Return the 64D P13 vector: 32 button bits, 16 depths, 16 velocities."""
    states, _, _, _ = state_table(trans)
    sim_state.validate(len(states))
    if sim_state.button_states is None or sim_state.button_states.shape[1] != 16:
        raise SchemaError("v0.8.2 object models require aligned 4x4 button states")
    if sim_state.qvel.shape[1] < 30:
        raise SchemaError("v0.8.2 object models require 16 button joint velocities in qvel")
    result = OGBenchPuzzleStateSchema().objects(states, {
        "qpos": sim_state.qpos,
        "qvel": sim_state.qvel,
        "button_states": sim_state.button_states,
    })
    if result.shape != (len(states), 64):
        raise SchemaError("4x4 Puzzle object state must have width 64")
    return np.ascontiguousarray(result, dtype=np.float32)


def normalize_object_state(
    raw: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Normalize depth/velocity dimensions and preserve exact one-hot bits."""
    values = np.asarray(raw, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 64 or mean.shape != (32,) or std.shape != (32,):
        raise SchemaError("Puzzle object normalization expects 64D states and 32D stats")
    if (np.any(~np.isfinite(values)) or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(std)) or np.any(std <= 0)):
        raise SchemaError("Puzzle object state and normalization statistics must be finite")
    normalized = values.copy()
    normalized[:, 32:] = (values[:, 32:] - mean) / std
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _five_step_endpoints(data: PuzzleModelData, trans: TransitionSet) -> tuple[np.ndarray, np.ndarray]:
    """Map valid t→t+5 transition starts to state-table endpoint IDs."""
    rows = np.arange(trans.n_rows, dtype=np.int64)
    end_rows = data.episode_end_rows
    target = rows + 5
    terminal_target = end_rows + 1
    valid = (target <= end_rows) | (target == terminal_target)
    result = target.copy()
    terminal_mask = target == terminal_target
    if np.any(terminal_mask):
        result[terminal_mask] = data.terminal_state_index[terminal_mask]
        if np.any(result[terminal_mask] < 0):
            raise SchemaError("five-step object chunk has no aligned terminal sidecar state")
    return valid, result


def _fit_rows(data: PuzzleModelData, trans: TransitionSet) -> np.ndarray:
    terminal_rows = np.flatnonzero(trans.terminal)
    train_terminal = terminal_rows[np.isin(trans.episode_id[terminal_rows], data.train_episodes)]
    terminal_state_ids = trans.n_rows + np.searchsorted(terminal_rows, train_terminal)
    return np.concatenate((data.train_rows, terminal_state_ids)).astype(np.int64)


def _state_time_lookup(trans: TransitionSet) -> tuple[dict[tuple[int, int], int], np.ndarray, np.ndarray]:
    from graphcritic.modes import state_episode_time

    episodes, times = state_episode_time(trans)
    lookup: dict[tuple[int, int], int] = {}
    for state_id, (episode_id, time) in enumerate(zip(episodes, times, strict=True)):
        key = (int(episode_id), int(time))
        if key in lookup:
            raise SchemaError(f"duplicate puzzle state occurrence at episode/time {key}")
        lookup[key] = state_id
    return lookup, episodes, times


def _group_contact_intervals(
    active_times: np.ndarray, *, merge_gap_frames: int = 5
) -> list[tuple[int, int]]:
    """Group moved chunk starts; merge gaps containing fewer than five frames."""
    times = np.unique(np.asarray(active_times, dtype=np.int64))
    if not len(times):
        return []
    intervals: list[tuple[int, int]] = []
    start = previous = int(times[0])
    for value in times[1:]:
        current = int(value)
        empty_frames = current - previous - 1
        if empty_frames < merge_gap_frames:
            previous = current
            continue
        intervals.append((start, previous))
        start = previous = current
    intervals.append((start, previous))
    return intervals


def _stationary_window(
    episode_id: int,
    start_time: int,
    end_time: int,
    *,
    lookup: dict[tuple[int, int], int],
    depths: np.ndarray,
    buttons: np.ndarray,
    epsilon: np.ndarray,
) -> bool:
    """Check that all five observed frames in a window are stationary."""
    ids = [lookup.get((episode_id, time)) for time in range(start_time, end_time + 1)]
    if len(ids) != 5 or any(state_id is None for state_id in ids):
        return False
    state_ids = np.asarray(ids, dtype=np.int64)
    depth_delta = np.abs(np.diff(depths[state_ids], axis=0))
    bits_static = np.all(buttons[state_ids] == buttons[state_ids[0]])
    return bool(bits_static and np.all(depth_delta <= epsilon[None, :]))


def _board_change_records(
    episodes: np.ndarray, times: np.ndarray, buttons: np.ndarray
) -> list[tuple[int, int, np.ndarray]]:
    changes: list[tuple[int, int, np.ndarray]] = []
    for episode_id in np.unique(episodes):
        ids = np.flatnonzero(episodes == episode_id)
        ids = ids[np.argsort(times[ids], kind="stable")]
        for previous, current in zip(ids[:-1], ids[1:], strict=True):
            difference = np.bitwise_xor(buttons[previous], buttons[current])
            if np.any(difference):
                changes.append((int(episode_id), int(times[current]), difference.copy()))
    return changes


def _label_diagnostics(
    *,
    intervals: list[tuple[int, int, int, int]],
    event_pair_interval_indices: list[int],
    board_changes: list[tuple[int, int, np.ndarray]],
    toggle_patterns: np.ndarray,
    moved: np.ndarray,
    valid_chunk_rows: np.ndarray,
    episode_ids: np.ndarray,
    times: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return D0 gates and the contact-start interval statistics."""
    covered_changes: set[int] = set()
    exact_single_pattern = 0
    intervals_with_changes = 0
    starts_by_episode: dict[int, list[int]] = {}
    lengths: list[int] = []
    for episode_id, start, end, _ in intervals:
        starts_by_episode.setdefault(episode_id, []).append(start)
        lengths.append(end - start + 1)
        indices = [
            index for index, (ep, time, _) in enumerate(board_changes)
            if ep == episode_id and start <= time <= end + 1
        ]
        covered_changes.update(indices)
        if indices:
            intervals_with_changes += 1
            if (len(indices) == 1 and any(
                np.array_equal(board_changes[indices[0]][2], pattern)
                for pattern in toggle_patterns
            )):
                exact_single_pattern += 1

    board_count = len(board_changes)
    contact_count = len(intervals)
    count_relative_error = (
        abs(contact_count - board_count) / board_count if board_count else None
    )
    exact_fraction = (
        exact_single_pattern / intervals_with_changes if intervals_with_changes else 0.0
    )
    uncovered_count = board_count - len(covered_changes)
    uncovered_fraction = uncovered_count / board_count if board_count else 1.0

    start_gaps = [
        float(b - a)
        for starts in starts_by_episode.values()
        for a, b in zip(sorted(starts)[:-1], sorted(starts)[1:], strict=True)
    ]
    if start_gaps:
        gap_array = np.asarray(start_gaps, dtype=np.float64)
        onset_report: dict[str, Any] = {
            "interval_count": contact_count,
            "minimum_frames": float(np.min(gap_array)),
            "p1_frames": float(np.quantile(gap_array, 0.01)),
            "p5_frames": float(np.quantile(gap_array, 0.05)),
            "start_gap_count": int(len(gap_array)),
        }
    else:
        onset_report = {
            "interval_count": contact_count,
            "minimum_frames": None,
            "p1_frames": None,
            "p5_frames": None,
            "start_gap_count": 0,
        }
    length_report = (
        {
            "count": len(lengths),
            "p10": float(np.quantile(lengths, 0.10)),
            "p50": float(np.quantile(lengths, 0.50)),
            "p90": float(np.quantile(lengths, 0.90)),
        }
        if lengths else {"count": 0, "p10": None, "p50": None, "p90": None}
    )
    p1 = onset_report["p1_frames"]
    gate = {
        "contact_count_matches_board_changes": bool(
            count_relative_error is not None and count_relative_error <= 0.10
        ),
        "single_toggle_pattern_fraction_at_least_0_95": bool(exact_fraction >= 0.95),
        "uncovered_board_change_fraction_at_most_0_01": bool(uncovered_fraction <= 0.01),
        "T_I_p1_between_20_and_40_frames": bool(p1 is not None and 20.0 <= p1 <= 40.0),
    }
    label_report = {
        "contact_interval_count": int(contact_count),
        "board_change_count": int(board_count),
        "contact_count_relative_error": count_relative_error,
        "intervals_with_board_changes": int(intervals_with_changes),
        "single_toggle_pattern_count": int(exact_single_pattern),
        "single_toggle_pattern_fraction": float(exact_fraction),
        "uncovered_board_change_count": int(uncovered_count),
        "uncovered_board_change_fraction": float(uncovered_fraction),
        "event_pair_count": int(len(event_pair_interval_indices)),
        "event_pair_interval_fraction": (
            float(len(event_pair_interval_indices) / contact_count) if contact_count else 0.0
        ),
        "moved_chunk_count": int(np.count_nonzero(moved[valid_chunk_rows])),
        "valid_chunk_count": int(len(valid_chunk_rows)),
        "moved_fraction": (
            float(np.mean(moved[valid_chunk_rows])) if len(valid_chunk_rows) else 0.0
        ),
        "contact_length_frames": length_report,
        "T_I": onset_report,
        "gates": gate,
        "gate_passed": bool(all(gate.values())),
    }
    return label_report, onset_report


def prepare_puzzle_object_data(
    trans: TransitionSet,
    sim_state: SimStateSet,
    *,
    seed: int = 0,
    window: int = 5,
    train_fraction: float = 0.9,
) -> PuzzleObjectData:
    """Prepare split-safe object windows, revised movement labels, and events."""
    if window != 5:
        raise ValueError("v0.8.2 fixes the object-state input window to five frames")
    split = prepare_puzzle_model_data(
        trans, sim_state, seed=seed, train_fraction=train_fraction
    )
    raw = puzzle_object_state_raw(trans, sim_state)
    fit_ids = _fit_rows(split, trans)
    continuous = raw[:, 32:].astype(np.float64)
    mean = np.mean(continuous[fit_ids], axis=0, dtype=np.float64)
    std = np.maximum(np.std(continuous[fit_ids], axis=0, dtype=np.float64), 1e-6)
    normalized = normalize_object_state(raw, mean, std)
    windows = puzzle_state_windows(
        normalized, trans, split.terminal_state_index, window=window
    )

    valid, endpoint = _five_step_endpoints(split, trans)
    rows = np.flatnonzero(valid).astype(np.int64)
    buttons = np.asarray(sim_state.button_states, dtype=np.uint8)
    depths = raw[:, 32:48].astype(np.float64)
    delta = depths[endpoint[rows]] - depths[rows]
    displacement = np.abs(delta)
    q999 = np.quantile(displacement, 0.999, axis=0)
    epsilon = 0.05 * q999
    button_changed = np.any(buttons[endpoint[rows]] != buttons[rows], axis=1)
    moved = np.zeros(trans.n_rows, dtype=np.bool_)
    moved[rows] = np.any(displacement > epsilon[None, :], axis=1) | button_changed

    lookup, state_episode_ids, state_times = _state_time_lookup(trans)
    active_times_by_episode: dict[int, np.ndarray] = {}
    for episode_id in np.unique(trans.episode_id):
        episode_rows = np.flatnonzero(trans.episode_id == episode_id)
        active = episode_rows[moved[episode_rows]]
        active_times_by_episode[int(episode_id)] = trans.t[active].astype(np.int64)

    intervals: list[tuple[int, int, int, int]] = []
    for episode_id, active_times in active_times_by_episode.items():
        for start_time, end_time in _group_contact_intervals(active_times, merge_gap_frames=5):
            start_row = lookup.get((episode_id, start_time))
            end_row = lookup.get((episode_id, end_time))
            if start_row is None or end_row is None or start_row >= trans.n_rows or end_row >= trans.n_rows:
                raise SchemaError("contact interval endpoints must map to transition rows")
            intervals.append((episode_id, start_time, end_time, start_row))

    event_sources: list[int] = []
    event_targets: list[int] = []
    event_interval_indices: list[int] = []
    for interval_index, (episode_id, start_time, end_time, start_row) in enumerate(intervals):
        before_start, before_end = start_time - 4, start_time
        after_start, after_end = end_time + 5, end_time + 9
        before_id = lookup.get((episode_id, before_end))
        after_id = lookup.get((episode_id, after_end))
        if before_id is None or after_id is None:
            continue
        if not _stationary_window(
            episode_id, before_start, before_end, lookup=lookup,
            depths=depths, buttons=buttons, epsilon=epsilon,
        ):
            continue
        if not _stationary_window(
            episode_id, after_start, after_end, lookup=lookup,
            depths=depths, buttons=buttons, epsilon=epsilon,
        ):
            continue
        event_sources.append(int(before_id))
        event_targets.append(int(after_id))
        event_interval_indices.append(interval_index)

    contact_start_rows = np.asarray([item[3] for item in intervals], dtype=np.int64)
    contact_end_rows = np.asarray([
        lookup[(episode_id, end_time)]
        for episode_id, _, end_time, _ in intervals
    ], dtype=np.int64)
    onset = np.zeros(trans.n_rows, dtype=np.bool_)
    if event_sources:
        onset[np.asarray(event_sources, dtype=np.int64)] = True

    episode_ids, state_times = state_episode_ids, state_times
    board_changes = _board_change_records(episode_ids, state_times, buttons)
    width_layout = {9: (3, 3), 16: (4, 4), 20: (4, 5)}
    if buttons.shape[1] not in width_layout:
        raise SchemaError("Puzzle event labels support 3x3, 4x4, or 4x5 boards")
    toggle_patterns = toggle_matrix(*width_layout[buttons.shape[1]]).astype(np.uint8).T
    label_report, onset_report = _label_diagnostics(
        intervals=intervals,
        event_pair_interval_indices=event_interval_indices,
        board_changes=board_changes,
        toggle_patterns=toggle_patterns,
        moved=moved,
        valid_chunk_rows=rows,
        episode_ids=episode_ids,
        times=state_times,
    )
    movement_report = {
        "valid_chunk_count": int(len(rows)),
        "moved_chunk_count": int(np.count_nonzero(moved[rows])),
        "moved_fraction": float(np.mean(moved[rows])) if len(rows) else 0.0,
        "button_changed_chunk_count": int(np.count_nonzero(button_changed)),
        "position_q999_displacement": q999.astype(np.float32).tolist(),
        "epsilon_position": epsilon.astype(np.float32).tolist(),
        "position_units": "raw Puzzle button-joint qpos units",
    }
    metadata = {
        "object_state_schema": "puzzle-button-onehot-depth-velocity-v1",
        "object_label_version": "contact-segment-position-q999-v2",
        "object_dim": 64,
        "frame_dim": 64,
        "window": int(window),
        "input_dim": int(window * 64),
        "continuous_mean": mean.astype(np.float32).tolist(),
        "continuous_std": std.astype(np.float32).tolist(),
        "position_q999_displacement": q999.astype(np.float32).tolist(),
        "epsilon_position": epsilon.astype(np.float32).tolist(),
        "movement": movement_report,
        "onset_interval": onset_report,
        "label_diagnostics": label_report,
        "train_episode_ids": split.train_episodes.tolist(),
        "heldout_episode_ids": split.heldout_episodes.tolist(),
        "contact_gap_merge_frames": 5,
        "movement_label_definition": "position_delta_gt_0.05_q999_or_button_bit_change",
    }
    return PuzzleObjectData(
        state_raw=raw,
        frame_features=normalized,
        windows=windows,
        moved=moved,
        onset=onset,
        chunk_endpoint=endpoint,
        valid_chunk_rows=rows,
        train_rows=split.train_rows,
        heldout_rows=split.heldout_rows,
        train_episodes=split.train_episodes,
        heldout_episodes=split.heldout_episodes,
        episode_end_rows=split.episode_end_rows,
        contact_start_rows=contact_start_rows,
        contact_end_rows=contact_end_rows,
        event_source_ids=np.asarray(event_sources, dtype=np.int64),
        event_target_ids=np.asarray(event_targets, dtype=np.int64),
        event_interval_indices=np.asarray(event_interval_indices, dtype=np.int64),
        continuous_mean=mean.astype(np.float32),
        continuous_std=std.astype(np.float32),
        epsilon_object=epsilon.astype(np.float32),
        movement_report=movement_report,
        onset_report=onset_report,
        label_report=label_report,
        metadata=metadata,
    )


def frame_object_features_from_checkpoint(
    trans: TransitionSet,
    sim_state: SimStateSet,
    *,
    continuous_mean: np.ndarray,
    continuous_std: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Recreate normalized object windows using checkpoint-fitted statistics."""
    raw = puzzle_object_state_raw(trans, sim_state)
    normalized = normalize_object_state(raw, continuous_mean, continuous_std)
    split = prepare_puzzle_model_data(trans, sim_state, seed=0)
    return puzzle_state_windows(
        normalized, trans, split.terminal_state_index, window=window
    )


def chunk_constraint_costs(data: PuzzleObjectData, cost: str) -> np.ndarray:
    """Return the v0.8.2 QRL cost by transition row for event or motion training."""
    values = np.zeros(len(data.moved), dtype=np.float32)
    if cost == "events":
        values[data.onset] = 1.0
    elif cost == "motion":
        values[data.moved] = 5.0
    else:
        raise ValueError("cost must be 'events' or 'motion'")
    return values
