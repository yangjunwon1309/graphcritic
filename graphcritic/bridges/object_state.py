"""Puzzle object-state features and motion/event labels for v0.8.2.

The object judge is deliberately independent of robot proprioception.  It
consumes per-button one-hot state, press depth, and press velocity, with only
the continuous coordinates standardized from the training episode split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from graphcritic.bridges.models import PuzzleModelData, prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.featurize import state_table
from graphcritic.state_schema import OGBenchPuzzleStateSchema
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


@dataclass(frozen=True)
class PuzzleObjectData:
    """Object features, five-frame chunk labels, and episode split metadata."""

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
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    epsilon_object: np.ndarray
    movement_report: dict[str, Any]
    onset_report: dict[str, Any]
    metadata: dict[str, Any]


def puzzle_object_state_raw(
    trans: TransitionSet, sim_state: SimStateSet
) -> np.ndarray:
    """Return the 64D 4x4 Puzzle object vector in sidecar state-table order."""
    states, _, _, _ = state_table(trans)
    sim_state.validate(len(states))
    if sim_state.button_states is None or sim_state.button_states.shape[1] != 16:
        raise SchemaError("v0.8.2 object models require aligned 4x4 button states")
    # Unlike a passive legacy schema readout, object learning must never
    # silently substitute zero velocity for a missing or malformed sidecar.
    if sim_state.qvel.shape[1] < 30:
        raise SchemaError("v0.8.2 object models require 16 button joint velocities in qvel")
    schema = OGBenchPuzzleStateSchema()
    result = schema.objects(states, {
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
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
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


def prepare_puzzle_object_data(
    trans: TransitionSet,
    sim_state: SimStateSet,
    *,
    seed: int = 0,
    window: int = 5,
    train_fraction: float = 0.9,
) -> PuzzleObjectData:
    """Prepare split-safe object windows, moved/onset labels, and thresholds."""
    if window != 5:
        raise ValueError("v0.8.2 fixes the object-state input window to five frames")
    split: PuzzleModelData = prepare_puzzle_model_data(
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
    delta = normalized[endpoint[rows]].astype(np.float64) - normalized[rows].astype(np.float64)
    epsilon = np.quantile(np.abs(delta), 0.90, axis=0)
    normalized_sigma = np.std(normalized[fit_ids], axis=0, dtype=np.float64)
    epsilon = np.maximum(epsilon, 1e-4 * normalized_sigma)
    # One-hot state changes are always moved, regardless of their per-coordinate
    # quantile (which is zero for the sparse P13 event stream).
    buttons = np.asarray(sim_state.button_states, dtype=np.uint8)
    moved = np.zeros(trans.n_rows, dtype=np.bool_)
    changed = np.any(buttons[endpoint[rows]] != buttons[rows], axis=1)
    moved[rows] = np.any(np.abs(delta) > epsilon[None, :], axis=1) | changed

    # The onset sequence is sampled on non-overlapping five-frame chunk starts.
    # Comparing every overlapping start would label several copies of the same
    # physical button change as separate onsets because each moved window
    # spans five possible start frames.
    starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
    episode_start = np.empty(trans.n_rows, dtype=np.int64)
    for start, end in zip(starts, np.r_[starts[1:], trans.n_rows], strict=True):
        episode_start[int(start):int(end)] = int(start)
    onset = np.zeros(trans.n_rows, dtype=np.bool_)
    for row in rows:
        if (int(trans.t[row]) - int(trans.t[episode_start[row]])) % 5 != 0:
            continue
        previous = int(row) - 5
        has_previous = previous >= episode_start[row] and valid[previous]
        onset[row] = bool(moved[row] and (not has_previous or not moved[previous]))

    times = trans.t.astype(np.int64)
    intervals: list[int] = []
    for episode_id in np.unique(trans.episode_id):
        episode_rows = np.flatnonzero((trans.episode_id == episode_id) & onset)
        if len(episode_rows) > 1:
            event_times = times[episode_rows]
            intervals.extend(np.diff(event_times).tolist())
    if intervals:
        interval_values = np.asarray(intervals, dtype=np.float64)
        onset_report: dict[str, Any] = {
            "interval_count": int(len(interval_values)),
            "minimum_frames": float(np.min(interval_values)),
            "p1_frames": float(np.quantile(interval_values, 0.01)),
            "p5_frames": float(np.quantile(interval_values, 0.05)),
        }
    else:
        onset_report = {
            "interval_count": 0,
            "minimum_frames": None,
            "p1_frames": None,
            "p5_frames": None,
        }
    moved_fraction = float(np.mean(moved[rows])) if len(rows) else 0.0
    movement_report = {
        "valid_chunk_count": int(len(rows)),
        "moved_chunk_count": int(np.count_nonzero(moved[rows])),
        "moved_fraction": moved_fraction,
        "q90_assumption_warning": bool(moved_fraction > 0.12),
        "button_changed_chunk_count": int(np.count_nonzero(changed)),
        "epsilon_object": epsilon.astype(np.float32).tolist(),
    }
    metadata = {
        "object_state_schema": "puzzle-button-onehot-depth-velocity-v1",
        "object_dim": 64,
        "frame_dim": 64,
        "window": int(window),
        "input_dim": int(window * 64),
        "continuous_mean": mean.astype(np.float32).tolist(),
        "continuous_std": std.astype(np.float32).tolist(),
        "epsilon_object": epsilon.astype(np.float32).tolist(),
        "movement": movement_report,
        "onset_interval": onset_report,
        "train_episode_ids": split.train_episodes.tolist(),
        "heldout_episode_ids": split.heldout_episodes.tolist(),
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
        continuous_mean=mean.astype(np.float32),
        continuous_std=std.astype(np.float32),
        epsilon_object=epsilon.astype(np.float32),
        movement_report=movement_report,
        onset_report=onset_report,
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
    """Return the v0.8.2 per-row QRL chunk costs for event or motion learning."""
    if cost == "events":
        source = data.onset
    elif cost == "motion":
        source = data.moved
    else:
        raise ValueError("cost must be 'events' or 'motion'")
    values = np.zeros_like(source, dtype=np.float32)
    values[source] = 1.0 if cost == "events" else 5.0
    return values


