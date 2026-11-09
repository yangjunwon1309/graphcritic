"""Windowed state features and chunk-boundary node selection for v0.8."""

from __future__ import annotations

import numpy as np

from graphcritic.types import SchemaError, TransitionSet


def puzzle_state_windows(
    frame_features: np.ndarray,
    trans: TransitionSet,
    terminal_state_index: np.ndarray,
    *,
    window: int = 5,
) -> np.ndarray:
    """Build flattened trailing windows for every state occurrence.

    Episode starts are left-padded by repeating that episode's first frame.
    Terminal next states are included in their episode's timeline. Each frame
    is already normalized independently, so the returned width is ``window *
    frame_features.shape[1]`` and no window-level statistics are fitted.
    """
    values = np.asarray(frame_features, dtype=np.float32)
    terminal_ids = np.asarray(terminal_state_index, dtype=np.int64)
    if window < 1:
        raise ValueError("window must be positive")
    if values.ndim != 2 or len(values) != trans.n_rows + int(np.count_nonzero(trans.terminal)):
        raise SchemaError("frame features must align with transition and terminal state rows")
    if terminal_ids.shape != (trans.n_rows,):
        raise SchemaError("terminal_state_index must align with transition rows")

    result = np.empty((len(values), window, values.shape[1]), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
    ends = np.r_[starts[1:], trans.n_rows]
    offsets = np.arange(window, dtype=np.int64) - (window - 1)
    for start, end in zip(starts, ends, strict=True):
        start, end = int(start), int(end)
        terminal_id = int(terminal_ids[end - 1])
        if terminal_id < trans.n_rows or terminal_id >= len(values):
            raise SchemaError(f"episode ending at row {end - 1} has an invalid terminal state")
        ids = np.concatenate((np.arange(start, end, dtype=np.int64), [terminal_id]))
        local = np.arange(len(ids), dtype=np.int64)[:, None] + offsets[None, :]
        local = np.clip(local, 0, len(ids) - 1)
        result[ids] = values[ids[local]]
    return np.ascontiguousarray(result.reshape(len(values), -1), dtype=np.float32)


def puzzle_window_candidate_nodes(
    trans: TransitionSet,
    is_goal: np.ndarray,
    *,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate node IDs, their episode IDs, and boundary-source mask.

    Window endpoints at ``t == window - 1 (mod window)`` are bridge sources
    and regular destinations. Success nodes are added as destinations even if
    their frame time is not a window boundary. Goal nodes are never bridge
    sources; Dijkstra continues to use all recorded states as graph nodes.
    """
    goals = np.asarray(is_goal, dtype=np.bool_)
    terminal_rows = np.flatnonzero(trans.terminal)
    episode_ids = np.concatenate((trans.episode_id, trans.episode_id[terminal_rows]))
    times = np.concatenate((trans.t, trans.t[terminal_rows] + 1))
    if goals.shape != episode_ids.shape:
        raise SchemaError("goal labels must align with transition and terminal state rows")
    if window < 1:
        raise ValueError("window must be positive")
    boundary = times % int(window) == int(window) - 1
    node_ids = np.flatnonzero(boundary | goals).astype(np.int64)
    source_mask = boundary[node_ids] & ~goals[node_ids]
    return node_ids, episode_ids[node_ids].astype(np.int64, copy=False), source_mask
