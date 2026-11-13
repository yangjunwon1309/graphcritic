"""Low-dimensional OGBench cube-double observation contract.

This module is deliberately simulator-free.  It decodes the published 37D
observation and binds task goals from OGBench's CubeEnv definition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

OBS_DIM = 37
ROBOT_SLICE = slice(0, 19)
CUBE_SLICES = (slice(19, 28), slice(28, 37))
CUBE_POSITION_INDICES = ((19, 20, 21), (28, 29, 30))
CUBE_QUATERNION_INDICES = ((22, 23, 24, 25), (31, 32, 33, 34))
XYZ_CENTER = np.asarray([0.425, 0.0, 0.0], dtype=np.float64)
XYZ_SCALE = 10.0


def validate_observations(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations)
    if values.ndim != 2 or values.shape[1] != OBS_DIM:
        raise ValueError(f"cube-double observations must have shape (N, {OBS_DIM}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("cube-double observations must be finite")
    return values


def split_observation(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return robot coordinates ``r`` and simulator object poses ``o``.

    ``o`` has shape ``(N, 2, 7)`` and stores metric XYZ followed by wxyz
    quaternions.  Cube identities are retained; this function never permutes
    object indices.
    """
    values = validate_observations(observations)
    robot = np.ascontiguousarray(values[:, ROBOT_SLICE], dtype=np.float32)
    objects = np.empty((len(values), 2, 7), dtype=np.float64)
    for cube, (pos_idx, quat_idx) in enumerate(zip(CUBE_POSITION_INDICES, CUBE_QUATERNION_INDICES, strict=True)):
        objects[:, cube, :3] = values[:, pos_idx] / XYZ_SCALE + XYZ_CENTER
        quat = np.asarray(values[:, quat_idx], dtype=np.float64)
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError("cube orientation quaternion has zero norm")
        objects[:, cube, 3:] = quat / norm
    return robot, objects


def decode_cube_xyz(observations: np.ndarray) -> np.ndarray:
    """Decode cube positions from a 37D observation to metric ``(...,2,3)``."""
    _, objects = split_observation(observations)
    return objects[:, :, :3]


def cube_double_goals() -> dict[int, np.ndarray]:
    """Return the five ordered two-cube goals from OGBench 1.2.2."""
    try:
        from ogbench.manipspace.envs.cube_env import CubeEnv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("cube_double_goals requires the benchmark environment with OGBench") from exc
    definition = CubeEnv.__new__(CubeEnv)
    definition._env_type = "double"
    definition._reward_task_id = None
    definition.set_tasks()
    goals = {
        i + 1: np.asarray(task["goal_xyzs"], dtype=np.float64)
        for i, task in enumerate(definition.task_infos)
    }
    if set(goals) != {1, 2, 3, 4, 5} or any(g.shape != (2, 3) for g in goals.values()):
        raise ValueError("OGBench cube-double task definitions must provide five ordered (2,3) goal arrays")
    return goals


def task_success(observations: np.ndarray, task: int, *, tolerance_m: float = 0.04) -> np.ndarray:
    """Match CubeEnv's per-index all-cubes success condition."""
    if task not in range(1, 6):
        raise ValueError("cube-double task must be in 1..5")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive")
    _, objects = split_observation(observations)
    goals = cube_double_goals()[task]
    return np.all(np.linalg.norm(objects[:, :, :3] - goals[None, :, :], axis=2) <= tolerance_m, axis=1)


def task_success_from_goals(
    observations: np.ndarray, goals: np.ndarray, *, tolerance_m: float = 0.04
) -> np.ndarray:
    """Per-index goal success helper that accepts explicit goals for tests/builders."""
    _, objects = split_observation(observations)
    targets = np.asarray(goals, dtype=np.float64)
    if targets.shape != (2, 3) or not np.all(np.isfinite(targets)):
        raise ValueError("cube-double goals must have shape (2,3)")
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("tolerance_m must be finite and positive")
    return np.all(np.linalg.norm(objects[:, :, :3] - targets[None, :, :], axis=2) <= tolerance_m, axis=1)


def make_cube_transition_set(
    observations: np.ndarray,
    actions: np.ndarray,
    raw_terminals: np.ndarray,
    *,
    goals: np.ndarray,
):
    """Align OGBench's observation archive with transition-row consumers."""
    from graphcritic.types import TransitionSet

    obs = validate_observations(observations).astype(np.float32, copy=False)
    action = np.asarray(actions, dtype=np.float32)
    terminal_states = np.asarray(raw_terminals, dtype=bool)
    if action.shape != (len(obs), 5) or terminal_states.shape != (len(obs),):
        raise ValueError("cube-double actions and terminal flags must align with observations")
    terminal_state_rows = np.flatnonzero(terminal_states)
    if not len(terminal_state_rows) or terminal_state_rows[-1] != len(obs) - 1:
        raise ValueError("each cube-double episode must end on its terminal observation")
    current_rows = np.flatnonzero(~terminal_states)
    next_rows = current_rows + 1
    if np.any(next_rows >= len(obs)):
        raise ValueError("cube-double transition row has no next-state observation")
    episodes = np.searchsorted(terminal_state_rows, current_rows, side="right").astype(np.int64)
    starts = np.r_[0, terminal_state_rows[:-1] + 1]
    times = (current_rows - starts[episodes]).astype(np.int64)
    terminals = terminal_states[next_rows]
    current_obs, next_obs = obs[current_rows], obs[next_rows]
    n = len(current_rows)
    zeros = np.zeros(n, dtype=np.int64)
    return TransitionSet(
        feat=current_obs,
        next_feat=next_obs,
        episode_id=episodes,
        t=times,
        success=task_success_from_goals(next_obs, goals),
        success_cur=task_success_from_goals(current_obs, goals),
        terminal=terminals,
        row_idx=current_rows.astype(np.int64),
        mode=zeros,
        mode_next=zeros.copy(),
        action=action[current_rows],
    )


def archive_dataset_id(path: str | Path, env_name: str) -> str:
    """Create the same immutable OGBench archive identifier used by DQC."""
    selected = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    metadata_path = selected.with_suffix(".json")
    tag = None
    if metadata_path.is_file():
        import json
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("dataset_sha256") not in (None, sha256):
            raise ValueError("OGBench dataset metadata SHA256 does not match the archive")
        tag = metadata.get("dataset_tag")
    tag_part = f":{tag}" if tag else ""
    return f"ogbench:{env_name}{tag_part}:{sha256}"
