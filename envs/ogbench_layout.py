"""Named OGBench manipulation observation layouts and position features.

The layout follows the installed OGBench ``SceneEnv.compute_observation``
implementation. It validates dimensions before interpreting observations so a
changed upstream layout cannot silently corrupt graph distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import SchemaError


@dataclass(frozen=True)
class SceneObservationLayout:
    observation_dim: int
    qpos_dim: int
    n_cubes: int
    n_buttons: int
    cube_base: int
    button_base: int
    drawer_obs: int
    window_obs: int
    cube_qpos_base: int
    drawer_qpos: int
    window_qpos: int

    @classmethod
    def from_dimensions(
        cls, observation_dim: int, qpos_dim: int, n_buttons: int = 2
    ) -> "SceneObservationLayout":
        if observation_dim < 1 or qpos_dim < 1 or n_buttons < 1:
            raise SchemaError("Scene observation and simulator dimensions must be positive")
        # Scene qpos: 14 robot coordinates, 7 per cube, one per button,
        # then drawer and window slide coordinates.
        object_dims = qpos_dim - 14 - n_buttons - 2
        if object_dims < 7 or object_dims % 7:
            raise SchemaError(
                f"Scene qpos dimension {qpos_dim} is inconsistent with {n_buttons} buttons"
            )
        n_cubes = object_dims // 7
        cube_base = 19
        button_base = cube_base + 9 * n_cubes
        expected_observation_dim = button_base + 4 * n_buttons + 4
        if observation_dim != expected_observation_dim:
            raise SchemaError(
                "Scene observation layout mismatch: expected "
                f"{expected_observation_dim} values for {n_cubes} cubes and "
                f"{n_buttons} buttons, found {observation_dim}"
            )
        drawer_qpos = 14 + 7 * n_cubes + n_buttons
        return cls(
            observation_dim=int(observation_dim),
            qpos_dim=int(qpos_dim),
            n_cubes=int(n_cubes),
            n_buttons=int(n_buttons),
            cube_base=cube_base,
            button_base=button_base,
            drawer_obs=button_base + 4 * n_buttons,
            window_obs=button_base + 4 * n_buttons + 2,
            cube_qpos_base=14,
            drawer_qpos=drawer_qpos,
            window_qpos=drawer_qpos + 1,
        )

    def dimensions(self) -> dict[str, Any]:
        """Return named raw-observation dimensions with half-open intervals."""
        result: dict[str, Any] = {
            "arm_joint_position": [0, 6],
            "arm_joint_velocity": [6, 12],
            "eef_position_scaled": [12, 15],
            "eef_yaw_cos_sin": [15, 17],
            "gripper_opening_scaled": [17, 18],
            "gripper_contact": [18, 19],
            "buttons": [],
            "drawer_position_scaled": [self.drawer_obs, self.drawer_obs + 1],
            "drawer_velocity": [self.drawer_obs + 1, self.drawer_obs + 2],
            "window_position_scaled": [self.window_obs, self.window_obs + 1],
            "window_velocity": [self.window_obs + 1, self.window_obs + 2],
            "qpos": {
                "arm_and_gripper": [0, 14],
                "cubes": [self.cube_qpos_base, self.cube_qpos_base + 7 * self.n_cubes],
                "buttons": [self.cube_qpos_base + 7 * self.n_cubes, self.drawer_qpos],
                "drawer": self.drawer_qpos,
                "window": self.window_qpos,
            },
        }
        for cube in range(self.n_cubes):
            start = self.cube_base + 9 * cube
            qstart = self.cube_qpos_base + 7 * cube
            result[f"cube_{cube}_position_scaled"] = [start, start + 3]
            result[f"cube_{cube}_quaternion_wxyz"] = [start + 3, start + 7]
            result[f"cube_{cube}_yaw_cos_sin"] = [start + 7, start + 9]
            result[f"qpos_cube_{cube}_position"] = [qstart, qstart + 3]
            result[f"qpos_cube_{cube}_quaternion_wxyz"] = [qstart + 3, qstart + 7]
        result["buttons"] = [
            [self.button_base + 4 * i, self.button_base + 4 * i + 4]
            for i in range(self.n_buttons)
        ]
        for button in range(self.n_buttons):
            start = self.button_base + 4 * button
            result[f"button_{button}_state_one_hot"] = [start, start + 2]
            result[f"button_{button}_position_scaled"] = [start + 2, start + 3]
            result[f"button_{button}_velocity"] = [start + 3, start + 4]
        return result


def quaternion_yaw_wxyz(quat: NDArray) -> NDArray[np.float64]:
    """Match OGBench's SO3.compute_yaw_radians convention."""
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or not np.all(np.isfinite(q)):
        raise SchemaError("cube quaternions must be finite (..., 4) wxyz arrays")
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scene_position_features(
    observation: NDArray, qpos: NDArray, *, n_buttons: int = 2
) -> tuple[NDArray[np.float32], tuple[tuple[str, int, int], ...], SceneObservationLayout]:
    """Build symmetric-invariant position features and group ranges.

    Group ranges are ``(name, start, end)``. Distances are standardized by the
    caller, then each group is scaled by the inverse square root of its width.
    """
    obs = np.asarray(observation, dtype=np.float32)
    pos = np.asarray(qpos, dtype=np.float64)
    if obs.ndim != 2 or pos.ndim != 2 or obs.shape[0] != pos.shape[0]:
        raise SchemaError("Scene observations and qpos must be aligned matrices")
    layout = SceneObservationLayout.from_dimensions(obs.shape[1], pos.shape[1], n_buttons)
    chunks: list[NDArray] = []
    groups: list[tuple[str, int, int]] = []
    cursor = 0

    def add_group(name: str, value: NDArray) -> None:
        nonlocal cursor
        chunk = np.asarray(value, dtype=np.float32)
        chunks.append(chunk)
        groups.append((name, cursor, cursor + chunk.shape[1]))
        cursor += chunk.shape[1]

    add_group("eef", obs[:, 12:17])
    add_group("gripper", obs[:, 17:18])
    cube_features = []
    for cube in range(layout.n_cubes):
        obs_start = layout.cube_base + 9 * cube
        qpos_start = layout.cube_qpos_base + 7 * cube
        yaw = quaternion_yaw_wxyz(pos[:, qpos_start + 3 : qpos_start + 7])
        cube_features.append(np.column_stack((
            obs[:, obs_start : obs_start + 3],
            np.cos(4.0 * yaw),
            np.sin(4.0 * yaw),
        )))
    add_group("cube", np.concatenate(cube_features, axis=1))
    add_group("drawer", obs[:, layout.drawer_obs : layout.drawer_obs + 1])
    add_group("window", obs[:, layout.window_obs : layout.window_obs + 1])
    return np.ascontiguousarray(np.concatenate(chunks, axis=1)), tuple(groups), layout


def robomimic_position_features(
    observation: NDArray,
) -> tuple[NDArray[np.float32], tuple[tuple[str, int, int], ...]]:
    """Return eef pose, gripper, and object position from QC MH observations."""
    obs = np.asarray(observation, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] < 12:
        raise SchemaError("robomimic_pos expects concatenated eef/gripper/object observations")
    quat = obs[:, 3:7].copy()
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise SchemaError("robomimic eef quaternion has zero norm")
    quat /= norm
    # Select a canonical representative of q ~ -q without assuming xyzw/wxyz.
    pivot = np.argmax(np.abs(quat), axis=1)
    sign = np.where(quat[np.arange(quat.shape[0]), pivot] < 0.0, -1.0, 1.0)
    quat *= sign[:, None]
    values = np.concatenate((obs[:, :3], quat, obs[:, 7:9], obs[:, 9:12]), axis=1)
    groups = (("eef", 0, 7), ("gripper", 7, 9), ("object", 9, 12))
    return np.ascontiguousarray(values, dtype=np.float32), groups


def verify_scene_position_layout(
    observations: NDArray,
    qpos: NDArray,
    *,
    rows: NDArray | None = None,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare qpos-derived cube/drawer/window values to OGBench observations."""
    obs = np.asarray(observations, dtype=np.float64)
    state = np.asarray(qpos, dtype=np.float64)
    if obs.ndim != 2 or state.ndim != 2 or obs.shape[0] != state.shape[0]:
        raise SchemaError("layout verification requires aligned observations and qpos")
    if rows is None:
        ids = np.arange(obs.shape[0], dtype=np.int64)
    else:
        ids = np.asarray(rows, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0 or np.any(ids < 0) or np.any(ids >= obs.shape[0]):
        raise SchemaError("layout verification rows must be a non-empty in-range index vector")
    layout = SceneObservationLayout.from_dimensions(obs.shape[1], state.shape[1])
    errors: dict[str, float] = {}
    center = np.asarray([0.425, 0.0, 0.0])
    for cube in range(layout.n_cubes):
        obs_start = layout.cube_base + 9 * cube
        qpos_start = layout.cube_qpos_base + 7 * cube
        actual = obs[ids, obs_start : obs_start + 3]
        expected = (state[ids, qpos_start : qpos_start + 3] - center) * 10.0
        errors[f"cube_{cube}_position_scaled"] = float(np.max(np.abs(actual - expected)))
    errors["drawer_position_scaled"] = float(np.max(np.abs(
        obs[ids, layout.drawer_obs] - state[ids, layout.drawer_qpos] * 18.0
    )))
    errors["window_position_scaled"] = float(np.max(np.abs(
        obs[ids, layout.window_obs] - state[ids, layout.window_qpos] * 15.0
    )))
    return {
        "n_checked": int(ids.size),
        "tolerance": float(tolerance),
        "max_abs_error_by_field": errors,
        "passed": bool(all(value <= tolerance for value in errors.values())),
        "layout": layout.dimensions(),
    }
