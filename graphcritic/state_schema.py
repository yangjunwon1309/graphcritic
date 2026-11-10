"""Physical state decompositions shared by graph building and simulator tools.

The graph core consumes the common ``r/o/e/q_p/mode`` coordinate convention.
These adapters are pure NumPy: they do not construct or step a simulator. A
runtime verifier may supply physical site readouts in ``aux`` to replace the
observation-derived end-effector pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import SchemaError


def wrap_angle(value: NDArray) -> NDArray[np.float64]:
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class StateCoordinates:
    """One aligned batch of state coordinates in physical units."""

    r: NDArray[np.float32]
    o: NDArray[np.float32]
    e: NDArray[np.float32]
    q_p: NDArray[np.float32]
    mode: NDArray[np.int64]
    q_a: NDArray[np.float32]


class StateSchema(Protocol):
    """Domain adapter with the stable interface used by Graph Critic/BOOST."""

    name: str
    action_limits: NDArray[np.float32]
    passive_names: tuple[str, ...]
    periodic_passive: tuple[int, ...]

    def robot(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray: ...
    def objects(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray: ...
    def relative(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray: ...
    def passive(self, obs: NDArray, aux: Mapping[str, NDArray] | None, mode: NDArray) -> NDArray: ...
    def mode(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray: ...
    def action_coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray: ...
    def kinematic_steps(self, r_u: NDArray, r_v: NDArray) -> NDArray: ...
    def coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> StateCoordinates: ...


def _matrix(obs: NDArray) -> tuple[NDArray[np.float64], bool]:
    value = np.asarray(obs, dtype=np.float64)
    one = value.ndim == 1
    if one:
        value = value[None, :]
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise SchemaError("state schema observations must be a finite vector or matrix")
    return value, one


def _aux_matrix(aux: Mapping[str, NDArray] | None, key: str, n: int) -> NDArray | None:
    if aux is None or key not in aux:
        return None
    value = np.asarray(aux[key])
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[0] != n or not np.all(np.isfinite(value)):
        raise SchemaError(f"state schema aux[{key!r}] must be a finite aligned matrix")
    return value


def _wrap_periodic_delta(a: NDArray, b: NDArray, indices: tuple[int, ...]) -> NDArray[np.float64]:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if indices:
        delta[..., list(indices)] = wrap_angle(delta[..., list(indices)])
    return delta


def _quat_inverse_rotate(q_xyzw: NDArray, vector: NDArray) -> NDArray[np.float64]:
    q = np.asarray(q_xyzw, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    xyz, w = q[..., :3], q[..., 3:4]
    # v' = v + 2w(q x v) + 2(q x (q x v)); conjugating q gives inverse rotation.
    xyz = -xyz
    first = 2.0 * np.cross(xyz, vector)
    return vector + w * first + np.cross(xyz, first)


@dataclass(frozen=True)
class OGBenchSceneStateSchema:
    """Scene schema using metres, radians, normalized gripper, and qpos objects."""

    name: str = "ogbench_scene"
    action_limits: NDArray[np.float32] = field(default_factory=lambda: np.asarray([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float32))
    passive_names: tuple[str, ...] = ("cube_x", "cube_y", "cube_z", "cube_yaw", "drawer", "window")
    periodic_passive: tuple[int, ...] = (3,)
    periodic_action: tuple[int, ...] = (3,)
    min_passive_tolerance: NDArray[np.float32] = field(default_factory=lambda: np.asarray([0.005, 0.005, 0.005, np.deg2rad(2), 0.005, 0.005], dtype=np.float32))

    def _physical_robot(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> tuple[NDArray, NDArray]:
        x, _ = _matrix(obs)
        qpos = _aux_matrix(aux, "qpos", x.shape[0])
        if qpos is None or qpos.shape[1] < 14:
            raise SchemaError("ogbench_scene robot coordinates require qpos with 14 robot joints")
        p = _aux_matrix(aux, "eef_position", x.shape[0])
        yaw = _aux_matrix(aux, "eef_yaw", x.shape[0])
        grip = _aux_matrix(aux, "gripper_opening", x.shape[0])
        if p is None:
            p = x[:, 12:15] / 10.0 + np.asarray([0.425, 0.0, 0.0])
        if yaw is None:
            yaw = np.arctan2(x[:, 16:17], x[:, 15:16])
        if grip is None:
            grip = np.clip(x[:, 17:18] / 3.0, 0.0, 1.0)
        if p.shape[1] != 3 or yaw.shape[1] != 1 or grip.shape[1] != 1:
            raise SchemaError("Scene eef_position/yaw/gripper readouts must have widths 3/1/1")
        return np.concatenate((p, yaw, grip), axis=1), qpos[:, :14]

    def action_coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        return np.asarray(self._physical_robot(obs, aux)[0], dtype=np.float32)

    def robot(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        q_a, joints = self._physical_robot(x, aux)
        return np.asarray(np.concatenate((joints, q_a), axis=1), dtype=np.float32)

    def _qpos(self, obs: NDArray, aux: Mapping[str, NDArray] | None) -> tuple[NDArray, int]:
        x, _ = _matrix(obs)
        qpos = _aux_matrix(aux, "qpos", x.shape[0])
        if qpos is None:
            raise SchemaError("ogbench_scene StateSchema requires aligned qpos in aux")
        from envs.ogbench_layout import SceneObservationLayout, quaternion_yaw_wxyz

        layout = SceneObservationLayout.from_dimensions(x.shape[1], qpos.shape[1])
        cubes: list[NDArray] = []
        for i in range(layout.n_cubes):
            start = layout.cube_qpos_base + 7 * i
            yaw = quaternion_yaw_wxyz(qpos[:, start + 3 : start + 7])
            # Scene cubes are symmetric under quarter-turn rotations.
            yaw = (yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
            cubes.append(np.column_stack((qpos[:, start : start + 3], yaw)))
        objects = np.concatenate((np.concatenate(cubes, axis=1), qpos[:, [layout.drawer_qpos, layout.window_qpos]]), axis=1)
        return objects, layout.n_cubes

    def objects(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        objects, _ = self._qpos(obs, aux)
        return np.asarray(objects, dtype=np.float32)

    def relative(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        q_a = self.action_coordinates(x, aux)
        objects, n_cubes = self._qpos(x, aux)
        eef = q_a[:, :3]
        eef_yaw = q_a[:, 3]
        rel = []
        for i in range(n_cubes):
            cube = objects[:, 4 * i : 4 * i + 4]
            delta = eef - cube[:, :3]
            c, s = np.cos(cube[:, 3]), np.sin(cube[:, 3])
            local = np.column_stack((c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1], delta[:, 2]))
            rel_yaw = wrap_angle(eef_yaw - cube[:, 3])
            rel.append(np.column_stack((local, rel_yaw)))
        return np.asarray(np.concatenate(rel, axis=1), dtype=np.float32)

    def mode(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray[np.int64]:
        x, _ = _matrix(obs)
        objects, n_cubes = self._qpos(x, aux)
        qpos = _aux_matrix(aux, "qpos", x.shape[0])
        buttons = _aux_matrix(aux, "button_states", x.shape[0])
        if qpos is None or buttons is None:
            raise SchemaError("ogbench_scene mode requires qpos and button_states")
        from envs.ogbench_layout import SceneObservationLayout

        layout = SceneObservationLayout.from_dimensions(x.shape[1], qpos.shape[1], buttons.shape[1])
        closed_threshold = 0.5
        gripper = np.clip(x[:, 17] / 3.0, 0.0, 1.0)
        closed = gripper <= closed_threshold
        eef = self.action_coordinates(x, aux)[:, :3]
        cube_dist = np.stack([
            np.linalg.norm(objects[:, 4 * i : 4 * i + 3] - eef, axis=1)
            for i in range(n_cubes)
        ], axis=1)
        contact = x[:, 18:19] > 0.0
        force_contact = _aux_matrix(aux, "gripper_contact", x.shape[0])
        if force_contact is not None:
            contact = force_contact > 0.0
        # Near-and-closed is not proof of a grasp. Use the pad contact signal
        # so the mode does not invent passive-coordinate changes in free space.
        grasped = closed & np.any(contact, axis=1)
        grasp_id = np.where(grasped, np.argmin(cube_dist, axis=1) + 1, 0)
        drawer_open = qpos[:, layout.drawer_qpos] <= -0.08
        window_open = qpos[:, layout.window_qpos] >= 0.10
        columns = [buttons[:, i].astype(np.int64) for i in range(buttons.shape[1])]
        result = np.zeros(x.shape[0], dtype=np.int64)
        multiplier = 1
        for values in columns:
            result += values * multiplier
            multiplier *= 4  # stable across batches, including single-state servo readouts
        result += drawer_open.astype(np.int64) * multiplier
        multiplier *= 2
        result += window_open.astype(np.int64) * multiplier
        multiplier *= 2
        result += grasp_id.astype(np.int64) * multiplier
        return result

    def passive(self, obs: NDArray, aux: Mapping[str, NDArray] | None, mode: NDArray) -> NDArray:
        objects, n_cubes = self._qpos(obs, aux)
        values = objects.copy()
        # Mode packs button bits, two articulation bits, then grasped cube id.
        buttons = _aux_matrix(aux, "button_states", values.shape[0])
        if buttons is None:
            raise SchemaError("ogbench_scene passive coordinates require button_states")
        grasp_id = np.asarray(mode, dtype=np.int64) // (4 ** buttons.shape[1] * 4)
        held_id = np.where(grasp_id > 0, grasp_id - 1, -1)
        relatives = self.relative(obs, aux)
        for row in np.flatnonzero(held_id >= 0):
            cube = int(held_id[row])
            if cube < n_cubes:
                values[row, 4 * cube : 4 * cube + 4] = relatives[row, 4 * cube : 4 * cube + 4]
        return np.asarray(values, dtype=np.float32)

    def kinematic_steps(self, r_u: NDArray, r_v: NDArray) -> NDArray:
        u, v = np.asarray(r_u, dtype=np.float64), np.asarray(r_v, dtype=np.float64)
        # robot() stores [14 robot qpos, p(3), yaw, opening].
        if u.shape[-1] != 19 or v.shape[-1] != 19:
            raise SchemaError("Scene kinematic_steps expects aligned robot vectors of width 19")
        u, v = np.broadcast_arrays(u, v)
        delta = v[..., 14:19] - u[..., 14:19]
        delta[..., 3] = wrap_angle(delta[..., 3])
        steps = np.ceil(np.maximum(np.abs(delta) / self.action_limits - 1e-6, 0.0))
        return np.max(steps, axis=-1).astype(np.int64)

    def coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> StateCoordinates:
        x, _ = _matrix(obs)
        mode = self.mode(x, aux)
        return StateCoordinates(
            r=self.robot(x, aux), o=self.objects(x, aux), e=self.relative(x, aux),
            q_p=self.passive(x, aux, mode), mode=mode, q_a=self.action_coordinates(x, aux),
        )


@dataclass(frozen=True)
class RobomimicStateSchema:
    """Low-dimensional robomimic adapter (eef pose, gripper, object position)."""

    name: str = "robomimic"
    action_limits: NDArray[np.float32] = field(default_factory=lambda: np.asarray([0.05, 0.05, 0.05, 0.3, 0.3, 0.3, 1.0], dtype=np.float32))
    passive_names: tuple[str, ...] = ("object_x", "object_y", "object_z")
    periodic_passive: tuple[int, ...] = ()
    periodic_action: tuple[int, ...] = ()

    def robot(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        if x.shape[1] < 12:
            raise SchemaError("robomimic StateSchema expects at least 12 observation values")
        quat = x[:, 3:7] / np.maximum(np.linalg.norm(x[:, 3:7], axis=1, keepdims=True), 1e-12)
        return np.asarray(np.concatenate((x[:, :3], quat, x[:, 7:9]), axis=1), dtype=np.float32)

    def objects(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        if x.shape[1] < 12:
            raise SchemaError("robomimic StateSchema expects object xyz at observation indices 9:12")
        return np.asarray(x[:, 9:12], dtype=np.float32)

    def relative(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        if x.shape[1] < 12:
            raise SchemaError("robomimic StateSchema expects at least 12 observation values")
        robot = self.robot(x, aux)
        # QC's low-dimensional Lift observation does not expose object
        # orientation, so its object frame is world-aligned. If a caller has
        # a physical object quaternion, use it to express both components in
        # that frame; otherwise preserve the available world-frame pose.
        object_quat = _aux_matrix(aux, "object_quaternion", x.shape[0])
        delta_position = x[:, :3] - x[:, 9:12]
        eef_quat = robot[:, 3:7]
        if object_quat is not None:
            if object_quat.shape[1] != 4:
                raise SchemaError("robomimic object_quaternion must have width 4")
            object_quat = object_quat / np.maximum(
                np.linalg.norm(object_quat, axis=1, keepdims=True), 1e-12
            )
            delta_position = _quat_inverse_rotate(object_quat, delta_position)
            from scipy.spatial.transform import Rotation

            # Input robomimic quaternion order is xyzw. R_o^-1 R_eef is also
            # returned as xyzw to match the observation convention.
            eef_xyzw = eef_quat[:, [0, 1, 2, 3]]
            eef_rotation = Rotation.from_quat(eef_xyzw)
            object_rotation = Rotation.from_quat(object_quat)
            eef_quat = (object_rotation.inv() * eef_rotation).as_quat()
        return np.asarray(np.concatenate((delta_position, eef_quat), axis=1), dtype=np.float32)

    def mode(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray[np.int64]:
        x, _ = _matrix(obs)
        grip = np.mean(x[:, 7:9], axis=1)
        near = np.linalg.norm(x[:, :3] - x[:, 9:12], axis=1) <= 0.05
        closed = grip <= np.median(grip)
        return (closed & near).astype(np.int64)

    def passive(self, obs: NDArray, aux: Mapping[str, NDArray] | None, mode: NDArray) -> NDArray:
        objects = self.objects(obs, aux).copy()
        held = np.asarray(mode).reshape(-1) > 0
        if np.any(held):
            objects[held] = self.relative(obs, aux)[held, :3]
        return objects

    def action_coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        robot = self.robot(x, aux)
        quat = robot[:, 3:7].astype(np.float64)
        quat = quat * np.where(quat[:, 3:4] < 0, -1.0, 1.0)
        angle = 2.0 * np.arccos(np.clip(quat[:, 3], -1.0, 1.0))
        denom = np.linalg.norm(quat[:, :3], axis=1, keepdims=True)
        axis = quat[:, :3] / np.maximum(denom, 1e-12)
        rotvec = axis * angle[:, None]
        grip = np.mean(robot[:, 7:9], axis=1, keepdims=True)
        return np.asarray(np.concatenate((robot[:, :3], rotvec, grip), axis=1), dtype=np.float32)

    def kinematic_steps(self, r_u: NDArray, r_v: NDArray) -> NDArray:
        u, v = np.asarray(r_u, dtype=np.float64), np.asarray(r_v, dtype=np.float64)
        if u.shape[-1] < 8 or v.shape[-1] != u.shape[-1]:
            raise SchemaError("robomimic kinematic_steps expects aligned robot vectors")
        u, v = np.broadcast_arrays(u, v)
        position = np.ceil(np.maximum(np.max(np.abs(v[..., :3] - u[..., :3]), axis=-1) / 0.05 - 1e-6, 0.0))
        dot = np.clip(np.abs(np.sum(u[..., 3:7] * v[..., 3:7], axis=-1)), 0.0, 1.0)
        angle = 2.0 * np.arccos(dot)
        orientation = np.ceil(np.maximum(angle / 0.3 - 1e-6, 0.0))
        grip = np.ceil(np.maximum(np.max(np.abs(v[..., 7:9] - u[..., 7:9]), axis=-1) / 1.0 - 1e-6, 0.0))
        return np.maximum.reduce((position, orientation, grip)).astype(np.int64)

    def coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> StateCoordinates:
        x, _ = _matrix(obs)
        mode = self.mode(x, aux)
        return StateCoordinates(
            r=self.robot(x, aux), o=self.objects(x, aux), e=self.relative(x, aux),
            q_p=self.passive(x, aux, mode), mode=mode, q_a=self.action_coordinates(x, aux),
        )


@dataclass(frozen=True)
class OGBenchPuzzleStateSchema:
    """Physical state adapter for OGBench Lights Out puzzle manipulation."""

    name: str = "ogbench_puzzle"
    action_limits: NDArray[np.float32] = field(default_factory=lambda: np.asarray([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float32))
    passive_names: tuple[str, ...] = ("button_configuration",)
    periodic_passive: tuple[int, ...] = ()
    periodic_action: tuple[int, ...] = (3,)
    press_height: float = 0.15
    press_xy_radius: float = 0.035

    @staticmethod
    def _button_centers(n_buttons: int) -> NDArray[np.float64]:
        shapes = {9: (3, 3), 16: (4, 4), 20: (4, 5)}
        if n_buttons not in shapes:
            raise SchemaError("OGBench puzzle supports 3x3, 4x4, and 4x5 button layouts")
        n_rows, n_cols = shapes[n_buttons]
        x = 0.425 - 0.05 * (n_rows - 1) + 0.1 * np.arange(n_rows)
        y = -0.05 * (n_cols - 1) + 0.1 * np.arange(n_cols)
        return np.asarray([[xx, yy] for xx in x for yy in y], dtype=np.float64)

    def _inputs(self, obs: NDArray, aux: Mapping[str, NDArray] | None):
        x, _ = _matrix(obs)
        buttons = _aux_matrix(aux, "button_states", x.shape[0])
        qpos = _aux_matrix(aux, "qpos", x.shape[0])
        if buttons is None or qpos is None:
            raise SchemaError("ogbench_puzzle StateSchema requires aligned qpos and button_states")
        if buttons.shape[1] not in {9, 16, 20}:
            raise SchemaError("puzzle button_states width must be 9, 16, or 20")
        expected_obs = 19 + 4 * buttons.shape[1]
        if x.shape[1] != expected_obs:
            raise SchemaError(f"puzzle observation layout mismatch: expected {expected_obs}, found {x.shape[1]}")
        if qpos.shape[1] != 14 + buttons.shape[1]:
            raise SchemaError("puzzle qpos width must contain 14 arm/gripper coordinates and one joint per button")
        if np.any((buttons != 0) & (buttons != 1)):
            raise SchemaError("puzzle button_states must be binary")
        return x, qpos, buttons.astype(np.int64)

    def action_coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, _ = _matrix(obs)
        if x.shape[1] < 19:
            raise SchemaError("puzzle observation must expose the 19-dimensional robot prefix")
        yaw = np.arctan2(x[:, 16], x[:, 15])
        grip = np.clip(x[:, 17] / 3.0, 0.0, 1.0)
        position = x[:, 12:15] / 10.0 + np.asarray([0.425, 0.0, 0.0])
        return np.asarray(np.column_stack((position, yaw, grip)), dtype=np.float32)

    def robot(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, qpos, _ = self._inputs(obs, aux)
        return np.asarray(np.concatenate((qpos[:, :14], self.action_coordinates(x, aux)), axis=1), dtype=np.float32)

    def objects(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        """Return button one-hot states, press depth, and button velocity.

        For ``B`` buttons this is a ``4B`` vector: two one-hot values per
        button followed by its physical joint position and velocity. The
        quasimetric trainer fits normalization for the continuous half from
        training episodes only. Missing qvel is accepted for legacy schema
        readouts and represented as zero; model-data preparation requires the
        aligned qvel sidecar explicitly.
        """
        x, qpos, buttons = self._inputs(obs, aux)
        del x
        n_buttons = buttons.shape[1]
        one_hot = np.stack((1 - buttons, buttons), axis=-1).reshape(len(buttons), 2 * n_buttons)
        depth = qpos[:, 14 : 14 + n_buttons]
        qvel = _aux_matrix(aux, "qvel", len(buttons))
        if qvel is None:
            velocity = np.zeros_like(depth)
        else:
            if qvel.shape[1] < 14 + n_buttons:
                raise SchemaError(
                    "puzzle qvel must contain 14 robot coordinates and one velocity per button"
                )
            velocity = qvel[:, 14 : 14 + n_buttons]
        return np.asarray(np.concatenate((one_hot, depth, velocity), axis=1), dtype=np.float32)

    def relative(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        x, qpos, buttons = self._inputs(obs, aux)
        eef = self.action_coordinates(x, aux)[:, :3].astype(np.float64)
        centers = self._button_centers(buttons.shape[1])
        distances = np.linalg.norm(eef[:, None, :2] - centers[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        top_z = 0.1254 + qpos[np.arange(qpos.shape[0]), 14 + nearest]
        selected = centers[nearest]
        return np.column_stack((eef[:, 0] - selected[:, 0], eef[:, 1] - selected[:, 1], eef[:, 2] - top_z)).astype(np.float32)

    def mode(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray[np.int64]:
        x, _, buttons = self._inputs(obs, aux)
        qa = self.action_coordinates(x, aux)
        centers = self._button_centers(buttons.shape[1])
        nearest_xy = np.min(np.linalg.norm(qa[:, None, :2] - centers[None, :, :], axis=2), axis=1)
        contact = x[:, 18] > 0.0
        pressing = contact | ((nearest_xy <= self.press_xy_radius) & (qa[:, 2] <= self.press_height))
        return pressing.astype(np.int64)

    def passive(self, obs: NDArray, aux: Mapping[str, NDArray] | None, mode: NDArray) -> NDArray:
        del mode
        _, _, buttons = self._inputs(obs, aux)
        bits = np.left_shift(
            np.ones(buttons.shape[1], dtype=np.int64),
            np.arange(buttons.shape[1], dtype=np.int64),
        )
        return (buttons @ bits).astype(np.float32)[:, None]

    def kinematic_steps(self, r_u: NDArray, r_v: NDArray) -> NDArray:
        u, v = np.broadcast_arrays(np.asarray(r_u, dtype=np.float64), np.asarray(r_v, dtype=np.float64))
        if u.shape[-1] != 19:
            raise SchemaError("puzzle kinematic_steps expects 14 arm/gripper coordinates and 5 pose coordinates")
        delta = v[..., 14:19] - u[..., 14:19]
        delta[..., 3] = wrap_angle(delta[..., 3])
        return np.max(np.ceil(np.maximum(np.abs(delta) / self.action_limits - 1e-6, 0.0)), axis=-1).astype(np.int64)

    def coordinates(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> StateCoordinates:
        x, _, _ = self._inputs(obs, aux)
        mode = self.mode(x, aux)
        return StateCoordinates(
            r=self.robot(x, aux), o=self.objects(x, aux), e=self.relative(x, aux),
            q_p=self.passive(x, aux, mode), mode=mode, q_a=self.action_coordinates(x, aux),
        )


def estimate_puzzle_press_height(
    observations: NDArray,
    button_states: NDArray,
    *,
    next_observations: NDArray | None = None,
    next_button_states: NDArray | None = None,
    default: float = 0.15,
    margin: float = 0.005,
) -> dict[str, float | int]:
    """Fit a conservative pressing-height threshold from observed toggle events."""
    obs = np.asarray(observations, dtype=np.float64)
    states = np.asarray(button_states, dtype=np.int64)
    if obs.ndim != 2 or states.ndim != 2 or obs.shape[0] != states.shape[0] or obs.shape[1] < 19:
        raise SchemaError("puzzle press-height calibration requires aligned observations and button states")
    if next_observations is not None or next_button_states is not None:
        if next_observations is None or next_button_states is None:
            raise SchemaError("both next observations and next button states are required")
        nxt_obs = np.asarray(next_observations, dtype=np.float64)
        nxt_states = np.asarray(next_button_states, dtype=np.int64)
        if nxt_obs.shape != obs.shape or nxt_states.shape != states.shape:
            raise SchemaError("puzzle current and next press-height arrays must have identical shapes")
        changed = np.any(states != nxt_states, axis=1)
        heights = np.maximum(obs[:, 14], nxt_obs[:, 14])[changed] / 10.0
    else:
        if states.shape[0] < 2:
            return {"press_height_m": float(default), "toggle_pairs": 0}
        changed = np.any(states[1:] != states[:-1], axis=1)
        heights = obs[1:, 14][changed] / 10.0
    if heights.size == 0:
        return {"press_height_m": float(default), "toggle_pairs": 0}
    threshold = float(np.clip(np.quantile(heights, 0.95) + margin, 0.13, 0.30))
    return {"press_height_m": threshold, "toggle_pairs": int(heights.size)}


STATE_SCHEMAS: dict[str, StateSchema] = {
    "ogbench_scene": OGBenchSceneStateSchema(),
    "ogbench_puzzle": OGBenchPuzzleStateSchema(),
    "robomimic": RobomimicStateSchema(),
}


def get_state_schema(name: str) -> StateSchema:
    try:
        return STATE_SCHEMAS[name]
    except KeyError as exc:
        raise SchemaError(f"unknown state schema {name!r}; available: {sorted(STATE_SCHEMAS)}") from exc


def passive_difference(schema: StateSchema, a: NDArray, b: NDArray) -> NDArray[np.float64]:
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    periodic = (
        tuple(range(3, max(3, delta.shape[-1] - 2), 4))
        if schema.name == "ogbench_scene" else schema.periodic_passive
    )
    for index in periodic:
        if schema.name == "ogbench_scene" and index % 4 == 3:
            period = np.pi / 2.0  # Scene cube yaw is invariant to quarter turns.
            delta[..., index] = (delta[..., index] + period / 2.0) % period - period / 2.0
        else:
            delta[..., index] = wrap_angle(delta[..., index])
    return delta


def estimate_passive_tolerance(
    schema: StateSchema,
    q_p: NDArray,
    episode_id: NDArray,
    modes: NDArray,
    q_a: NDArray,
    *,
    multiplier: float = 1.0,
    far_from_contact: float = 0.10,
    contact_mask: NDArray | None = None,
    free_mask: NDArray | None = None,
) -> dict[str, NDArray | dict[str, float]]:
    """Estimate eps_p = max(3*q99 physical jitter, per-coordinate floors)."""
    q = np.asarray(q_p, dtype=np.float64)
    ep = np.asarray(episode_id, dtype=np.int64)
    md = np.asarray(modes, dtype=np.int64)
    qa = np.asarray(q_a, dtype=np.float64)
    if q.ndim != 2 or ep.shape != (q.shape[0],) or md.shape != ep.shape or qa.shape[0] != q.shape[0]:
        raise SchemaError("passive tolerance inputs are not aligned")
    delta = np.zeros_like(q)
    eligible = np.zeros(q.shape[0], dtype=np.bool_)
    rows = np.flatnonzero(ep[1:] == ep[:-1])
    if rows.size:
        delta[rows + 1] = passive_difference(schema, q[rows], q[rows + 1])
        # The controller must be at least 10 cm from every passive object and
        # the mode must not encode a grasp/contact transition.
        if free_mask is not None:
            free = np.asarray(free_mask, dtype=np.bool_)
            if free.shape != (q.shape[0],):
                raise SchemaError("free_mask must have one aligned boolean per state")
        else:
            if isinstance(schema, OGBenchSceneStateSchema):
                n_buttons = 2
                grasp_id = md // (4 ** n_buttons * 4)
                is_grasp_mode = grasp_id > 0
            else:
                is_grasp_mode = md > 0
            if isinstance(schema, OGBenchSceneStateSchema) and q.shape[1] >= 6 and (q.shape[1] - 2) % 4 == 0:
                cube_values = q[:, : q.shape[1] - 2].reshape(q.shape[0], -1, 4)
                cube_xyz = cube_values[:, :, :3]
                distance = np.linalg.norm(qa[:, None, :3] - cube_xyz, axis=2).min(axis=1)
            elif q.shape[1] >= 3:
                distance = np.linalg.norm(qa[:, :3] - q[:, :3], axis=1)
            else:
                distance = np.full(q.shape[0], np.inf)
            free = (~is_grasp_mode) & (distance > far_from_contact)
        if contact_mask is not None:
            contact = np.asarray(contact_mask, dtype=np.bool_)
            if contact.shape != (q.shape[0],):
                raise SchemaError("contact_mask must have one aligned boolean per state")
            free &= ~contact
        eligible[rows + 1] = free[rows] & free[rows + 1]
    floors = getattr(schema, "min_passive_tolerance", np.full(q.shape[1], 0.005, dtype=np.float32))
    floors = np.asarray(floors, dtype=np.float64)
    if floors.size != q.shape[1]:
        floors = np.full(q.shape[1], 0.005, dtype=np.float64)
        periodic = (
            tuple(range(3, max(3, q.shape[1] - 2), 4))
            if schema.name == "ogbench_scene" else schema.periodic_passive
        )
        for index in periodic:
            floors[index] = np.deg2rad(2.0)
    if np.any(eligible):
        q99 = np.quantile(np.abs(delta[eligible]), 0.99, axis=0)
    else:
        q99 = np.zeros(q.shape[1], dtype=np.float64)
    eps = np.maximum(floors, 3.0 * q99) * float(multiplier)
    return {
        "epsilon": eps.astype(np.float32),
        "jitter_q99": q99.astype(np.float32),
        "eligible_steps": int(np.count_nonzero(eligible)),
        "multiplier": float(multiplier),
    }
