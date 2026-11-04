"""Discrete contact/task modes used to constrain graph switches."""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import SchemaError, SimStateSet, TransitionSet
from graphcritic.featurize import state_table


class ModeFn(Protocol):
    def __call__(self, feat: NDArray[np.float32], aux: Mapping[str, NDArray]) -> NDArray[np.int64]: ...


def _pack_discrete(columns: list[NDArray]) -> NDArray[np.int64]:
    if not columns:
        return np.zeros(0, dtype=np.int64)
    n = columns[0].shape[0]
    result = np.zeros(n, dtype=np.int64)
    multiplier = 1
    for column in columns:
        values = np.asarray(column, dtype=np.int64).reshape(n)
        minimum = int(values.min(initial=0))
        values = values - minimum
        base = int(values.max(initial=0)) + 1
        result += values * multiplier
        multiplier *= max(base, 1)
    return result


def _valley_threshold(values: NDArray, bins: int = 64) -> float:
    """Select the lowest histogram bin between the two strongest modes."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise SchemaError("cannot infer a gripper threshold from empty values")
    low, high = float(np.min(x)), float(np.max(x))
    if high - low < 1e-8:
        return (low + high) / 2.0
    hist, edges = np.histogram(x, bins=min(bins, max(8, int(np.sqrt(x.size)))), range=(low, high))
    peaks = np.flatnonzero((hist[1:-1] >= hist[:-2]) & (hist[1:-1] >= hist[2:])) + 1
    if peaks.size < 2:
        return float(np.median(x))
    top = peaks[np.argsort(hist[peaks])[-2:]]
    left, right = int(np.min(top)), int(np.max(top))
    if right - left < 2:
        return float((edges[left + 1] + edges[right]) / 2.0)
    valley = left + int(np.argmin(hist[left : right + 1]))
    return float((edges[valley] + edges[valley + 1]) / 2.0)


def _ogbench_grasp(feat: NDArray[np.float32], n_cubes: int) -> NDArray[np.bool_]:
    # OGBench's state observation is [6 joint qpos, 6 joint qvel, eef xyz,
    # eef yaw cos/sin, gripper opening, contact], followed by 9 values per cube.
    cube_start = 19
    if feat.shape[1] < cube_start + 9 * n_cubes:
        raise SchemaError("OGBench manipulation observation does not match the documented state layout")
    opening = feat[:, 17]
    contact = feat[:, 18] > 0
    closed = opening <= _valley_threshold(opening)
    eef = feat[:, 12:15]
    distances = []
    for cube in range(n_cubes):
        start = cube_start + 9 * cube
        distances.append(np.linalg.norm(feat[:, start : start + 3] - eef, axis=1))
    near_cube = np.min(np.stack(distances, axis=1), axis=1) <= 0.20
    return closed & (contact | near_cube)


def _scene_mode(feat: NDArray[np.float32], aux: Mapping[str, NDArray]) -> NDArray[np.int64]:
    if "qpos" not in aux or "button_states" not in aux:
        raise SchemaError("ogbench_scene mode requires sim_state qpos and button_states")
    qpos = np.asarray(aux["qpos"])
    buttons = np.asarray(aux["button_states"])
    if qpos.shape[0] != feat.shape[0] or buttons.shape[0] != feat.shape[0] or qpos.shape[1] < 23:
        raise SchemaError("OGBench Scene mode inputs are not aligned or have invalid qpos dimensions")
    n_buttons = buttons.shape[1]
    remainder = qpos.shape[1] - 14 - n_buttons - 2
    if remainder < 7 or remainder % 7:
        raise SchemaError("cannot infer Scene cube count from qpos/button-state dimensions")
    n_cubes = remainder // 7
    drawer_open = qpos[:, -2] <= -0.08
    window_open = qpos[:, -1] >= 0.10
    grasped = _ogbench_grasp(feat, n_cubes)
    columns = [buttons[:, i].astype(np.int64) for i in range(n_buttons)]
    columns.extend([drawer_open.astype(np.int64), window_open.astype(np.int64), grasped.astype(np.int64)])
    return _pack_discrete(columns)


def _cube_mode(feat: NDArray[np.float32], aux: Mapping[str, NDArray]) -> NDArray[np.int64]:
    if "qpos" not in aux:
        raise SchemaError("ogbench_cube mode requires sim_state qpos")
    qpos = np.asarray(aux["qpos"])
    n_cubes = max(1, (qpos.shape[1] - 14) // 7)
    return _ogbench_grasp(feat, n_cubes).astype(np.int64)


def _robomimic_grasp(feat: NDArray[np.float32], aux: Mapping[str, NDArray]) -> NDArray[np.int64]:
    del aux
    if feat.shape[1] < 12:
        raise SchemaError("robomimic_grasp expects eef pose, gripper qpos, and object position")
    gripper = feat[:, 7:9].mean(axis=1)
    closed = gripper <= _valley_threshold(gripper)
    near = np.linalg.norm(feat[:, 9:12] - feat[:, :3], axis=1) <= 0.05
    return (closed & near).astype(np.int64)


def _scene_v04_mode(feat: NDArray[np.float32], aux: Mapping[str, NDArray]) -> NDArray[np.int64]:
    from graphcritic.state_schema import get_state_schema

    return get_state_schema("ogbench_scene").mode(feat, aux)


def _robomimic_v04_mode(feat: NDArray[np.float32], aux: Mapping[str, NDArray]) -> NDArray[np.int64]:
    from graphcritic.state_schema import get_state_schema

    return get_state_schema("robomimic").mode(feat, aux)


MODE_FNS: dict[str, ModeFn] = {
    "none": lambda feat, aux: np.zeros(feat.shape[0], dtype=np.int64),
    "ogbench_scene": _scene_mode,
    "ogbench_cube": _cube_mode,
    "robomimic_grasp": _robomimic_grasp,
    "ogbench_scene_v04": _scene_v04_mode,
    "robomimic_v04": _robomimic_v04_mode,
}


def state_episode_time(trans: TransitionSet) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Episode/time labels for the state table, including terminal next states."""
    n = trans.n_rows
    final = np.flatnonzero(trans.terminal)
    episode = np.concatenate((trans.episode_id, trans.episode_id[final])).astype(np.int64, copy=False)
    times = np.concatenate((trans.t, trans.t[final] + 1)).astype(np.int64, copy=False)
    return episode, times


def smooth_mode_changes(
    modes: NDArray[np.int64],
    episode_id: NDArray[np.int64],
    times: NDArray[np.int64],
    min_dwell: int,
) -> NDArray[np.int64]:
    """Suppress brief mode runs by reverting them to the prior mode."""
    result = np.asarray(modes, dtype=np.int64).copy()
    for ep in np.unique(episode_id):
        ids = np.flatnonzero(episode_id == ep)
        ids = ids[np.argsort(times[ids], kind="stable")]
        if ids.size < 2:
            continue
        start = 1
        while start < ids.size:
            if result[ids[start]] == result[ids[start - 1]]:
                start += 1
                continue
            end = start + 1
            while end < ids.size and result[ids[end]] == result[ids[start]]:
                end += 1
            if end - start < min_dwell:
                result[ids[start:end]] = result[ids[start - 1]]
            start = end
    return result


def resolve_state_modes(
    trans: TransitionSet,
    mode_fn: str,
    *,
    sim_state: SimStateSet | None = None,
    min_dwell: int = 3,
) -> NDArray[np.int64]:
    """Resolve stored mode labels or derive/smooth a registered domain mode."""
    x, _, _, stored = state_table(trans)
    if mode_fn == "none":
        raw = stored
    elif mode_fn not in MODE_FNS:
        raise SchemaError(f"unknown mode_fn {mode_fn!r}; expected one of {sorted(MODE_FNS)}")
    else:
        aux: dict[str, NDArray] = {}
        if sim_state is not None:
            sim_state.validate(x.shape[0])
            aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
            if sim_state.button_states is not None:
                aux["button_states"] = sim_state.button_states
        raw = MODE_FNS[mode_fn](x, aux)
    if raw.shape != (x.shape[0],):
        raise SchemaError("mode function must return one integer label per state occurrence")
    eps, times = state_episode_time(trans)
    return smooth_mode_changes(raw, eps, times, min_dwell)
