"""Offline estimates of free-space controls and conservative physical time."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from graphcritic.state_schema import StateCoordinates, StateSchema, wrap_angle
from graphcritic.types import SchemaError, TransitionSet


# World-space Scene task1 obstacle sites. These are static model coordinates;
# the two slide handles move along the corresponding qpos joint axes.
_SCENE_BUTTON_CENTERS = np.asarray(
    [[0.58, -0.05006164, 0.12539998], [0.58, 0.04994171, 0.12119998]],
    dtype=np.float64,
)
_SCENE_DRAWER_HANDLE_AT_ZERO = np.asarray([0.33028668, -0.24000023, 0.08])
_SCENE_DRAWER_AXIS = np.asarray([-0.00159265, -0.99999873, 0.0])
_SCENE_WINDOW_HANDLE_AT_ZERO = np.asarray([0.286, 0.225, 0.237])
_SCENE_WINDOW_AXIS = np.asarray([1.0, 0.0, 0.0])


def scene_obstacle_centers(qpos: NDArray) -> NDArray[np.float64]:
    """Return cube centers, button sites, and articulated handle centers.

    The handle constants match the official ``scene-singletask-task1-v0``
    MuJoCo model. The helper only evaluates the saved qpos against fixed model
    geometry; it does not construct or step an environment.
    """
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 25 or not np.all(np.isfinite(q)):
        raise SchemaError("Scene obstacle geometry requires finite aligned qpos with at least 25 columns")
    n_cubes = (q.shape[1] - 18) // 7
    if 18 + 7 * n_cubes != q.shape[1] or n_cubes < 1:
        raise SchemaError("Scene qpos width is inconsistent with cube, button, drawer, and window joints")
    cube = np.stack([q[:, 14 + 7 * i : 17 + 7 * i] for i in range(n_cubes)], axis=1)
    drawer_qpos = 14 + 7 * n_cubes + 2
    window_qpos = drawer_qpos + 1
    if window_qpos >= q.shape[1]:
        raise SchemaError("Scene qpos does not contain drawer and window slide coordinates")
    buttons = np.broadcast_to(_SCENE_BUTTON_CENTERS, (q.shape[0], 2, 3))
    drawer = _SCENE_DRAWER_HANDLE_AT_ZERO + q[:, drawer_qpos, None] * _SCENE_DRAWER_AXIS
    window = _SCENE_WINDOW_HANDLE_AT_ZERO + q[:, window_qpos, None] * _SCENE_WINDOW_AXIS
    return np.concatenate((cube, buttons, drawer[:, None, :], window[:, None, :]), axis=1)


def free_space_mask(
    observations: NDArray,
    qpos: NDArray,
    coordinates: StateCoordinates,
    *,
    distance: float = 0.05,
) -> NDArray[np.bool_]:
    """States with no gripper contact/grasp and at least ``distance`` clearance."""
    obs = np.asarray(observations, dtype=np.float64)
    qa = np.asarray(coordinates.q_a, dtype=np.float64)
    mode = np.asarray(coordinates.mode, dtype=np.int64)
    if obs.ndim != 2 or qa.shape != (obs.shape[0], 5) or mode.shape != (obs.shape[0],):
        raise SchemaError("free-space inputs are not aligned Scene observations and coordinates")
    if obs.shape[1] <= 18 or distance <= 0:
        raise SchemaError("free-space classification requires Scene contact observation and positive distance")
    obstacles = scene_obstacle_centers(qpos)
    if obstacles.shape[0] != obs.shape[0]:
        raise SchemaError("Scene qpos and observations are not aligned")
    nearest = np.linalg.norm(qa[:, None, :3] - obstacles, axis=-1).min(axis=1)
    # Scene mode packs buttons in base 4, drawer/window in base 2, and grasp id
    # above bit 6. Gripper contact is independently disqualifying.
    grasped = mode // 64 > 0
    contact = obs[:, 18] > 0.0
    return (~grasped) & (~contact) & (nearest >= float(distance))


def estimate_static_articulation_tolerance(
    q_p: NDArray,
    episode_id: NDArray,
    times: NDArray,
    free: NDArray,
    *,
    min_run_states: int = 10,
    min_static_steps: int = 100,
    static_delta: float = 0.001,
) -> dict[str, Any]:
    """Estimate drawer/window epsilon from long, free, nearly static runs."""
    q = np.asarray(q_p, dtype=np.float64)
    ep = np.asarray(episode_id, dtype=np.int64)
    t = np.asarray(times, dtype=np.int64)
    free = np.asarray(free, dtype=np.bool_)
    if q.ndim != 2 or q.shape[1] < 2 or ep.shape != (q.shape[0],) or t.shape != ep.shape or free.shape != ep.shape:
        raise SchemaError("static articulation tolerance inputs must be aligned with q_p")
    stats: list[dict[str, Any]] = []
    epsilon = np.full(2, 0.01, dtype=np.float64)
    for out_dim, dim in enumerate((q.shape[1] - 2, q.shape[1] - 1)):
        delta = np.zeros(q.shape[0] - 1, dtype=np.float64)
        adjacent = (ep[:-1] == ep[1:]) & (t[1:] == t[:-1] + 1) & free[:-1] & free[1:]
        delta[:] = np.abs(q[1:, dim] - q[:-1, dim])
        eligible = adjacent & (delta <= static_delta)
        retained = np.zeros_like(eligible)
        runs = 0
        same_episode_edges = ep[:-1] == ep[1:]
        run_start = eligible.copy()
        run_end = eligible.copy()
        run_start[1:] &= ~(eligible[:-1] & same_episode_edges[:-1] & same_episode_edges[1:])
        run_end[:-1] &= ~(eligible[1:] & same_episode_edges[:-1] & same_episode_edges[1:])
        starts = np.flatnonzero(run_start)
        ends = np.flatnonzero(run_end)
        for start, end in zip(starts, ends, strict=True):
            if end - start + 2 >= min_run_states:
                retained[start : end + 1] = True
                runs += 1
        values = delta[retained]
        enough = values.size >= min_static_steps
        q99 = float(np.quantile(values, 0.99)) if values.size else 0.0
        epsilon[out_dim] = max(3.0 * q99, 0.01) if enough else 0.01
        stats.append({
            "name": "drawer" if out_dim == 0 else "window",
            "static_free_steps": int(values.size),
            "qualifying_runs": int(runs),
            "static_delta_threshold_m": float(static_delta),
            "q99_abs_delta_m": q99,
            "epsilon_m": float(epsilon[out_dim]),
            "used_one_cm_fallback": not bool(enough),
            "min_run_states": int(min_run_states),
            "min_static_steps": int(min_static_steps),
        })
    return {"epsilon": epsilon.astype(np.float32), "dimensions": stats}


def _adjacent_rows(trans: TransitionSet, free: NDArray[np.bool_]) -> NDArray[np.int64]:
    rows = np.flatnonzero(~trans.terminal)
    rows = rows[trans.episode_id[rows] == trans.episode_id[rows + 1]]
    return rows[free[rows] & free[rows + 1]]


def _action_input(action: NDArray, limits: NDArray, periodic: tuple[int, ...]) -> NDArray[np.float64]:
    u = np.asarray(action, dtype=np.float64) * np.asarray(limits, dtype=np.float64)[None, :]
    # Pose yaw is represented by a bounded delta command, so no wrapping is
    # required for the input. The realized q_a difference is wrapped below.
    return u


def estimate_realized_speed(
    trans: TransitionSet,
    coordinates: StateCoordinates,
    free: NDArray,
    schema: StateSchema,
    *,
    quantiles: tuple[float, ...] = (0.95, 0.75),
) -> dict[str, Any]:
    """Estimate per-coordinate realized motion rates and action response gains."""
    if trans.action is None:
        raise SchemaError("realized-time estimation requires recorded actions")
    qa = np.asarray(coordinates.q_a, dtype=np.float64)
    free = np.asarray(free, dtype=np.bool_)
    expected_states = trans.n_rows + int(np.count_nonzero(trans.terminal))
    if qa.ndim != 2 or qa.shape[0] != free.size or free.size != expected_states:
        raise SchemaError("realized-time coordinates/free mask must align with state_table")
    rows = _adjacent_rows(trans, free)
    if rows.size < 2:
        raise SchemaError("too few free transitions to estimate realized speed")
    delta = qa[rows + 1] - qa[rows]
    for dim in getattr(schema, "periodic_action", ()):
        delta[:, dim] = wrap_angle(delta[:, dim])
    moving = (np.linalg.norm(delta[:, :3], axis=1) > 0.001) | (np.abs(delta[:, 3]) > np.deg2rad(0.5))
    speed_rows = rows[moving]
    motion = np.abs(delta[moving])
    if motion.shape[0] < 2:
        raise SchemaError("too few moving free transitions for realized speed estimation")
    rates = {
        str(float(q)): np.maximum(np.quantile(motion, q, axis=0), 1e-7).tolist()
        for q in quantiles
    }

    limits = np.asarray(schema.action_limits, dtype=np.float64)
    if trans.action.shape[1] != limits.size or qa.shape[1] != limits.size:
        raise SchemaError("recorded actions and physical q_a coordinates have incompatible widths")
    u = _action_input(trans.action[rows], limits, getattr(schema, "periodic_action", ()))
    response = _response_fit(delta, u)
    lag_rows = rows[rows > 0]
    lag_rows = lag_rows[
        (trans.episode_id[lag_rows - 1] == trans.episode_id[lag_rows])
        & free[lag_rows - 1]
    ]
    lag_u = _action_input(trans.action[lag_rows - 1], limits, getattr(schema, "periodic_action", ()))
    lag_response = _response_fit(
        qa[lag_rows + 1] - qa[lag_rows], lag_u,
        periodic=getattr(schema, "periodic_action", ()),
    )
    alpha = np.asarray(response["alpha"], dtype=np.float64)
    gain = np.clip(1.0 / np.maximum(alpha, 1e-6), 1.0, 4.0)
    return {
        "free_transition_count": int(rows.size),
        "moving_free_transition_count": int(speed_rows.size),
        "speed_quantiles": rates,
        "speed_dimensions": ["x_m", "y_m", "z_m", "yaw_rad", "gripper"],
        "speed_over_action_limit": {
            str(float(q)): (np.asarray(rates[str(float(q))]) / limits).tolist()
            for q in quantiles
        },
        "instantaneous_response": response,
        "one_step_lag_response": lag_response,
        "feedback_gain_by_dimension": gain.tolist(),
    }


def _response_fit(
    delta: NDArray,
    u: NDArray,
    *,
    periodic: tuple[int, ...] = (),
) -> dict[str, Any]:
    y = np.asarray(delta, dtype=np.float64).copy()
    x = np.asarray(u, dtype=np.float64)
    for dim in periodic:
        y[:, dim] = wrap_angle(y[:, dim])
    alpha = np.zeros(y.shape[1], dtype=np.float64)
    r2: list[float | None] = []
    n = min(y.shape[0], x.shape[0])
    y, x = y[:n], x[:n]
    for dim in range(y.shape[1]):
        denom = float(np.sum(x[:, dim] ** 2))
        alpha[dim] = float(np.sum(y[:, dim] * x[:, dim]) / denom) if denom > 1e-12 else 0.0
        residual = y[:, dim] - alpha[dim] * x[:, dim]
        total = float(np.sum((y[:, dim] - np.mean(y[:, dim])) ** 2))
        r2.append(float(1.0 - np.sum(residual ** 2) / total) if total > 1e-12 else None)
    return {"n": int(n), "alpha": alpha.tolist(), "r2": r2}


def realized_travel_steps(
    qa_u: NDArray,
    qa_v: NDArray,
    speed: NDArray,
    *,
    settle: int = 2,
    periodic: tuple[int, ...] = (),
) -> NDArray[np.int64]:
    """Compute max-coordinate travel time plus a fixed settle margin."""
    u, v = np.broadcast_arrays(np.asarray(qa_u, dtype=np.float64), np.asarray(qa_v, dtype=np.float64))
    velocity = np.asarray(speed, dtype=np.float64)
    if u.shape[-1] != velocity.size or np.any(~np.isfinite(velocity)) or np.any(velocity <= 0):
        raise SchemaError("realized travel speed must be finite, positive, and match q_a")
    if settle < 0:
        raise SchemaError("realized travel settle must be non-negative")
    delta = v - u
    for dim in periodic:
        delta[..., dim] = wrap_angle(delta[..., dim])
    steps = np.ceil(np.abs(delta) / velocity - 1e-12).astype(np.int64)
    return np.max(steps, axis=-1) + int(settle)


def realized_time_validation(
    trans: TransitionSet,
    coordinates: StateCoordinates,
    free: NDArray,
    speed: NDArray,
    schema: StateSchema,
    *,
    seed: int = 0,
    max_pairs: int = 50_000,
    settle: int = 2,
) -> dict[str, Any]:
    """Compare the estimator with same-episode free-state travel durations."""
    free = np.asarray(free, dtype=np.bool_)
    episodes = np.asarray(trans.episode_id, dtype=np.int64)
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    final_rows = np.flatnonzero(trans.terminal)
    state_episode = np.r_[episodes, episodes[final_rows]]
    state_times = np.r_[trans.t, trans.t[final_rows] + 1]
    unique_episodes = np.unique(episodes)
    free_ids = np.flatnonzero(free)
    order = np.argsort(state_episode[free_ids], kind="stable")
    sorted_ids = free_ids[order]
    sorted_episode = state_episode[sorted_ids]
    boundaries = np.r_[0, np.flatnonzero(sorted_episode[1:] != sorted_episode[:-1]) + 1, sorted_ids.size]
    by_episode = {
        int(sorted_episode[start]): sorted_ids[start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    }
    for _ in range(max_pairs * 2):
        if len(pairs) >= max_pairs:
            break
        ep = int(rng.choice(unique_episodes))
        ids = by_episode.get(ep, np.empty(0, dtype=np.int64))
        if ids.size < 2:
            continue
        a, b = np.sort(rng.choice(ids, 2, replace=False))
        if b - a < 2:
            continue
        pairs.append((int(a), int(b)))
    if not pairs:
        raise SchemaError("no same-episode free-state pairs found for realized-time validation")
    ij = np.asarray(pairs, dtype=np.int64)
    qa = np.asarray(coordinates.q_a, dtype=np.float64)
    predicted = realized_travel_steps(
        qa[ij[:, 0]], qa[ij[:, 1]], speed, settle=settle,
        periodic=getattr(schema, "periodic_action", ()),
    )
    elapsed = np.maximum(state_times[ij[:, 1]] - state_times[ij[:, 0]], 1)
    kinematic = np.asarray(
        schema.kinematic_steps(coordinates.r[ij[:, 0]], coordinates.r[ij[:, 1]]),
        dtype=np.int64,
    )
    position_distance = np.linalg.norm(qa[ij[:, 1], :3] - qa[ij[:, 0], :3], axis=1)
    bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, np.inf)]
    distance_table = []
    for lo, hi in bins:
        mask = (position_distance >= lo) & (position_distance < hi)
        if not np.any(mask):
            distance_table.append({"distance_m": [lo, None if not np.isfinite(hi) else hi], "n": 0})
            continue
        minimum_elapsed = int(np.min(elapsed[mask]))
        minimum_predicted = int(np.min(predicted[mask]))
        distance_table.append({
            "distance_m": [lo, None if not np.isfinite(hi) else hi],
            "n": int(np.count_nonzero(mask)),
            "elapsed_min": minimum_elapsed,
            "tau_real_min": minimum_predicted,
            "pairs_tau_real_below_elapsed": float(np.mean(predicted[mask] < elapsed[mask])),
        })
    return {
        "n_pairs": int(ij.shape[0]),
        "fraction_elapsed_le_tau_real": float(np.mean(elapsed <= predicted)),
        "fraction_tau_real_below_observed_elapsed": float(np.mean(predicted < elapsed)),
        "tau_kinematic_violation_count": int(np.count_nonzero(predicted < kinematic)),
        "tau_real_minus_tau_kin_p05": float(np.percentile(predicted - kinematic, 5)),
        "tau_real_minus_tau_kin_p50": float(np.percentile(predicted - kinematic, 50)),
        "distance_bins": distance_table,
        "sample_seed": int(seed),
    }

