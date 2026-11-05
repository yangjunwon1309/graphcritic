"""Passive-coordinate bridge construction for Graph Critic v0.4."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from graphcritic.config import BridgeConfig, BridgeCostConfig, GraphConfig
from graphcritic.modes import smooth_mode_changes, state_episode_time
from graphcritic.physical_time import (
    estimate_realized_speed,
    estimate_static_articulation_tolerance,
    free_space_mask,
    realized_travel_steps,
)
from graphcritic.state_schema import estimate_passive_tolerance, get_state_schema, passive_difference
from graphcritic.tau_model import FrameGapModel
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


def _indices_in_mode(indices: NDArray, modes: NDArray, mode: int) -> NDArray[np.int64]:
    """Filter a row-index subset by mode without masking it with a full-row mask."""
    ids = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(modes, dtype=np.int64)
    if labels.ndim != 1 or (ids.size and (ids.min() < 0 or ids.max() >= labels.size)):
        raise SchemaError("mode-filter indices are outside the aligned state table")
    return ids[labels[ids] == int(mode)]


def physical_tau_lower_bounds(
    tau_learned: NDArray,
    tau_kin: NDArray,
    tau_pass: NDArray,
    *,
    settle: int = 2,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Apply tau_M=max(tau_learned,tau_kin+settle,tau_pass) elementwise."""
    learned = np.asarray(tau_learned, dtype=np.int64)
    kinematic = np.asarray(tau_kin, dtype=np.int64) + int(settle)
    passive = np.asarray(tau_pass, dtype=np.int64)
    try:
        learned, kinematic, passive = np.broadcast_arrays(learned, kinematic, passive)
    except ValueError as exc:
        raise SchemaError("tau lower-bound arrays are not broadcast-compatible") from exc
    if int(settle) < 0 or np.any(learned < 0) or np.any(kinematic < 0) or np.any(passive < 0):
        raise SchemaError("tau lower bounds and settle must be non-negative")
    # The reported violation follows the spec's diagnostic tau_learned <
    # tau_kin. The actual bridge cost still includes the additional settle
    # margin in its physical lower bound.
    return (
        np.maximum.reduce((learned, kinematic, passive)).astype(np.int64),
        learned < np.broadcast_to(np.asarray(tau_kin, dtype=np.int64), learned.shape),
        learned < passive,
    )


def _r_tau_cap(bridge_cfg: BridgeConfig, cost_cfg: BridgeCostConfig) -> int:
    """Return the R-edge cap, allowing M bridges to use a longer horizon."""
    configured = cost_cfg.tau_max_steps if bridge_cfg.r_tau_max_steps is None else bridge_cfg.r_tau_max_steps
    return min(int(configured), int(cost_cfg.tau_max_steps))


def build_passive_bridges(
    trans: TransitionSet,
    states: NDArray[np.float32],
    successes: NDArray[np.bool_],
    cfg: GraphConfig,
    sim_state: SimStateSet | None,
    *,
    use_manipulation: bool,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
    """Return src, dst, distance, anchor, tau, R/M type, modes, diagnostics."""
    bridge_cfg, cost_cfg = cfg.bridge, cfg.bridge.cost
    schema_name = bridge_cfg.state_schema
    if schema_name is None:
        schema_name = {"ogbench_scene_pos": "ogbench_scene", "robomimic_pos": "robomimic"}.get(cfg.feature.name)
    if schema_name is None:
        raise SchemaError("passive bridge strategy requires bridge.state_schema")
    schema = get_state_schema(schema_name)
    is_strict = bridge_cfg.strategy == "passive_r_strict"
    use_realized_time = bridge_cfg.strategy in {"passive_r_real", "passive_r_strict", "passive_rm_real"}
    r_tau_cap = _r_tau_cap(bridge_cfg, cost_cfg)
    if is_strict:
        rho_success = cost_cfg.bridge_success_rho
        if rho_success is None:
            raise SchemaError("passive_r_strict requires bridge.cost.bridge_success_rho")
        expected_lambda = -np.log(rho_success) / -np.log(cfg.gamma)
        if not np.isclose(cost_cfg.lambda0_steps, expected_lambda, rtol=1e-6, atol=1e-6):
            raise SchemaError(
                "passive_r_strict lambda0_steps must equal -log(rho)/-log(gamma)"
            )
    aux: dict[str, NDArray] = {}
    if sim_state is not None:
        sim_state.validate(states.shape[0])
        aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
        if sim_state.button_states is not None:
            aux["button_states"] = sim_state.button_states
    coords = schema.coordinates(states, aux)
    n_states = states.shape[0]
    episode, times = state_episode_time(trans)
    if episode.shape != (n_states,):
        raise SchemaError("state schema coordinates and state episode table are misaligned")
    modes = smooth_mode_changes(coords.mode, episode, times, cfg.min_dwell)
    # Keep q_p tied to measured contact so the held-object coordinate frame
    # switches with the physical grasp. Only the discrete graph labels use
    # dwell smoothing; delaying the q_p frame switch would look like false
    # world-space object motion while a grasp is already active.
    coords = replace(coords, mode=modes)
    contact_mask = states[:, 18] > 0.0 if schema_name == "ogbench_scene" and states.shape[1] > 18 else None
    if schema_name != "ogbench_scene" or sim_state is None:
        if is_strict or use_realized_time:
            raise SchemaError("v0.5 free-space/realized-time bridges require Scene qpos sidecar data")
        free = np.zeros(n_states, dtype=np.bool_)
    else:
        free = free_space_mask(states, sim_state.qpos, coords, distance=bridge_cfg.free_distance)
    tolerance = estimate_passive_tolerance(
        schema, coords.q_p, episode, modes, coords.q_a,
        multiplier=bridge_cfg.passive_jitter_multiplier,
        contact_mask=contact_mask,
        free_mask=free if is_strict else None,
    )
    epsilon = np.asarray(tolerance["epsilon"], dtype=np.float64)
    static_tolerance: dict[str, Any] | None = None
    if is_strict and schema_name == "ogbench_scene":
        static_tolerance = estimate_static_articulation_tolerance(
            coords.q_p, episode, times, free,
        )
        static_tolerance["epsilon"] = np.asarray(static_tolerance["epsilon"]).tolist()
        epsilon[-2:] = np.asarray(static_tolerance["epsilon"], dtype=np.float64)
    if epsilon.shape != (coords.q_p.shape[1],) or np.any(epsilon <= 0):
        raise SchemaError("StateSchema passive tolerance does not match q_p dimensions")
    realized_meta: dict[str, Any] | None = None
    realized_speed: NDArray[np.float64] | None = None
    if use_realized_time:
        realized_meta = estimate_realized_speed(
            trans, coords, free, schema,
            quantiles=(bridge_cfg.real_speed_quantile, 0.75),
        )
        key = str(float(bridge_cfg.real_speed_quantile))
        realized_speed = np.asarray(realized_meta["speed_quantiles"][key], dtype=np.float64)
    normalized = np.asarray(coords.q_p, dtype=np.float64) / epsilon[None, :]
    eligible_state = free if is_strict else np.ones(n_states, dtype=np.bool_)
    source_ids_r = np.flatnonzero((~successes) & eligible_state).astype(np.int64)
    source_ids_m = (
        np.flatnonzero(~successes).astype(np.int64) if use_manipulation else source_ids_r
    )
    refs_all_r = np.flatnonzero(eligible_state).astype(np.int64)
    refs_all_m = (
        np.arange(n_states, dtype=np.int64)
        if use_manipulation and is_strict else refs_all_r
    )
    query_count = min(
        n_states,
        max(bridge_cfg.k, cost_cfg.candidate_k) + 2 * bridge_cfg.min_time_gap + 16,
    )
    src_r: list[int] = []
    dst_r: list[int] = []
    dist_r: list[float] = []
    tau_r: list[int] = []
    src_m: list[int] = []
    dst_m: list[int] = []
    dist_m: list[float] = []
    tau_m: list[int] = []
    nn_cross = np.full(n_states, np.inf, dtype=np.float64)
    violation_kin = violation_pass = m_scored = 0

    # Bound q_p motion using only adjacent data states with unchanged mode;
    # this also handles the object-to-relative coordinate switch while grasped.
    speed = np.zeros(coords.q_p.shape[1], dtype=np.float64)
    rows = np.flatnonzero(~trans.terminal)
    if rows.size:
        rows = rows[(episode[rows] == episode[rows + 1]) & (modes[rows] == modes[rows + 1])]
        if rows.size:
            passive_delta = passive_difference(schema, coords.q_p[rows], coords.q_p[rows + 1])
            speed = np.quantile(np.abs(passive_delta), 0.999, axis=0)
    speed = np.maximum(speed, 1e-9)

    schema_input = np.concatenate((coords.r, coords.o, coords.e), axis=1).astype(np.float32)
    tau_model = None
    if use_manipulation:
        if cost_cfg.tau_model != "learned" or not cost_cfg.tau_ckpt:
            raise SchemaError("passive_rm requires a learned bridge.cost.tau_ckpt")
        tau_model = FrameGapModel.load(cost_cfg.tau_ckpt)
        if tau_model.input_dim != schema_input.shape[1]:
            raise SchemaError(
                f"tau checkpoint expects schema input width {tau_model.input_dim}; "
                f"StateSchema {schema_name} provides {schema_input.shape[1]}"
            )

    for mode in np.unique(modes[source_ids_m]):
        refs_r = _indices_in_mode(refs_all_r, modes, int(mode))
        refs_m = _indices_in_mode(refs_all_m, modes, int(mode))
        sources = _indices_in_mode(source_ids_m, modes, int(mode))
        if (refs_r.size < 2 and not use_manipulation) or refs_m.size < 2 or sources.size == 0:
            continue
        tree_r = cKDTree(normalized[refs_r]) if refs_r.size >= 2 else None
        tree_m = tree_r if np.array_equal(refs_r, refs_m) else cKDTree(normalized[refs_m])
        k_query_r = min(refs_r.size, query_count) if refs_r.size else 0
        k_query_m = min(refs_m.size, query_count)
        cross_k = min(refs_r.size, max(512, k_query_r)) if refs_r.size else 0
        for start in range(0, sources.size, 1024):
            batch = sources[start : start + 1024]
            # The retained graph candidate frontier is small, but
            # nn_passive_cross asks for a different-episode neighbor. Query a
            # wider frontier separately so repeated states from one episode
            # cannot silently hide cross-episode rows.
            if tree_r is not None:
                _, cross_local = tree_r.query(
                    normalized[batch], k=cross_k, p=np.inf, workers=-1
                )
                if cross_k == 1:
                    cross_local = cross_local[:, None]
                cross_ids = refs_r[np.asarray(cross_local, dtype=np.int64)]
                cross_mask = episode[cross_ids] != episode[batch, None]
                cross_delta = passive_difference(
                    schema, coords.q_p[batch, None, :], coords.q_p[cross_ids]
                )
                cross_distance = np.max(
                    np.abs(cross_delta) / epsilon[None, None, :], axis=2
                )
                cross_distance[~cross_mask] = np.inf
                nn_cross[batch] = np.min(cross_distance, axis=1)
            # The spec uses an infinity-norm passive radius. Query in that
            # metric as well, then retain the explicit componentwise check
            # below as a guard against backend/version differences.
            if tree_r is not None:
                _, local_r = tree_r.query(
                    normalized[batch], k=k_query_r, p=np.inf, workers=-1
                )
                if k_query_r == 1:
                    local_r = local_r[:, None]
                candidates_r = refs_r[np.asarray(local_r, dtype=np.int64)]
            else:
                candidates_r = np.empty((batch.size, 0), dtype=np.int64)
            if tree_m is tree_r:
                candidates_m = candidates_r
            else:
                _, local_m = tree_m.query(
                    normalized[batch], k=k_query_m, p=np.inf, workers=-1
                )
                if k_query_m == 1:
                    local_m = local_m[:, None]
                candidates_m = refs_m[np.asarray(local_m, dtype=np.int64)]
            for batch_index, source in enumerate(batch):
                dest = candidates_r[batch_index]
                delta = passive_difference(schema, coords.q_p[source], coords.q_p[dest])
                scaled_inf = np.max(np.abs(delta) / epsilon[None, :], axis=1)
                valid_r = (dest != source) & (
                    (episode[dest] != episode[source])
                    | (np.abs(times[dest] - times[source]) > bridge_cfg.min_time_gap)
                )
                if bridge_cfg.same_mode:
                    valid_r &= modes[dest] == modes[source]
                if is_strict:
                    valid_r &= free[dest] & free[source]
                kinematic = np.asarray(schema.kinematic_steps(coords.r[source], coords.r[dest]), dtype=np.int64)
                if use_realized_time:
                    assert realized_speed is not None
                    tau_r_candidate = np.asarray(realized_travel_steps(
                        coords.q_a[source], coords.q_a[dest], realized_speed,
                        settle=bridge_cfg.real_settle_steps,
                        periodic=getattr(schema, "periodic_action", ()),
                    ), dtype=np.int64)
                    invalid_real_lower_bound = tau_r_candidate < kinematic
                    r_eligible = valid_r & (scaled_inf <= 1.0)
                    if np.any(r_eligible & invalid_real_lower_bound):
                        raise SchemaError(
                            "tau_real fell below tau_kin for a bridge candidate; refusing optimistic graph build"
                        )
                    tau_r_candidate = np.maximum(tau_r_candidate, 1)
                    r_ids = np.flatnonzero(
                        r_eligible & (tau_r_candidate <= r_tau_cap)
                    )
                else:
                    tau_r_candidate = np.maximum(1, kinematic + bridge_cfg.kinematic_settle)
                    r_ids = np.flatnonzero(
                        valid_r & (scaled_inf <= 1.0)
                        & (tau_r_candidate <= r_tau_cap)
                    )
                if r_ids.size:
                    order = np.lexsort((scaled_inf[r_ids], tau_r_candidate[r_ids]))[: bridge_cfg.k]
                    for pos in r_ids[order]:
                        src_r.append(int(source))
                        dst_r.append(int(dest[pos]))
                        dist_r.append(float(scaled_inf[pos]))
                        tau_r.append(int(tau_r_candidate[pos]))

                if use_manipulation and tau_model is not None:
                    dest_m = candidates_m[batch_index]
                    delta_m = passive_difference(
                        schema, coords.q_p[source], coords.q_p[dest_m]
                    )
                    scaled_inf_m = np.max(np.abs(delta_m) / epsilon[None, :], axis=1)
                    valid_m = (dest_m != source) & (
                        (episode[dest_m] != episode[source])
                        | (np.abs(times[dest_m] - times[source]) > bridge_cfg.min_time_gap)
                    )
                    if bridge_cfg.same_mode:
                        valid_m &= modes[dest_m] == modes[source]
                    m_ids = np.flatnonzero(
                        valid_m & (scaled_inf_m > 1.0)
                        & (scaled_inf_m <= bridge_cfg.passive_max_multiplier)
                    )
                    if m_ids.size:
                        m_ids = m_ids[np.argsort(scaled_inf_m[m_ids], kind="stable")[: cost_cfg.candidate_k]]
                        targets = dest_m[m_ids]
                        source_rows = np.broadcast_to(
                            schema_input[source], (targets.size, schema_input.shape[1])
                        )
                        probabilities = tau_model.predict_proba_pairs(
                            source_rows, schema_input[targets], backend="numpy"
                        )
                        learned, _ = tau_model.predict_tau(probabilities, cost_cfg.tau_quantile)
                        tau_kin = np.asarray(
                            schema.kinematic_steps(coords.r[source], coords.r[targets]),
                            dtype=np.int64,
                        )
                        tau_pass = np.max(
                            np.ceil(np.abs(passive_difference(schema, coords.q_p[source], coords.q_p[targets])) / speed[None, :]),
                            axis=1,
                        ).astype(np.int64)
                        if use_realized_time:
                            assert realized_speed is not None
                            tau_real = np.asarray(realized_travel_steps(
                                coords.q_a[source], coords.q_a[targets], realized_speed,
                                settle=bridge_cfg.real_settle_steps,
                                periodic=getattr(schema, "periodic_action", ()),
                            ), dtype=np.int64)
                            if np.any(tau_real < tau_kin):
                                raise SchemaError("M bridge tau_real fell below tau_kin")
                            tau_values = np.maximum.reduce((learned, tau_real, tau_pass)).astype(np.int64)
                            kin_violations = tau_values < tau_kin
                            pass_violations = tau_values < tau_pass
                        else:
                            tau_values, kin_violations, pass_violations = physical_tau_lower_bounds(
                                learned, tau_kin, tau_pass, settle=bridge_cfg.kinematic_settle
                            )
                        violation_kin += int(np.count_nonzero(kin_violations))
                        violation_pass += int(np.count_nonzero(pass_violations))
                        m_scored += int(m_ids.size)
                        within_tau_cap = tau_values <= cost_cfg.tau_max_steps
                        m_ids = m_ids[within_tau_cap]
                        targets = targets[within_tau_cap]
                        tau_values = tau_values[within_tau_cap]
                        if not m_ids.size:
                            continue
                        order = np.lexsort((scaled_inf_m[m_ids], tau_values))[: bridge_cfg.k]
                        for pos in order:
                            src_m.append(int(source))
                            dst_m.append(int(targets[pos]))
                            dist_m.append(float(scaled_inf_m[m_ids[pos]]))
                            tau_m.append(max(1, int(tau_values[pos])))

    def arrays(src_list: list[int], dst_list: list[int], dist_list: list[float], tau_list: list[int]):
        if not src_list:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), np.empty(0, dtype=np.float32), empty.copy()
        src = np.asarray(src_list, dtype=np.int64)
        dst = np.asarray(dst_list, dtype=np.int64)
        dist = np.asarray(dist_list, dtype=np.float32)
        tau = np.asarray(tau_list, dtype=np.int64)
        order = np.lexsort((tau, dist, dst, src))
        src, dst, dist, tau = src[order], dst[order], dist[order], tau[order]
        keep = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
        return src[keep], dst[keep], dist[keep], tau[keep]

    src_r_a, dst_r_a, dist_r_a, tau_r_a = arrays(src_r, dst_r, dist_r, tau_r)
    src_m_a, dst_m_a, dist_m_a, tau_m_a = arrays(src_m, dst_m, dist_m, tau_m)
    src = np.concatenate((src_r_a, src_m_a))
    dst = np.concatenate((dst_r_a, dst_m_a))
    dist = np.concatenate((dist_r_a, dist_m_a))
    tau = np.concatenate((tau_r_a, tau_m_a))
    bridge_type = np.concatenate((np.zeros(src_r_a.size, dtype=np.int8), np.ones(src_m_a.size, dtype=np.int8)))
    nn = nn_cross[np.isfinite(nn_cross)]
    all_dist = np.concatenate((dist_r_a, dist_m_a)).astype(np.float64)
    tau_values, tau_counts = np.unique(tau, return_counts=True)
    meta: dict[str, Any] = {
        "bridge_strategy": bridge_cfg.strategy,
        "bridge_state_schema": schema_name,
        "bridge_type_counts": {"R": int(src_r_a.size), "M": int(src_m_a.size)},
        "bridge_candidates": int(src.size),
        "bridge_radius_abs": 1.0,
        "bridge_passive_epsilon": epsilon.tolist(),
        "bridge_passive_jitter_q99": np.asarray(tolerance["jitter_q99"]).tolist(),
        "bridge_passive_eligible_steps": int(tolerance["eligible_steps"]),
        "bridge_passive_multiplier": float(bridge_cfg.passive_jitter_multiplier),
        "bridge_enabled": bool(bridge_cfg.enabled),
        "bridge_free_state_fraction": float(np.mean(free)) if free.size else 0.0,
        "bridge_free_distance_m": float(bridge_cfg.free_distance) if is_strict or use_realized_time else None,
        "bridge_strict_free_exclusion_fraction": float(np.mean(~free)) if is_strict and free.size else None,
        "bridge_static_articulation_tolerance": static_tolerance,
        "bridge_realized_time": realized_meta,
        "bridge_passive_diff_epsilon_units": {
            "p50": float(np.percentile(all_dist, 50)) if all_dist.size else 0.0,
            "p95": float(np.percentile(all_dist, 95)) if all_dist.size else 0.0,
            "max": float(np.max(all_dist)) if all_dist.size else 0.0,
        },
        "nn_passive_cross": {
            "n": int(nn.size),
            "p50_epsilon_units": float(np.percentile(nn, 50)) if nn.size else None,
            "p95_epsilon_units": float(np.percentile(nn, 95)) if nn.size else None,
            "frac_le_1": float(np.mean(nn <= 1.0)) if nn.size else 0.0,
            "neighbor_query_k": 512,
            "metric": "epsilon-normalized infinity norm",
        },
        "bridge_candidate_search": {
            "query_k": int(query_count),
            "metric": "epsilon-normalized infinity norm",
            "radius_inf": 1.0,
            "retained_per_source": int(bridge_cfg.k),
            "min_time_gap": int(bridge_cfg.min_time_gap),
        },
        "bridge_tau_kinematic_violation": float(violation_kin / m_scored) if m_scored else 0.0,
        "bridge_tau_passive_violation": float(violation_pass / m_scored) if m_scored else 0.0,
        "bridge_tau_m_scored": int(m_scored),
        "bridge_tau_passive_speed_q999": speed.tolist(),
        "bridge_tau_histogram": {str(int(k)): int(v) for k, v in zip(tau_values, tau_counts, strict=True)},
        "bridge_kinematic_settle": int(bridge_cfg.kinematic_settle),
        "bridge_real_settle_steps": int(bridge_cfg.real_settle_steps),
        "bridge_real_speed_quantile": float(bridge_cfg.real_speed_quantile),
        "bridge_tau_max_steps": int(cost_cfg.tau_max_steps),
        "bridge_r_tau_max_steps": int(r_tau_cap),
        "bridge_success_rho": cost_cfg.bridge_success_rho,
        "bridge_lambda_rho_steps": float(cost_cfg.lambda0_steps) if bridge_cfg.strategy == "passive_r_strict" else 0.0,
    }
    if use_realized_time:
        assert realized_speed is not None
        if src_r_a.size:
            tau_kin_r = np.asarray(schema.kinematic_steps(coords.r[src_r_a], coords.r[dst_r_a]), dtype=np.int64)
            if np.any(tau_r_a < tau_kin_r):
                raise SchemaError("a retained R bridge has tau_real < tau_kin")
            meta["bridge_r_tau_real"] = {
                "n": int(tau_r_a.size),
                "p50": float(np.percentile(tau_r_a, 50)),
                "p95": float(np.percentile(tau_r_a, 95)),
                "kinematic_violation_count": int(np.count_nonzero(tau_r_a < tau_kin_r)),
                "tau_kin_p50": float(np.percentile(tau_kin_r, 50)),
                "tau_kin_p95": float(np.percentile(tau_kin_r, 95)),
            }
        else:
            meta["bridge_r_tau_real"] = {"n": 0, "p50": None, "p95": None, "kinematic_violation_count": 0}
    if tau_model is not None:
        meta.update({
            "bridge_m_tau_model_sha256": tau_model.checkpoint_sha256,
            "bridge_m_tau_quantile": float(cost_cfg.tau_quantile),
            "bridge_m_tau_calibrated_quantile": tau_model.calibrated_quantile(cost_cfg.tau_quantile),
            "bridge_m_tau_horizon": int(tau_model.horizon),
            "bridge_m_tau_split_seed": tau_model.metadata.get("split_seed"),
            "bridge_m_tau_far_exclusion_radius": tau_model.metadata.get("far_exclusion_radius"),
        })
    return src, dst, dist, dst.copy(), tau, bridge_type, modes, meta
