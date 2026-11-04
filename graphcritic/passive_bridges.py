"""Passive-coordinate bridge construction for Graph Critic v0.4."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from graphcritic.config import GraphConfig
from graphcritic.modes import smooth_mode_changes, state_episode_time
from graphcritic.state_schema import estimate_passive_tolerance, get_state_schema, passive_difference
from graphcritic.tau_model import FrameGapModel
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


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
    tolerance = estimate_passive_tolerance(
        schema, coords.q_p, episode, modes, coords.q_a,
        multiplier=bridge_cfg.passive_jitter_multiplier,
        contact_mask=contact_mask,
    )
    epsilon = np.asarray(tolerance["epsilon"], dtype=np.float64)
    if epsilon.shape != (coords.q_p.shape[1],) or np.any(epsilon <= 0):
        raise SchemaError("StateSchema passive tolerance does not match q_p dimensions")
    normalized = np.asarray(coords.q_p, dtype=np.float64) / epsilon[None, :]
    source_ids = np.flatnonzero(~successes).astype(np.int64)
    refs_all = np.arange(n_states, dtype=np.int64)
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

    for mode in np.unique(modes[source_ids]):
        refs = refs_all[modes == mode]
        sources = source_ids[modes[source_ids] == mode]
        if refs.size < 2 or sources.size == 0:
            continue
        tree = cKDTree(normalized[refs])
        k_query = min(refs.size, query_count)
        cross_k = min(refs.size, max(512, k_query))
        for start in range(0, sources.size, 1024):
            batch = sources[start : start + 1024]
            # The retained graph candidate frontier is small, but
            # nn_passive_cross asks for a different-episode neighbor. Query a
            # wider frontier separately so repeated states from one episode
            # cannot silently hide cross-episode rows.
            _, cross_local = tree.query(
                normalized[batch], k=cross_k, p=np.inf, workers=-1
            )
            if cross_k == 1:
                cross_local = cross_local[:, None]
            cross_ids = refs[np.asarray(cross_local, dtype=np.int64)]
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
            _, local = tree.query(normalized[batch], k=k_query, p=np.inf, workers=-1)
            if k_query == 1:
                local = local[:, None]
            candidates = refs[np.asarray(local, dtype=np.int64)]
            for source, dest in zip(batch, candidates, strict=True):
                delta = passive_difference(schema, coords.q_p[source], coords.q_p[dest])
                scaled_inf = np.max(np.abs(delta) / epsilon[None, :], axis=1)
                valid = (dest != source) & (
                    (episode[dest] != episode[source])
                    | (np.abs(times[dest] - times[source]) > bridge_cfg.min_time_gap)
                )
                if bridge_cfg.same_mode:
                    valid &= modes[dest] == modes[source]
                kinematic = np.asarray(schema.kinematic_steps(coords.r[source], coords.r[dest]), dtype=np.int64)
                r_ids = np.flatnonzero(valid & (scaled_inf <= 1.0))
                if r_ids.size:
                    order = np.lexsort((scaled_inf[r_ids], kinematic[r_ids]))[: bridge_cfg.k]
                    for pos in r_ids[order]:
                        src_r.append(int(source))
                        dst_r.append(int(dest[pos]))
                        dist_r.append(float(scaled_inf[pos]))
                        tau_r.append(max(1, int(kinematic[pos]) + bridge_cfg.kinematic_settle))

                if use_manipulation and tau_model is not None:
                    m_ids = np.flatnonzero(
                        valid & (scaled_inf > 1.0)
                        & (scaled_inf <= bridge_cfg.passive_max_multiplier)
                    )
                    if m_ids.size:
                        m_ids = m_ids[np.argsort(scaled_inf[m_ids], kind="stable")[: cost_cfg.candidate_k]]
                        targets = dest[m_ids]
                        probabilities = tau_model.predict_proba_pairs(
                            schema_input[source : source + 1], schema_input[targets], backend="numpy"
                        )
                        learned, _ = tau_model.predict_tau(probabilities, cost_cfg.tau_quantile)
                        tau_kin = kinematic[m_ids]
                        tau_pass = np.max(
                            np.ceil(np.abs(passive_difference(schema, coords.q_p[source], coords.q_p[targets])) / speed[None, :]),
                            axis=1,
                        ).astype(np.int64)
                        tau_values, kin_violations, pass_violations = physical_tau_lower_bounds(
                            learned, tau_kin, tau_pass, settle=bridge_cfg.kinematic_settle
                        )
                        violation_kin += int(np.count_nonzero(kin_violations))
                        violation_pass += int(np.count_nonzero(pass_violations))
                        m_scored += int(m_ids.size)
                        order = np.lexsort((scaled_inf[m_ids], tau_values))[: bridge_cfg.k]
                        for pos in order:
                            src_m.append(int(source))
                            dst_m.append(int(targets[pos]))
                            dist_m.append(float(scaled_inf[m_ids[pos]]))
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
        "bridge_strategy": "passive_rm" if use_manipulation else "passive_r",
        "bridge_state_schema": schema_name,
        "bridge_type_counts": {"R": int(src_r_a.size), "M": int(src_m_a.size)},
        "bridge_candidates": int(src.size),
        "bridge_radius_abs": 1.0,
        "bridge_passive_epsilon": epsilon.tolist(),
        "bridge_passive_jitter_q99": np.asarray(tolerance["jitter_q99"]).tolist(),
        "bridge_passive_eligible_steps": int(tolerance["eligible_steps"]),
        "bridge_passive_multiplier": float(bridge_cfg.passive_jitter_multiplier),
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
    }
    return src, dst, dist, dst.copy(), tau, bridge_type, modes, meta
