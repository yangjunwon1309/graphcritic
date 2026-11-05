"""Exact-discrete-state, realized-time reposition bridges for OGBench Puzzle."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from graphcritic.config import GraphConfig
from graphcritic.modes import state_episode_time
from graphcritic.physical_time import estimate_realized_speed, realized_travel_steps
from graphcritic.state_schema import OGBenchPuzzleStateSchema, estimate_puzzle_press_height
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


def build_puzzle_bridges(
    trans: TransitionSet,
    states: np.ndarray,
    successes: np.ndarray,
    cfg: GraphConfig,
    sim_state: SimStateSet | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build R bridges only between exact, equal button configurations.

    Candidate lookup is partitioned by the integer button bitfield before the
    end-effector KD tree is queried. The graph-building path reads saved
    transition/sidecar arrays only and does not create or step a simulator.
    """
    if sim_state is None or sim_state.button_states is None:
        raise SchemaError("puzzle_r requires aligned qpos and button_states sidecar arrays")
    sim_state.validate(states.shape[0])
    base_schema = OGBenchPuzzleStateSchema()
    aux = {
        "qpos": sim_state.qpos,
        "qvel": sim_state.qvel,
        "button_states": sim_state.button_states,
    }
    final_rows = np.flatnonzero(trans.terminal)
    next_state_ids = np.empty(trans.n_rows, dtype=np.int64)
    nonterminal_rows = np.flatnonzero(~trans.terminal)
    next_state_ids[nonterminal_rows] = nonterminal_rows + 1
    next_state_ids[final_rows] = trans.n_rows + np.arange(final_rows.size, dtype=np.int64)
    calibration = estimate_puzzle_press_height(
        trans.feat,
        sim_state.button_states[: trans.n_rows],
        next_observations=trans.next_feat,
        next_button_states=sim_state.button_states[next_state_ids],
    )
    schema = replace(base_schema, press_height=float(calibration["press_height_m"]))
    coordinates = schema.coordinates(states, aux)
    modes = np.asarray(coordinates.mode, dtype=np.int64)
    free = modes == 0
    episode, times = state_episode_time(trans)
    if episode.shape != (states.shape[0],) or successes.shape != episode.shape:
        raise SchemaError("puzzle bridge state rows do not align with episode, time, and success labels")
    realized = estimate_realized_speed(
        trans, coordinates, free, schema,
        quantiles=(cfg.bridge.real_speed_quantile, 0.75),
    )
    speed_key = str(float(cfg.bridge.real_speed_quantile))
    speed = np.asarray(realized["speed_quantiles"][speed_key], dtype=np.float64)
    tau_cap = int(cfg.bridge.cost.tau_max_steps)
    board_codes = coordinates.q_p[:, 0].astype(np.int64)
    ref_ids = np.flatnonzero(free).astype(np.int64)
    source_ids = np.flatnonzero(free & ~successes).astype(np.int64)
    if not source_ids.size or not ref_ids.size:
        raise SchemaError("puzzle dataset has no free non-goal R sources or destinations")

    def _bucket_rows(row_ids: np.ndarray) -> dict[int, np.ndarray]:
        codes = board_codes[row_ids]
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        unique, starts, counts = np.unique(
            sorted_codes, return_index=True, return_counts=True
        )
        return {
            int(code): row_ids[order[start : start + count]]
            for code, start, count in zip(unique, starts, counts, strict=True)
        }

    refs_by_code = _bucket_rows(ref_ids)
    sources_by_code = _bucket_rows(source_ids)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    tau_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []
    bucket_sizes: list[int] = []
    queried = retained = 0
    n_query = max(64, int(cfg.bridge.cost.candidate_k), int(cfg.bridge.k) * 4)
    for code, refs in refs_by_code.items():
        bucket_sizes.append(int(refs.size))
        sources = sources_by_code.get(code, np.empty(0, dtype=np.int64))
        if refs.size < 2 or sources.size == 0:
            continue
        tree = cKDTree(coordinates.q_a[refs, :3])
        query_k = min(refs.size, n_query)
        for start in range(0, sources.size, 512):
            batch = sources[start : start + 512]
            _, local = tree.query(coordinates.q_a[batch, :3], k=query_k, workers=-1)
            if query_k == 1:
                local = np.asarray(local)[:, None]
            candidates = refs[np.asarray(local, dtype=np.int64)]
            queried += int(candidates.size)
            allowed = candidates != batch[:, None]
            allowed &= (
                (episode[candidates] != episode[batch, None])
                | (np.abs(times[candidates] - times[batch, None]) > cfg.bridge.min_time_gap)
            )
            # Integer bitfield equality is checked explicitly, even though
            # candidates came from the matching bucket.
            allowed &= board_codes[candidates] == board_codes[batch, None]
            qa_u = np.broadcast_to(coordinates.q_a[batch, None, :], (batch.size, query_k, coordinates.q_a.shape[1]))
            qa_v = coordinates.q_a[candidates]
            tau_real = realized_travel_steps(
                qa_u, qa_v, speed,
                settle=cfg.bridge.real_settle_steps,
                periodic=schema.periodic_action,
            )
            tau_kin = schema.kinematic_steps(
                np.broadcast_to(coordinates.r[batch, None, :], (batch.size, query_k, coordinates.r.shape[1])),
                coordinates.r[candidates],
            )
            # The realized estimator is conservative, but retain the exact
            # kinematic lower bound in case a speed quantile is aggressive.
            tau = np.maximum(tau_real, tau_kin)
            allowed &= tau <= tau_cap
            distance = np.linalg.norm(qa_v[..., :3] - qa_u[..., :3], axis=-1)
            for row in range(batch.size):
                ids = np.flatnonzero(allowed[row])
                if ids.size == 0:
                    continue
                order = np.lexsort((distance[row, ids], tau[row, ids]))[: cfg.bridge.k]
                chosen = ids[order]
                src_parts.append(np.full(chosen.size, batch[row], dtype=np.int64))
                dst_parts.append(candidates[row, chosen].astype(np.int64, copy=False))
                tau_parts.append(tau[row, chosen].astype(np.int64, copy=False))
                dist_parts.append(distance[row, chosen].astype(np.float32, copy=False))
                retained += int(chosen.size)

    if src_parts:
        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
        tau = np.concatenate(tau_parts)
        dist = np.concatenate(dist_parts)
        order = np.lexsort((tau, dist, dst, src))
        src, dst, tau, dist = src[order], dst[order], tau[order], dist[order]
        keep = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
        src, dst, tau, dist = src[keep], dst[keep], tau[keep], dist[keep]
    else:
        src = dst = tau = np.empty(0, dtype=np.int64)
        dist = np.empty(0, dtype=np.float32)
    kinematic = schema.kinematic_steps(coordinates.r[src], coordinates.r[dst]) if src.size else np.empty(0, dtype=np.int64)
    if np.any(tau < kinematic):
        raise SchemaError("retained puzzle bridge has tau_real < tau_kin")
    tau_values, tau_counts = np.unique(tau, return_counts=True)
    sizes = np.asarray(bucket_sizes, dtype=np.int64)
    meta: dict[str, Any] = {
        "bridge_strategy": "puzzle_r",
        "bridge_state_schema": "ogbench_puzzle",
        "bridge_type_counts": {"R": int(src.size), "M": 0},
        "bridge_candidates": int(queried),
        "bridge_retained_candidates_pre_dedup": int(retained),
        "bridge_enabled": True,
        "bridge_free_state_fraction": float(np.mean(free)),
        "bridge_puzzle_press_calibration": calibration,
        "bridge_puzzle_tau_real": {
            "n": int(tau.size),
            "p50": float(np.percentile(tau, 50)) if tau.size else None,
            "p95": float(np.percentile(tau, 95)) if tau.size else None,
            "max": int(np.max(tau)) if tau.size else None,
            "kinematic_violation_count": int(np.count_nonzero(tau < kinematic)),
            "speed_quantile": float(cfg.bridge.real_speed_quantile),
            "speed": speed.tolist(),
        },
        "bridge_puzzle_bucket_sizes": {
            "n": int(sizes.size),
            "p10": float(np.percentile(sizes, 10)) if sizes.size else 0.0,
            "p50": float(np.percentile(sizes, 50)) if sizes.size else 0.0,
            "p90": float(np.percentile(sizes, 90)) if sizes.size else 0.0,
            "max": int(np.max(sizes)) if sizes.size else 0,
            "visited_configs": int(sizes.size),
            "candidate_query_k": int(n_query),
        },
        "bridge_tau_histogram": {str(int(k)): int(v) for k, v in zip(tau_values, tau_counts, strict=True)},
        "bridge_tau_max_steps": tau_cap,
        "bridge_success_rho": cfg.bridge.cost.bridge_success_rho,
        "bridge_lambda_rho_steps": float(cfg.bridge.cost.lambda0_steps),
        "bridge_candidate_search": {
            "metric": "exact button bitfield bucket then Euclidean EE xyz",
            "retained_per_source": int(cfg.bridge.k),
            "min_time_gap": int(cfg.bridge.min_time_gap),
        },
    }
    return src, dst, dist, dst.copy(), tau, np.zeros(src.size, dtype=np.int8), modes, meta
