"""Occurrence-level directed graphs with explicit cross-trajectory bridges."""

from __future__ import annotations

from typing import Any
import importlib
import warnings
import sys

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaincinv
from scipy.spatial import cKDTree

from graphcritic.config import GraphConfig
from graphcritic.featurize import state_table
from graphcritic.modes import resolve_state_modes, state_episode_time
from graphcritic.passive_bridges import build_passive_bridges
from graphcritic.puzzle_bridges import build_puzzle_bridges
from graphcritic.segments import contract_segments
from graphcritic.tau_model import FrameGapModel
from graphcritic.types import EdgeSet, NodeSet, SchemaError, SimStateSet, TransitionSet


def _step_distance(phi: NDArray[np.float32], trans: TransitionSet, modes: NDArray[np.int64]) -> float:
    rows = np.flatnonzero(~trans.terminal)
    same_mode = modes[rows] == modes[rows + 1]
    rows = rows[same_mode]
    if rows.size == 0:
        rows = np.arange(trans.n_rows)
    distances = np.linalg.norm(phi[rows] - phi[rows + 1], axis=1).astype(np.float64)
    positive = distances[np.isfinite(distances) & (distances > 1e-8)]
    if positive.size == 0:
        # Constant-feature toys still need a well-defined threshold unit.
        return 1.0
    return float(np.median(positive))


def _state_nodes(trans: TransitionSet, states: NDArray[np.float32], success: NDArray[np.bool_]) -> NodeSet:
    n_rows = trans.n_rows
    terminal_rows = np.flatnonzero(trans.terminal)
    ref_current = np.column_stack((np.arange(n_rows, dtype=np.int64), np.zeros(n_rows, dtype=np.int64)))
    ref_terminal = np.column_stack((terminal_rows, np.ones(terminal_rows.size, dtype=np.int64)))
    ref = np.concatenate((ref_current, ref_terminal), axis=0)
    assign_cur = np.arange(n_rows, dtype=np.int64)
    assign_next = np.empty(n_rows, dtype=np.int64)
    nonterminal = ~trans.terminal
    assign_next[nonterminal] = np.flatnonzero(nonterminal) + 1
    assign_next[terminal_rows] = n_rows + np.arange(terminal_rows.size, dtype=np.int64)
    nodes = NodeSet(
        rep_feat=states,
        rep_ref=ref,
        assign_cur=assign_cur,
        assign_next=assign_next,
        is_goal=success,
    )
    nodes.validate(trans)
    return nodes


def _landing_states(
    trans: TransitionSet,
    modes: NDArray[np.int64],
    successes: NDArray[np.bool_],
    landing_steps: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return valid action-anchor state IDs and their L-step landing state IDs."""
    episode_starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
    episode_ends = np.r_[episode_starts[1:], trans.n_rows]
    final_state_by_row = np.full(trans.n_rows, -1, dtype=np.int64)
    final_rows = np.flatnonzero(trans.terminal)
    final_state_by_row[final_rows] = trans.n_rows + np.arange(final_rows.size, dtype=np.int64)
    terminal_prefix = np.r_[0, np.cumsum(trans.terminal, dtype=np.int64)]
    success_prefix = np.r_[0, np.cumsum(successes[: trans.n_rows], dtype=np.int64)]
    # A change at row j denotes a mode boundary from state j-1 to state j.
    mode_change = np.r_[False, modes[1 : trans.n_rows] != modes[: trans.n_rows - 1]]
    mode_change_prefix = np.r_[0, np.cumsum(mode_change, dtype=np.int64)]
    anchors: list[int] = []
    landings: list[int] = []
    for start, end in zip(episode_starts, episode_ends, strict=True):
        start, end = int(start), int(end)
        if end - start < landing_steps:
            continue
        candidate = np.arange(start, end - landing_steps + 1, dtype=np.int64)
        endpoint_row = candidate + landing_steps
        ends_episode = endpoint_row == end
        current_endpoint = np.minimum(endpoint_row, end - 1)

        # A terminal may occur only on the final borrowed action, whose landing
        # is the separately stored terminal-next state.
        action_stop = np.where(ends_episode, end - 1, endpoint_row)
        no_early_terminal = (
            terminal_prefix[action_stop] - terminal_prefix[candidate]
        ) == 0
        no_success = (success_prefix[endpoint_row] - success_prefix[candidate]) == 0
        no_mode_change = (
            mode_change_prefix[current_endpoint + 1]
            - mode_change_prefix[candidate + 1]
        ) == 0
        terminal_id = int(final_state_by_row[end - 1])
        if terminal_id < 0:
            continue
        terminal_mode_matches = (~ends_episode) | (modes[terminal_id] == modes[end - 1])
        valid = no_early_terminal & no_success & no_mode_change & terminal_mode_matches
        candidate = candidate[valid]
        endpoint_row = endpoint_row[valid]
        ends_episode = ends_episode[valid]
        endpoint_state = np.where(ends_episode, terminal_id, endpoint_row)
        anchors.append(candidate)
        landings.append(endpoint_state.astype(np.int64, copy=False))
    if not anchors:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    return np.concatenate(anchors), np.concatenate(landings)


def _build_knn_index(
    reference: NDArray[np.float32], backend: str
) -> tuple[Any, str]:
    chosen = backend
    if chosen == "auto":
        if reference.shape[1] > 20:
            try:
                importlib.import_module("faiss")
                chosen = "faiss"
            except ImportError:
                try:
                    jax = importlib.import_module("jax")
                    if jax.devices("gpu"):
                        chosen = "jax_gpu"
                    else:
                        chosen = "ckdtree"
                except Exception:
                    chosen = "ckdtree"
        else:
            chosen = "ckdtree"
    if chosen == "faiss":
        try:
            faiss = importlib.import_module("faiss")
        except ImportError as exc:
            raise SchemaError("knn_backend='faiss' requires faiss-cpu or faiss-gpu") from exc
        index = faiss.IndexFlatL2(reference.shape[1])
        index.add(np.ascontiguousarray(reference, dtype=np.float32))
        if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
            resources = faiss.StandardGpuResources()
            return (faiss.index_cpu_to_gpu(resources, 0, index), resources), chosen
        return index, chosen
    if chosen == "jax_gpu":
        try:
            jax = importlib.import_module("jax")
            jnp = importlib.import_module("jax.numpy")
        except ImportError as exc:
            raise SchemaError("knn_backend='jax_gpu' requires JAX with a GPU runtime") from exc
        try:
            device = jax.devices("gpu")[0]
        except (IndexError, RuntimeError) as exc:
            raise SchemaError("knn_backend='jax_gpu' requires an available JAX GPU") from exc
        reference_device = jax.device_put(np.ascontiguousarray(reference, dtype=np.float32), device)
        reference_norm = jnp.sum(reference_device * reference_device, axis=1)
        return {
            "jax": jax,
            "jnp": jnp,
            "reference": reference_device,
            "reference_cpu": np.ascontiguousarray(reference, dtype=np.float32),
            "reference_norm": reference_norm,
            "device": device,
            "compiled": {},
        }, chosen
    return cKDTree(reference), "ckdtree"


def _query_neighbors(
    index: Any,
    query: NDArray[np.float32],
    k: int,
    backend: str,
    n_reference: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if backend == "faiss":
        if isinstance(index, tuple):
            index = index[0]
        squared, ids = index.search(np.ascontiguousarray(query, dtype=np.float32), min(k, n_reference))
        return np.sqrt(np.maximum(squared, 0.0)).astype(np.float64), ids.astype(np.int64)
    if backend == "jax_gpu":
        jax, jnp = index["jax"], index["jnp"]
        count = min(k, n_reference)
        compiled = index["compiled"].get(count)
        if compiled is None:
            reference = index["reference"]
            reference_norm = index["reference_norm"]

            def query_top_k(query_device):
                query_norm = jnp.sum(query_device * query_device, axis=1, keepdims=True)
                squared = jnp.maximum(
                    query_norm + reference_norm[None, :]
                    - 2.0 * jnp.matmul(
                        query_device, reference.T, precision=jax.lax.Precision.HIGHEST
                    ),
                    0.0,
                )
                negative, ids = jax.lax.top_k(-squared, count)
                return -negative, ids

            compiled = jax.jit(query_top_k)
            index["compiled"][count] = compiled
        query_device = jax.device_put(np.ascontiguousarray(query, dtype=np.float32), index["device"])
        squared, ids = compiled(query_device)
        ids = np.asarray(ids, dtype=np.int64)
        # Recompute and sort candidate distances in float64 on CPU. This removes
        # cancellation from the norm/dot-product identity used by the GPU search.
        delta = (
            np.asarray(query, dtype=np.float64)[:, None, :]
            - index["reference_cpu"][ids].astype(np.float64)
        )
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        order = np.argsort(distances, axis=1, kind="stable")
        return (
            np.take_along_axis(distances, order, axis=1),
            np.take_along_axis(ids, order, axis=1),
        )
    distances, ids = index.query(query, k=min(k, n_reference), workers=-1)
    if distances.ndim == 1:
        distances = distances[:, None]
        ids = ids[:, None]
    return np.asarray(distances, dtype=np.float64), np.asarray(ids, dtype=np.int64)


def _bridge_candidates(
    trans: TransitionSet,
    phi: NDArray[np.float32],
    modes: NDArray[np.int64],
    successes: NDArray[np.bool_],
    cfg: GraphConfig,
    d_step: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], NDArray[np.int64], dict[str, Any]]:
    bridge_cfg = cfg.bridge
    radius = bridge_cfg.radius * d_step if bridge_cfg.radius_unit == "step" else bridge_cfg.radius
    anchor_ids, landing_ids = _landing_states(trans, modes, successes, bridge_cfg.landing_steps)
    episode, times = state_episode_time(trans)
    source_ids = np.flatnonzero(
        (~successes[: trans.n_rows]) & (trans.t % bridge_cfg.source_stride == 0)
    ).astype(np.int64)
    src_parts: list[NDArray[np.int64]] = []
    anchor_parts: list[NDArray[np.int64]] = []
    land_parts: list[NDArray[np.int64]] = []
    distance_parts: list[NDArray[np.float32]] = []
    backend_used: set[str] = set()
    query_count = bridge_cfg.k + 16
    batch = 8192

    mode_values = np.unique(modes[source_ids]) if source_ids.size else []
    for mode in mode_values:
        if bridge_cfg.same_mode:
            mode_anchors = anchor_ids[modes[anchor_ids] == mode]
            mode_sources = source_ids[modes[source_ids] == mode]
        else:
            mode_anchors = anchor_ids
            mode_sources = source_ids
        if mode_anchors.size == 0 or mode_sources.size == 0:
            continue
        print(
            f"graphcritic bridge search: mode={int(mode)} sources={mode_sources.size} anchors={mode_anchors.size}",
            file=sys.stderr,
            flush=True,
        )
        knn_index, backend_used_name = _build_knn_index(phi[mode_anchors], bridge_cfg.knn_backend)
        backend_used.add(backend_used_name)
        query_batch = 512 if backend_used_name == "jax_gpu" else batch
        for start in range(0, mode_sources.size, query_batch):
            sources = mode_sources[start : start + query_batch]
            distances, indices = _query_neighbors(
                knn_index, phi[sources], query_count, backend_used_name, mode_anchors.size
            )
            candidates = mode_anchors[np.maximum(indices, 0)]
            valid = (indices >= 0) & np.isfinite(distances) & (distances <= radius)
            same_episode = episode[candidates] == episode[sources, None]
            far_enough = np.abs(times[candidates] - times[sources, None]) > bridge_cfg.min_time_gap
            valid &= (~same_episode) | far_enough
            if bridge_cfg.same_mode:
                valid &= modes[candidates] == modes[sources, None]
            ranked = np.where(valid, distances, np.inf)
            order = np.argsort(ranked, axis=1, kind="stable")[:, : bridge_cfg.k]
            selected_dist = np.take_along_axis(ranked, order, axis=1)
            selected = np.take_along_axis(candidates, order, axis=1)
            selected_rows = np.broadcast_to(sources[:, None], selected.shape)
            keep = np.isfinite(selected_dist)
            if not np.any(keep):
                continue
            src_parts.append(selected_rows[keep].astype(np.int64, copy=False))
            anchor_parts.append(selected[keep].astype(np.int64, copy=False))
            land_parts.append(landing_ids[np.searchsorted(anchor_ids, selected[keep])])
            distance_parts.append(selected_dist[keep].astype(np.float32, copy=False))

    if src_parts:
        src = np.concatenate(src_parts)
        anchor = np.concatenate(anchor_parts)
        dst = np.concatenate(land_parts)
        dist = np.concatenate(distance_parts)
        # Deduplicate equal source/landing pairs, retaining the nearest anchor.
        order = np.lexsort((anchor, dist, dst, src))
        src, anchor, dst, dist = src[order], anchor[order], dst[order], dist[order]
        unique = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
        src, anchor, dst, dist = src[unique], anchor[unique], dst[unique], dist[unique]
    else:
        src = anchor = dst = np.empty(0, dtype=np.int64)
        dist = np.empty(0, dtype=np.float32)
    meta = {
        "bridge_radius_abs": float(radius),
        "bridge_candidates": int(src.size),
        "bridge_anchor_count": int(anchor_ids.size),
        "bridge_source_count": int(source_ids.size),
        "bridge_n_bins": int(bridge_cfg.cost.n_bins),
        "knn_backend": sorted(backend_used),
    }
    return src, dst, dist, anchor, meta


def _bridge_time_candidates(
    trans: TransitionSet,
    states: NDArray[np.float32],
    phi: NDArray[np.float32],
    modes: NDArray[np.int64],
    successes: NDArray[np.bool_],
    cfg: GraphConfig,
    d_step: float,
    dataset_id: str | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    """Create direct state-to-state bridges with estimated travel times."""
    bridge_cfg, cost_cfg = cfg.bridge, cfg.bridge.cost
    episode, times = state_episode_time(trans)
    source_ids = np.flatnonzero(
        (~successes[: trans.n_rows]) & (trans.t % bridge_cfg.source_stride == 0)
    ).astype(np.int64)
    candidate_radius = (
        cost_cfg.candidate_radius * d_step
        if bridge_cfg.radius_unit == "step" else cost_cfg.candidate_radius
    )
    radius = bridge_cfg.radius * d_step if bridge_cfg.radius_unit == "step" else bridge_cfg.radius
    if cost_cfg.tau_model == "learned" and not cost_cfg.tau_ckpt:
        raise SchemaError("bridge.cost.tau_ckpt is required for the learned time policy")
    tau_model = FrameGapModel.load(cost_cfg.tau_ckpt) if cost_cfg.tau_model == "learned" else None
    if tau_model is not None and tau_model.input_dim != states.shape[1]:
        raise SchemaError(
            f"tau checkpoint expects {tau_model.input_dim} observation values, "
            f"but the graph has {states.shape[1]}"
        )
    if tau_model is not None:
        model_meta = tau_model.metadata
        feature_meta = model_meta.get("feature")
        if isinstance(feature_meta, dict) and feature_meta.get("name") != cfg.feature.name:
            raise SchemaError("tau checkpoint feature representation does not match graph features")
        if dataset_id is not None and model_meta.get("dataset_id") not in (None, dataset_id):
            raise SchemaError("tau checkpoint dataset_id does not match graph transitions")
        if "d_step" in model_meta and not np.isclose(
            float(model_meta["d_step"]), d_step, rtol=1e-4, atol=1e-6
        ):
            raise SchemaError("tau checkpoint d_step does not match the graph feature metric")
        configured_radius = (
            cfg.bridge.radius * d_step
            if cfg.bridge.radius_unit == "step" else cfg.bridge.radius
        )
        if "bridge_radius" in model_meta and not np.isclose(
            float(model_meta["bridge_radius"]), configured_radius, rtol=1e-4, atol=1e-6
        ):
            raise SchemaError("tau checkpoint far-label radius does not match bridge.radius")
    # Same-episode states within the exclusion window can occupy the nearest
    # neighbors. Over-query enough rows to leave a useful candidate set after
    # that temporal filter, then cap the classifier input at candidate_k.
    exclusion_buffer = 2 * bridge_cfg.min_time_gap + 1
    search_k = (
        cost_cfg.candidate_k + exclusion_buffer
        if tau_model is not None
        else max(bridge_cfg.k + 16, bridge_cfg.k + exclusion_buffer)
    )
    search_radius = candidate_radius if tau_model is not None else radius
    selected_sources: list[NDArray[np.int64]] = []
    selected_destinations: list[NDArray[np.int64]] = []
    selected_distances: list[NDArray[np.float32]] = []
    selected_times: list[NDArray[np.int64]] = []
    f_h_parts: list[NDArray[np.float32]] = []
    candidate_count = same_episode_filtered = radius_filtered = far_filtered = 0
    backend_used: set[str] = set()
    mode_values = np.unique(modes[source_ids]) if source_ids.size else []

    for mode in mode_values:
        if bridge_cfg.same_mode:
            reference_ids = np.flatnonzero(modes == mode).astype(np.int64)
            mode_sources = source_ids[modes[source_ids] == mode]
        else:
            reference_ids = np.arange(states.shape[0], dtype=np.int64)
            mode_sources = source_ids
        if reference_ids.size == 0 or mode_sources.size == 0:
            continue
        index, backend = _build_knn_index(phi[reference_ids], bridge_cfg.knn_backend)
        backend_used.add(backend)
        query_batch_size = 512 if backend == "jax_gpu" else 8192
        # Keep pair-feature batches bounded for the 3-layer 512-wide classifier.
        source_batch_size = 1024 if tau_model is not None else query_batch_size
        for start in range(0, mode_sources.size, source_batch_size):
            sources = mode_sources[start : start + source_batch_size]
            distances, local_ids = _query_neighbors(
                index, phi[sources], search_k, backend, reference_ids.size
            )
            candidates = reference_ids[np.maximum(local_ids, 0)]
            valid = (local_ids >= 0) & np.isfinite(distances) & (distances <= search_radius)
            valid &= candidates != sources[:, None]
            same_episode = episode[candidates] == episode[sources, None]
            far_enough = np.abs(times[candidates] - times[sources, None]) > bridge_cfg.min_time_gap
            valid &= (~same_episode) | far_enough
            if bridge_cfg.same_mode:
                valid &= modes[candidates] == modes[sources, None]
            if tau_model is not None and valid.shape[1] > cost_cfg.candidate_k:
                nearest_order = np.argsort(
                    np.where(valid, distances, np.inf), axis=1, kind="stable"
                )[:, : cost_cfg.candidate_k]
                nearest_valid = np.zeros_like(valid)
                np.put_along_axis(nearest_valid, nearest_order, True, axis=1)
                valid &= nearest_valid
            same_episode_filtered += int(np.count_nonzero((local_ids >= 0) & ~valid))
            candidate_count += int(np.count_nonzero(valid))
            if tau_model is None:
                valid &= distances <= radius
                radius_filtered += int(np.count_nonzero((local_ids >= 0) & ~valid))
                within_radius = valid.copy()
                chosen_tau = np.full(distances.shape, cost_cfg.tau_max_steps + 1, dtype=np.int64)
                chosen_tau[valid] = np.maximum(
                    1,
                    np.ceil(
                        cost_cfg.rho * distances[valid] / max(d_step, 1e-12)
                    ).astype(np.int64),
                )
                valid &= chosen_tau <= cost_cfg.tau_max_steps
                far_filtered += int(np.count_nonzero(within_radius & ~valid))
                destinations = candidates
                final_distances = distances
                final_tau = chosen_tau
                final_f_h = np.full(chosen_tau.shape, np.nan, dtype=np.float32)
            else:
                flat_valid = np.flatnonzero(valid.reshape(-1))
                chosen_tau = np.full(valid.size, cost_cfg.tau_max_steps + 1, dtype=np.int64)
                all_f_h = np.zeros(valid.size, dtype=np.float32)
                if flat_valid.size:
                    source_grid = np.broadcast_to(sources[:, None], candidates.shape).reshape(-1)
                    candidate_flat = candidates.reshape(-1)
                    tau_values = np.empty(flat_valid.size, dtype=np.int64)
                    f_h_values = np.empty(flat_valid.size, dtype=np.float32)
                    pair_batch = 8192
                    for pair_start in range(0, flat_valid.size, pair_batch):
                        pair_end = min(pair_start + pair_batch, flat_valid.size)
                        ids = flat_valid[pair_start:pair_end]
                        probabilities = tau_model.predict_proba_pairs(
                            states[source_grid[ids]], states[candidate_flat[ids]], backend="auto"
                        )
                        tau_chunk, f_h_chunk = tau_model.predict_tau(
                            probabilities, cost_cfg.tau_quantile
                        )
                        tau_values[pair_start:pair_end] = tau_chunk
                        f_h_values[pair_start:pair_end] = f_h_chunk
                    chosen_tau[flat_valid] = tau_values
                    all_f_h[flat_valid] = f_h_values
                chosen_tau = chosen_tau.reshape(valid.shape)
                final_f_h = all_f_h.reshape(valid.shape)
                far_valid = chosen_tau <= min(cost_cfg.tau_max_steps, tau_model.horizon)
                far_filtered += int(np.count_nonzero(valid & ~far_valid))
                valid &= far_valid
                destinations = candidates
                final_distances = distances
                final_tau = chosen_tau

            if not np.any(valid):
                continue
            # Per source, retain the shortest predicted travel times. Distance
            # and state ID provide deterministic tie breaks.
            order = np.lexsort((candidates, distances, final_tau), axis=1)
            chosen = order[:, : bridge_cfg.k]
            keep = np.take_along_axis(valid, chosen, axis=1)
            src_grid = np.broadcast_to(sources[:, None], candidates.shape)
            rows = np.take_along_axis(src_grid, chosen, axis=1)[keep]
            cols = np.take_along_axis(destinations, chosen, axis=1)[keep]
            dists = np.take_along_axis(final_distances, chosen, axis=1)[keep]
            taus = np.take_along_axis(final_tau, chosen, axis=1)[keep]
            f_h_selected = np.take_along_axis(final_f_h, chosen, axis=1)[keep]
            selected_sources.append(rows.astype(np.int64, copy=False))
            selected_destinations.append(cols.astype(np.int64, copy=False))
            selected_distances.append(dists.astype(np.float32, copy=False))
            selected_times.append(taus.astype(np.int64, copy=False))
            f_h_parts.append(f_h_selected.astype(np.float32, copy=False))

    if selected_sources:
        src = np.concatenate(selected_sources)
        dst = np.concatenate(selected_destinations)
        dist = np.concatenate(selected_distances)
        tau = np.concatenate(selected_times)
        f_h = np.concatenate(f_h_parts)
        order = np.lexsort((dst, dist, tau, src))
        src, dst, dist, tau, f_h = src[order], dst[order], dist[order], tau[order], f_h[order]
        unique = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
        src, dst, dist, tau, f_h = src[unique], dst[unique], dist[unique], tau[unique], f_h[unique]
    else:
        src = dst = tau = np.empty(0, dtype=np.int64)
        dist = np.empty(0, dtype=np.float32)
        f_h = np.empty(0, dtype=np.float32)

    finite_f_h = f_h[np.isfinite(f_h)]
    tau_values, tau_counts = np.unique(tau, return_counts=True)
    goal_bridge_modes, goal_bridge_counts = np.unique(
        modes[dst[successes[dst]]], return_counts=True
    ) if dst.size and np.any(successes[dst]) else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    meta: dict[str, Any] = {
        "bridge_radius_abs": float(search_radius),
        "bridge_candidates": int(candidate_count),
        "bridge_anchor_count": int(states.shape[0]),
        "bridge_source_count": int(source_ids.size),
        "bridge_n_bins": 0,
        "knn_backend": sorted(backend_used),
        "bridge_tau_model": cost_cfg.tau_model,
        "bridge_tau_quantile": float(cost_cfg.tau_quantile),
        "bridge_tau_horizon": int(tau_model.horizon) if tau_model is not None else None,
        "bridge_tau_split_seed": (
            int(tau_model.metadata["split_seed"])
            if tau_model is not None and "split_seed" in tau_model.metadata else None
        ),
        "bridge_tau_far_exclusion_radius": (
            tau_model.metadata.get("far_exclusion_radius") if tau_model is not None else None
        ),
        "bridge_tau_calibrated_quantile": (
            tau_model.calibrated_quantile(cost_cfg.tau_quantile)
            if tau_model is not None else None
        ),
        "bridge_tau_candidate_count": int(src.size),
        "bridge_goal_edge_count": int(np.count_nonzero(successes[dst])) if dst.size else 0,
        "bridge_goal_mode_edge_counts": {
            str(int(mode)): int(count)
            for mode, count in zip(goal_bridge_modes, goal_bridge_counts, strict=True)
        },
        "bridge_tau_p50": float(np.percentile(tau, 50)) if tau.size else 0.0,
        "bridge_tau_p90": float(np.percentile(tau, 90)) if tau.size else 0.0,
        "bridge_tau_max": int(np.max(tau)) if tau.size else 0,
        "bridge_tau_histogram": {
            str(int(value)): int(count)
            for value, count in zip(tau_values, tau_counts, strict=True)
        },
        "bridge_f_h_p50": float(np.percentile(finite_f_h, 50)) if finite_f_h.size else None,
        "bridge_f_h_p10": float(np.percentile(finite_f_h, 10)) if finite_f_h.size else None,
        "bridge_f_h_p90": float(np.percentile(finite_f_h, 90)) if finite_f_h.size else None,
        "bridge_tau_dropped_far": int(far_filtered),
        "bridge_candidate_radius_filtered": int(radius_filtered),
        "bridge_candidate_time_mode_filtered": int(same_episode_filtered),
    }
    # In the v0.3 time graph, anchor records the candidate endpoint itself.
    anchor = dst.copy()
    if tau_model is not None:
        meta["tau_model_sha256"] = tau_model.checkpoint_sha256
    return src, dst, dist, anchor, tau, meta


def _calibration_counts(
    trans: TransitionSet,
    phi: NDArray[np.float32],
    modes: NDArray[np.int64],
    src: NDArray[np.int64],
    anchor: NDArray[np.int64],
    dst: NDArray[np.int64],
    dist: NDArray[np.float32],
    source_landing: NDArray[np.int64],
    cfg: GraphConfig,
    d_step: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    cost_cfg = cfg.bridge.cost
    if src.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), {
            "bridge_bin_edges": [], "bridge_bin_evidence": []
        }
    if trans.action is None:
        raise SchemaError("data_calibrated bridge costs require TransitionSet.action")
    action = trans.action.astype(np.float64)
    action_mean = np.mean(action, axis=0)
    action_std = np.maximum(np.std(action, axis=0), 1e-6)
    action_z = (action - action_mean) / action_std
    L = cfg.bridge.landing_steps
    positive_evidence = np.zeros(src.size, dtype=np.bool_)
    negative_evidence = np.zeros(src.size, dtype=np.bool_)
    chunk = 16384
    for start in range(0, src.size, chunk):
        end = min(start + chunk, src.size)
        s = src[start:end]
        a = anchor[start:end]
        # Candidate anchors and usable sources have L recorded actions.
        rows_s = s[:, None] + np.arange(L, dtype=np.int64)[None, :]
        rows_a = a[:, None] + np.arange(L, dtype=np.int64)[None, :]
        landing_source = source_landing[s]
        usable = landing_source >= 0
        action_dist = np.full(end - start, np.inf, dtype=np.float64)
        if np.any(usable):
            valid_rows = np.flatnonzero(usable)
            action_dist[valid_rows] = np.sqrt(np.mean(
                np.square(action_z[rows_s[valid_rows]] - action_z[rows_a[valid_rows]]),
                axis=(1, 2),
            ))
        landing_dist = np.full(end - start, np.inf, dtype=np.float64)
        if np.any(usable):
            landing_dist[usable] = np.linalg.norm(
                phi[landing_source[usable]] - phi[dst[start:end][usable]], axis=1
            )
        same_mode = np.zeros(end - start, dtype=np.bool_)
        same_mode[usable] = modes[landing_source[usable]] == modes[dst[start:end][usable]]
        evidence = usable & (action_dist <= cost_cfg.action_tol)
        radius = cfg.bridge.radius * d_step if cfg.bridge.radius_unit == "step" else cfg.bridge.radius
        success = (landing_dist <= radius) & same_mode
        positive_evidence[start:end] = evidence & success
        negative_evidence[start:end] = evidence & ~success

    n_bins = min(cost_cfg.n_bins, max(1, int(src.size)))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.unique(np.quantile(dist.astype(np.float64), quantiles))
    if bin_edges.size == 1:
        bin_ids = np.zeros(src.size, dtype=np.int64)
        n_bins = 1
    else:
        bin_ids = np.clip(np.searchsorted(bin_edges[1:-1], dist, side="right"), 0, bin_edges.size - 2)
        n_bins = bin_edges.size - 1
    pos = np.bincount(bin_ids, weights=positive_evidence.astype(np.int64), minlength=n_bins).astype(np.int64)
    neg = np.bincount(bin_ids, weights=negative_evidence.astype(np.int64), minlength=n_bins).astype(np.int64)
    edge_pos, edge_neg = pos[bin_ids], neg[bin_ids]
    evidence_counts = pos + neg
    for b, count in enumerate(evidence_counts):
        if count < cost_cfg.min_evidence:
            warnings.warn(
                f"bridge distance bin {b} has only {int(count)} action-matched evidence pairs",
                RuntimeWarning,
                stacklevel=2,
            )
    return edge_pos, edge_neg, {
        "bridge_n_bins": int(n_bins),
        "bridge_bin_edges": bin_edges.tolist(),
        "bridge_bin_evidence": [
            {"n_pos": int(pos[i]), "n_neg": int(neg[i]), "n_evidence": int(evidence_counts[i])}
            for i in range(n_bins)
        ],
    }


def build_state_graph(
    trans: TransitionSet,
    cfg: GraphConfig,
    phi: NDArray[np.float32],
    *,
    sim_state: SimStateSet | None = None,
    dataset_id: str | None = None,
) -> tuple[NodeSet, EdgeSet, dict[str, Any]]:
    """Build one vertex per recorded state, empirical temporal edges, and bridges."""
    trans.validate()
    states, _, success, _ = state_table(trans)
    z = np.asarray(phi, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != states.shape[0] or z.shape[1] == 0 or not np.all(np.isfinite(z)):
        raise SchemaError("phi must be a finite matrix aligned with state_table")
    modes = resolve_state_modes(
        trans, cfg.mode_fn, sim_state=sim_state, min_dwell=cfg.min_dwell
    )
    d_step = _step_distance(z, trans, modes)
    nodes = _state_nodes(trans, states, success)

    cost_meta: dict[str, Any] = {"bridge_cost_policy": cfg.bridge.cost.policy}
    bridge_type_b = np.full(0, -1, dtype=np.int8)
    manual_bridges = cfg.bridge.strategy in {
        "passive_r", "passive_r_real", "passive_r_strict", "passive_rm", "passive_rm_real", "puzzle_r"
    }
    if not cfg.bridge.enabled:
        src_b = dst_b = anchor_b = kappa_b = np.empty(0, dtype=np.int64)
        dist_b = np.empty(0, dtype=np.float32)
        bridge_type_b = np.empty(0, dtype=np.int8)
        n_pos_b = n_neg_b = np.empty(0, dtype=np.int64)
        bridge_meta = {"bridge_enabled": False, "bridge_type_counts": {"R": 0, "M": 0}}
        cost_meta["bridge_cost_policy"] = "disabled"
    elif cfg.bridge.strategy == "puzzle_r":
        (
            src_b, dst_b, dist_b, anchor_b, kappa_b, bridge_type_b, modes, bridge_meta
        ) = build_puzzle_bridges(trans, states, success, cfg, sim_state)
        n_pos_b = np.zeros(src_b.size, dtype=np.int64)
        n_neg_b = np.zeros(src_b.size, dtype=np.int64)
        cost_meta["bridge_cost_policy"] = "puzzle_realized_time_plus_rho"
    elif manual_bridges:
        (
            src_b, dst_b, dist_b, anchor_b, kappa_b, bridge_type_b, modes, bridge_meta
        ) = build_passive_bridges(
            trans, states, success, cfg, sim_state,
            use_manipulation=(
                cfg.bridge.strategy in {"passive_rm", "passive_rm_real"}
                or cfg.bridge.include_m_bridges
            ),
        )
        n_pos_b = np.zeros(src_b.size, dtype=np.int64)
        n_neg_b = np.zeros(src_b.size, dtype=np.int64)
        cost_meta["bridge_cost_policy"] = (
            "strict_realized_plus_learned_m" if cfg.bridge.include_m_bridges
            else "realized_plus_learned" if cfg.bridge.strategy == "passive_rm_real"
            else "kinematic_plus_learned" if cfg.bridge.strategy == "passive_rm"
            else "realized_time" if cfg.bridge.strategy in {"passive_r_real", "passive_r_strict"}
            else "kinematic"
        )
    elif cfg.bridge.cost.policy == "time":
        src_b, dst_b, dist_b, anchor_b, kappa_b, bridge_meta = _bridge_time_candidates(
            trans, states, z, modes, success, cfg, d_step, dataset_id
        )
        bridge_type_b = np.full(src_b.size, -1, dtype=np.int8)
        n_pos_b = np.zeros(src_b.size, dtype=np.int64)
        n_neg_b = np.zeros(src_b.size, dtype=np.int64)
    else:
        src_b, dst_b, dist_b, anchor_b, bridge_meta = _bridge_candidates(
            trans, z, modes, success, cfg, d_step
        )
        bridge_type_b = np.full(src_b.size, -1, dtype=np.int8)
        kappa_b = np.full(src_b.size, cfg.bridge.landing_steps, dtype=np.int64)
        anchor_ids, landing_ids = _landing_states(trans, modes, success, cfg.bridge.landing_steps)
        source_landing = np.full(states.shape[0], -1, dtype=np.int64)
        source_landing[anchor_ids] = landing_ids
    if not cfg.bridge.enabled:
        cost_meta["bridge_pruned_over_lambda_cap"] = 0
    elif manual_bridges:
        cost_meta["bridge_pruned_over_lambda_cap"] = 0
    elif cfg.bridge.cost.policy == "time":
        cost_meta["bridge_pruned_over_lambda_cap"] = 0
    elif cfg.bridge.cost.policy == "data_calibrated":
        n_pos_b, n_neg_b, calibration_meta = _calibration_counts(
            trans, z, modes, src_b, anchor_b, dst_b, dist_b, source_landing, cfg, d_step
        )
        cost_meta.update(calibration_meta)
        n_bins = int(calibration_meta.get("bridge_n_bins", 1))
        level = cfg.bridge.cost.delta / max(n_bins, 1)
        for bin_stats in calibration_meta.get("bridge_bin_evidence", []):
            ell_bin = float(betaincinv(
                cfg.bridge.cost.prior[0] + bin_stats["n_pos"],
                cfg.bridge.cost.prior[1] + bin_stats["n_neg"],
                level,
            ))
            bin_stats["ell_lcb"] = ell_bin
            bin_stats["lambda_steps"] = float(
                -np.log(max(ell_bin, 1e-300)) / -np.log(cfg.gamma)
            )
        ell = betaincinv(
            cfg.bridge.cost.prior[0] + n_pos_b,
            cfg.bridge.cost.prior[1] + n_neg_b,
            level,
        ) if src_b.size else np.empty(0)
        lambda_steps = -np.log(np.maximum(ell, 1e-300)) / -np.log(cfg.gamma)
        keep = lambda_steps <= cfg.bridge.cost.lambda_cap_steps
        cost_meta["bridge_pruned_over_lambda_cap"] = int(np.count_nonzero(~keep))
        src_b, dst_b, dist_b, anchor_b = src_b[keep], dst_b[keep], dist_b[keep], anchor_b[keep]
        n_pos_b, n_neg_b = n_pos_b[keep], n_neg_b[keep]
    elif cfg.bridge.cost.policy == "sim_verified":
        # The verifier replaces these placeholders after closed-loop validation.
        n_pos_b = np.zeros(src_b.size, dtype=np.int64)
        n_neg_b = np.zeros(src_b.size, dtype=np.int64)
        cost_meta["sim_verification_pending"] = True
    else:
        n_pos_b = np.zeros(src_b.size, dtype=np.int64)
        n_neg_b = np.zeros(src_b.size, dtype=np.int64)
        radius = cfg.bridge.radius * d_step if cfg.bridge.radius_unit == "step" else cfg.bridge.radius
        lambda_steps = cfg.bridge.cost.lambda0_steps + cfg.bridge.cost.lambda1_steps * np.square(dist_b / radius)
        keep = lambda_steps <= cfg.bridge.cost.lambda_cap_steps
        cost_meta["bridge_pruned_over_lambda_cap"] = int(np.count_nonzero(~keep))
        src_b, dst_b, dist_b, anchor_b = src_b[keep], dst_b[keep], dist_b[keep], anchor_b[keep]
        n_pos_b, n_neg_b = n_pos_b[keep], n_neg_b[keep]

    rows = np.arange(trans.n_rows, dtype=np.int64)
    temporal_src = nodes.assign_cur
    temporal_dst = nodes.assign_next
    src = np.concatenate((temporal_src, src_b))
    dst = np.concatenate((temporal_dst, dst_b))
    edges = EdgeSet(
        src=src,
        dst=dst,
        kappa=np.concatenate((np.ones(trans.n_rows, dtype=np.int64), kappa_b)),
        n_pos=np.concatenate((np.ones(trans.n_rows, dtype=np.int64), n_pos_b)),
        n_neg=np.concatenate((np.zeros(trans.n_rows, dtype=np.int64), n_neg_b)),
        kind=np.concatenate((np.zeros(trans.n_rows, dtype=np.int8), np.ones(src_b.size, dtype=np.int8))),
        dist=np.concatenate((np.zeros(trans.n_rows, dtype=np.float32), dist_b)),
        anchor=np.concatenate((np.full(trans.n_rows, -1, dtype=np.int64), anchor_b)),
        bridge_type=np.concatenate((np.full(trans.n_rows, -1, dtype=np.int8), bridge_type_b)),
    )
    edges.validate(nodes.n_nodes)
    segments = contract_segments(
        trans, nodes, edges, modes, max_len=cfg.segment_max_len
    )
    temporal_lengths = segments.kappa[segments.kind == 0]
    goal_modes, goal_mode_counts = np.unique(modes[success], return_counts=True)
    goal_mode_distribution = {
        str(int(mode)): int(count)
        for mode, count in zip(goal_modes, goal_mode_counts, strict=True)
    }
    if temporal_lengths.size:
        segment_stats = {
            "segment_count": int(temporal_lengths.size),
            "segment_length_p50": float(np.percentile(temporal_lengths, 50)),
            "segment_length_p95": float(np.percentile(temporal_lengths, 95)),
            "segment_length_max": int(np.max(temporal_lengths)),
        }
    else:
        segment_stats = {
            "segment_count": 0,
            "segment_length_p50": 0.0,
            "segment_length_p95": 0.0,
            "segment_length_max": 0,
        }
    meta = {
        "graph_kind": "state",
        "d_step": float(d_step),
        "n_state_occurrences": int(states.shape[0]),
        "n_states": int(states.shape[0]),
        "n_temporal_edges": int(trans.n_rows),
        "n_mode_values": int(np.unique(modes).size),
        "goal_mode_distribution": goal_mode_distribution,
        **segment_stats,
        **bridge_meta,
        **cost_meta,
    }
    return nodes, edges, meta
