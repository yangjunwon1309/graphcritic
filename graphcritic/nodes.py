"""Radius-based state merging and representative graph-node construction."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from graphcritic.config import GraphConfig
from graphcritic.featurize import Standardizer, state_table
from graphcritic.types import NodeSet, SchemaError, TransitionSet


_NUMBA_GREEDY_ASSIGN = None
_NUMBA_LOOKED_UP = False


def _get_numba_greedy_assign():
    """Load an optional JIT accelerator without making it a core dependency."""
    global _NUMBA_GREEDY_ASSIGN, _NUMBA_LOOKED_UP
    if _NUMBA_LOOKED_UP:
        return _NUMBA_GREEDY_ASSIGN
    _NUMBA_LOOKED_UP = True
    try:
        from numba import njit

        @njit
        def greedy_assign(labels, indptr, neighbors):
            cluster_id = 0
            for leader in range(labels.size):
                if labels[leader] >= 0:
                    continue
                labels[leader] = cluster_id
                for offset in range(indptr[leader], indptr[leader + 1]):
                    neighbor = neighbors[offset]
                    if labels[neighbor] < 0:
                        labels[neighbor] = cluster_id
                cluster_id += 1
            return cluster_id

        _NUMBA_GREEDY_ASSIGN = greedy_assign
    except Exception:  # Numba is optional; the NumPy/SciPy path stays supported.
        _NUMBA_GREEDY_ASSIGN = None
    return _NUMBA_GREEDY_ASSIGN


def _projected_search_features(
    member_features: NDArray[np.float32], radius: float
) -> tuple[NDArray, float, float]:
    """Build a complete candidate-search embedding for exact radius queries."""
    search_features: NDArray = member_features
    eps = 0.0
    search_radius = float(radius)
    p = member_features.shape[1]
    # Keep projected candidate lists bounded as the configured radius grows.
    # Orthogonal projection is non-expansive; full-feature filtering below
    # removes false positives without losing any true neighbors.
    projection_dim = 32 if radius >= 1.5 else 16
    if p > projection_dim and member_features.shape[0] >= 50_000:
        values64 = member_features.astype(np.float64)
        centered = values64 - values64.mean(axis=0)
        covariance = centered.T @ centered / max(1, centered.shape[0])
        _, eigenvectors = np.linalg.eigh(covariance)
        basis = eigenvectors[:, -projection_dim:]
        search_features = np.ascontiguousarray(centered @ basis)
        del values64, centered, covariance, eigenvectors, basis
        # For cKDTree's eps traversal, a branch is pruned only when its lower
        # bound exceeds search_radius/(1+eps). This inflated radius therefore
        # preserves all candidates within the exact radius, with a margin for
        # floating point projection error.
        eps = 0.5
        search_radius = float(radius) * (1.0 + eps) * (1.0 + 1e-5)
    return search_features, eps, search_radius


def _greedy_cluster_on_demand(
    member_features: NDArray[np.float32],
    search_features: NDArray,
    tree: cKDTree,
    radius: float,
    eps: float,
    search_radius: float,
) -> tuple[NDArray[np.int64], int]:
    """Apply the exact greedy radius clustering while querying only leaders.

    Dense radius graphs can have billions of neighbor pairs even though the
    greedy cover needs queries only for cluster leaders. Small batches keep
    candidate memory bounded; assigned seeds in each batch are skipped.
    """
    n = member_features.shape[0]
    radius2 = float(radius) ** 2
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0
    query_batch_size = 128
    for start in range(0, n, query_batch_size):
        end = min(start + query_batch_size, n)
        seeds = np.flatnonzero(labels[start:end] < 0).astype(np.int64) + start
        if seeds.size == 0:
            continue
        candidates = tree.query_ball_point(
            search_features[seeds],
            search_radius,
            eps=eps,
            workers=-1,
            return_sorted=True,
        )
        counts = np.fromiter((len(row) for row in candidates), dtype=np.int64, count=seeds.size)
        candidate_indices = np.concatenate(candidates).astype(np.int64, copy=False)
        rows = np.repeat(np.arange(seeds.size, dtype=np.int64), counts)
        delta = (
            member_features[seeds[rows]].astype(np.float64)
            - member_features[candidate_indices].astype(np.float64)
        )
        exact = np.einsum("ij,ij->i", delta, delta) <= radius2
        rows = rows[exact]
        candidate_indices = candidate_indices[exact]
        indptr = np.empty(seeds.size + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(np.bincount(rows, minlength=seeds.size), out=indptr[1:])

        # The seeds are already in state order. A prior seed in this batch can
        # absorb later seeds, so check labels again before making each cluster.
        for pos, seed in enumerate(seeds):
            if labels[seed] >= 0:
                continue
            labels[seed] = cluster_id
            neighbors = candidate_indices[indptr[pos] : indptr[pos + 1]]
            available = neighbors[labels[neighbors] < 0]
            labels[available] = cluster_id
            cluster_id += 1
    return labels, cluster_id


def _cluster_group(
    z: NDArray[np.float32],
    members: NDArray[np.int64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return per-member cluster labels, first state IDs, and medoid state IDs."""
    member_features = np.ascontiguousarray(z[members])
    search_features, eps, search_radius = _projected_search_features(member_features, radius)
    tree = cKDTree(search_features)
    if member_features.shape[1] > 16 and members.size >= 50_000 and radius >= 1.0:
        labels, n_clusters = _greedy_cluster_on_demand(
            member_features, search_features, tree, radius, eps, search_radius
        )
    else:
        indptr = np.empty(members.size + 1, dtype=np.int64)
        indptr[0] = 0
        neighbor_parts: list[np.ndarray] = []
        chunk_size = 8_192 if radius >= 0.5 else 32_768
        radius2 = float(radius) ** 2
        for start in range(0, members.size, chunk_size):
            end = min(start + chunk_size, members.size)
            candidates = tree.query_ball_point(
                search_features[start:end],
                search_radius,
                eps=eps,
                workers=-1,
                return_sorted=True,
            )
            counts = np.fromiter((len(row) for row in candidates), dtype=np.int64, count=end - start)
            candidate_indices = np.concatenate(candidates).astype(np.int64, copy=False)
            rows = np.repeat(np.arange(start, end, dtype=np.int64), counts)
            delta = (
                member_features[rows].astype(np.float64)
                - member_features[candidate_indices].astype(np.float64)
            )
            exact = np.einsum("ij,ij->i", delta, delta) <= radius2
            rows = rows[exact]
            candidate_indices = candidate_indices[exact]
            exact_counts = np.bincount(rows - start, minlength=end - start)
            indptr[start + 1 : end + 1] = indptr[start] + np.cumsum(exact_counts)
            neighbor_parts.append(candidate_indices)
        neighbor = np.ascontiguousarray(np.concatenate(neighbor_parts))

        labels = np.full(members.size, -1, dtype=np.int64)
        greedy_assign = _get_numba_greedy_assign()
        if greedy_assign is not None:
            n_clusters = int(greedy_assign(labels, indptr, neighbor))
        else:
            n_clusters = 0
            for leader in range(members.size):
                if labels[leader] >= 0:
                    continue
                labels[leader] = n_clusters
                neighbors = neighbor[indptr[leader] : indptr[leader + 1]]
                available = neighbors[labels[neighbors] < 0]
                labels[available] = n_clusters
                n_clusters += 1

    # Compute cluster means and medoids in vectorized passes. This avoids a
    # Python loop over every node when a high-dimensional dataset has few
    # radius merges and therefore close to one million singleton clusters.
    cluster_counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    centroid = np.empty((n_clusters, z.shape[1]), dtype=np.float64)
    for dim in range(z.shape[1]):
        centroid[:, dim] = np.bincount(
            labels,
            weights=member_features[:, dim].astype(np.float64),
            minlength=n_clusters,
        ) / cluster_counts

    distance2 = np.empty(members.size, dtype=np.float64)
    block_size = 65536
    for start in range(0, members.size, block_size):
        end = min(start + block_size, members.size)
        delta = (
            member_features[start:end].astype(np.float64)
            - centroid[labels[start:end]]
        )
        distance2[start:end] = np.einsum("ij,ij->i", delta, delta)

    first_local = np.full(n_clusters, members.size, dtype=np.int64)
    np.minimum.at(first_local, labels, np.arange(members.size, dtype=np.int64))
    min_distance2 = np.full(n_clusters, np.inf, dtype=np.float64)
    np.minimum.at(min_distance2, labels, distance2)
    max_state = np.iinfo(np.int64).max
    rep_candidates = np.where(distance2 == min_distance2[labels], members, max_state)
    representatives = np.full(n_clusters, max_state, dtype=np.int64)
    np.minimum.at(representatives, labels, rep_candidates)
    return labels, members[first_local], representatives


def build_nodes(
    trans: TransitionSet,
    cfg: GraphConfig,
    stdz: Standardizer,
    *,
    cluster_features: NDArray[np.float32] | None = None,
    state_modes: NDArray[np.int64] | None = None,
    radius: float | None = None,
) -> NodeSet:
    """Legacy node graph: greedily merge states within each goal/mode class."""
    trans.validate()
    cfg.validate()
    x, ref, is_success, stored_modes = state_table(trans)
    if stdz.mean.shape != (trans.feat.shape[1],):
        raise SchemaError("standardizer feature dimension does not match transitions")
    z = stdz.transform(x) if cluster_features is None else np.asarray(cluster_features, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != x.shape[0] or z.shape[1] == 0 or not np.all(np.isfinite(z)):
        raise SchemaError("cluster_features must be a finite (n_states, feature_dim) matrix")
    modes = stored_modes if state_modes is None else np.asarray(state_modes, dtype=np.int64)
    if modes.shape != (x.shape[0],):
        raise SchemaError("state_modes must have one value per state occurrence")
    cluster_radius = float(cfg.radius if radius is None else radius)
    if cluster_radius <= 0 or not np.isfinite(cluster_radius):
        raise SchemaError("effective clustering radius must be finite and positive")
    n_rows = trans.n_rows
    n_states = x.shape[0]
    if n_states == 0:
        return NodeSet(
            rep_feat=np.empty((0, trans.feat.shape[1]), dtype=np.float32),
            rep_ref=np.empty((0, 2), dtype=np.int64),
            assign_cur=np.empty((0,), dtype=np.int64),
            assign_next=np.empty((0,), dtype=np.int64),
            is_goal=np.empty((0,), dtype=np.bool_),
        )

    state_to_node = np.full(n_states, -1, dtype=np.int64)
    leader_parts: list[np.ndarray] = []
    representative_parts: list[np.ndarray] = []
    goal_parts: list[np.ndarray] = []
    n_clusters_total = 0
    group_keys = sorted({(bool(goal), int(mode)) for goal, mode in zip(is_success, modes, strict=True)})
    for goal_flag, mode_value in group_keys:
        members = np.flatnonzero((is_success == goal_flag) & (modes == mode_value))
        labels, leaders, representatives = _cluster_group(z, members, cluster_radius)
        state_to_node[members] = labels + n_clusters_total
        leader_parts.append(leaders)
        representative_parts.append(representatives)
        goal_parts.append(np.full(leaders.size, goal_flag, dtype=np.bool_))
        n_clusters_total += leaders.size

    # Cluster IDs from the two success classes are interleaved by their first
    # global state occurrence, exactly as the public schema requires.
    all_leaders = np.concatenate(leader_parts)
    all_representatives = np.concatenate(representative_parts)
    all_goals = np.concatenate(goal_parts)
    cluster_order = np.argsort(all_leaders, kind="stable")
    old_to_new = np.empty(n_clusters_total, dtype=np.int64)
    old_to_new[cluster_order] = np.arange(n_clusters_total, dtype=np.int64)
    state_to_node = old_to_new[state_to_node]
    representative_states = all_representatives[cluster_order]

    assign_cur = state_to_node[:n_rows]
    assign_next = np.empty(n_rows, dtype=np.int64)
    nonterminal = ~trans.terminal
    if np.any(nonterminal):
        assign_next[nonterminal] = assign_cur[np.flatnonzero(nonterminal) + 1]
    final_rows = np.flatnonzero(trans.terminal)
    if final_rows.size:
        final_state_ids = n_rows + np.arange(final_rows.size, dtype=np.int64)
        assign_next[final_rows] = state_to_node[final_state_ids]

    nodes = NodeSet(
        rep_feat=x[representative_states],
        rep_ref=ref[representative_states],
        assign_cur=assign_cur.copy(),
        assign_next=assign_next,
        is_goal=all_goals[cluster_order],
    )
    nodes.validate(trans)
    return nodes
