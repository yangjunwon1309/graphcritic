"""Shortest-path graph-cost and successor computation."""

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from graphcritic.types import EdgeSet, NodeSet, SchemaError


def solve(
    nodes: NodeSet,
    edges: EdgeSet,
    w: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    nodes.validate()
    edges.validate(nodes.n_nodes)
    weights = np.asarray(w, dtype=np.float64)
    if weights.shape != (edges.n_edges,):
        raise SchemaError("w must have one value per edge")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise SchemaError("Dijkstra edge weights must be finite and strictly positive")

    n_nodes = nodes.n_nodes
    goals = np.flatnonzero(nodes.is_goal).astype(np.int64, copy=False)
    if goals.size == 0:
        warnings.warn("graph has no goal nodes; all graph costs are unreachable", RuntimeWarning, stacklevel=2)
        return np.full(n_nodes, np.inf, dtype=np.float64), np.full(n_nodes, -1, dtype=np.int64)

    if edges.n_edges == 0:
        return np.where(nodes.is_goal, 0.0, np.inf), np.full(n_nodes, -1, dtype=np.int64)

    # CSR construction sums duplicate (row, column) entries. Collapse parallel
    # original transitions first, retaining the cheapest data/bridge edge.
    order = np.lexsort((weights, edges.dst, edges.src))
    src = edges.src[order]
    dst = edges.dst[order]
    ordered_weights = weights[order]
    first_pair = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
    src = src[first_pair]
    dst = dst[first_pair]
    ordered_weights = ordered_weights[first_pair]

    # Reverse each edge: goal-rooted paths in this graph correspond to
    # original node-to-goal paths. The predecessor is the original successor.
    reverse = csr_matrix((ordered_weights, (dst, src)), shape=(n_nodes, n_nodes))
    distances, predecessors, _ = dijkstra(
        reverse,
        directed=True,
        indices=goals,
        min_only=True,
        return_predecessors=True,
    )
    cost = np.asarray(distances, dtype=np.float64)
    pred = np.asarray(predecessors, dtype=np.int64)
    succ = np.where(pred < 0, -1, pred).astype(np.int64, copy=False)
    cost[nodes.is_goal] = 0.0
    succ[nodes.is_goal] = -1
    succ[~np.isfinite(cost)] = -1
    return cost, succ
