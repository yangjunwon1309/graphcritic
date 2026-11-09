"""One-shot layered shortest paths for validated action-chunk bridges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class LayeredPathSolution:
    """Shortest paths with an upper bound on the number of bridge edges.

    Layer ``b`` represents the number of additional bridges still available.
    ``cost_steps[b, s]`` is the exact minimum cost from state ``s`` using at
    most ``b`` bridges. Successors are flattened layer/state indices.
    """

    cost_steps: np.ndarray
    successor: np.ndarray
    bridge_count: np.ndarray
    used_bridge_mask: np.ndarray


def layered_reverse_dijkstra(
    n_nodes: int,
    goals: np.ndarray,
    base_src: np.ndarray,
    base_dst: np.ndarray,
    base_weight: np.ndarray,
    bridge_src: np.ndarray,
    bridge_dst: np.ndarray,
    bridge_weight: np.ndarray,
    *,
    b_max: int = 2,
) -> LayeredPathSolution:
    """Run one reverse Dijkstra on the budget-layered graph.

    Data edges stay within a layer. A bridge edge goes from layer ``b`` to
    layer ``b - 1``. All goal copies are zero-cost sources, giving the exact
    shortest path under a limit of at most ``b`` bridges for each layer.
    """
    n = int(n_nodes)
    bmax = int(b_max)
    goal_ids = np.asarray(goals, dtype=np.int64)
    src = np.asarray(base_src, dtype=np.int64)
    dst = np.asarray(base_dst, dtype=np.int64)
    weight = np.asarray(base_weight, dtype=np.float64)
    bsrc = np.asarray(bridge_src, dtype=np.int64)
    bdst = np.asarray(bridge_dst, dtype=np.int64)
    bweight = np.asarray(bridge_weight, dtype=np.float64)
    if n <= 0 or bmax < 0:
        raise ValueError("n_nodes must be positive and b_max nonnegative")
    if src.shape != dst.shape or src.shape != weight.shape:
        raise ValueError("base edge arrays must align")
    if bsrc.shape != bdst.shape or bsrc.shape != bweight.shape:
        raise ValueError("bridge edge arrays must align")
    if (np.any(goal_ids < 0) or np.any(goal_ids >= n)
            or np.any(src < 0) or np.any(src >= n)
            or np.any(dst < 0) or np.any(dst >= n)
            or np.any(bsrc < 0) or np.any(bsrc >= n)
            or np.any(bdst < 0) or np.any(bdst >= n)):
        raise ValueError("node index is outside the graph")
    # Learned quasimetric bridges can have an exact zero cost (for example,
    # when two observed occurrences have identical state vectors). Temporal
    # edges remain positive, but the generic layered solver must preserve the
    # judge's bridge cost without adding an artificial epsilon.
    if np.any(~np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("base edge weights must be finite and positive")
    if np.any(~np.isfinite(bweight)) or np.any(bweight < 0):
        raise ValueError("bridge edge weights must be finite and non-negative")

    layer_count = bmax + 1
    layer_ids = np.arange(layer_count, dtype=np.int64)
    base_src_layers = (layer_ids[:, None] * n + src[None, :]).reshape(-1)
    base_dst_layers = (layer_ids[:, None] * n + dst[None, :]).reshape(-1)
    base_weight_layers = np.broadcast_to(weight, (layer_count, len(weight))).reshape(-1)
    if bmax and len(bsrc):
        bridge_layers = np.arange(1, layer_count, dtype=np.int64)
        bridge_src_layers = (bridge_layers[:, None] * n + bsrc[None, :]).reshape(-1)
        bridge_dst_layers = ((bridge_layers[:, None] - 1) * n + bdst[None, :]).reshape(-1)
        bridge_weight_layers = np.broadcast_to(bweight, (bmax, len(bweight))).reshape(-1)
        edge_src = np.concatenate((base_src_layers, bridge_src_layers))
        edge_dst = np.concatenate((base_dst_layers, bridge_dst_layers))
        edge_weight = np.concatenate((base_weight_layers, bridge_weight_layers))
    else:
        edge_src, edge_dst, edge_weight = base_src_layers, base_dst_layers, base_weight_layers

    total_nodes = layer_count * n
    reverse = csr_matrix((edge_weight, (edge_dst, edge_src)), shape=(total_nodes, total_nodes))
    goal_layers = (layer_ids[:, None] * n + goal_ids[None, :]).reshape(-1)
    if goal_layers.size:
        costs, predecessor, _ = dijkstra(
            reverse, directed=True, indices=goal_layers, min_only=True,
            return_predecessors=True,
        )
        costs = np.asarray(costs, dtype=np.float64)
        successor = np.asarray(predecessor, dtype=np.int64)
        successor[successor < 0] = -1
    else:
        costs = np.full(total_nodes, np.inf, dtype=np.float64)
        successor = np.full(total_nodes, -1, dtype=np.int64)

    cost_layers = costs.reshape(layer_count, n)
    successor_layers = successor.reshape(layer_count, n)
    cost_layers[:, goal_ids] = 0.0
    successor_layers[:, goal_ids] = -1

    # Positive weights ensure every chosen successor has a smaller cost.
    bridge_counts_flat = np.zeros(total_nodes, dtype=np.int16)
    finite_order = np.flatnonzero(np.isfinite(costs))
    # Process equal-cost successor layers from the lower bridge budget first.
    # This handles zero-cost bridge edges, which cross from layer b to b-1
    # without decreasing the numeric distance.
    finite_order = finite_order[np.lexsort((finite_order // n, costs[finite_order]))]
    for node in finite_order:
        nxt = int(successor[node])
        if nxt >= 0:
            crosses_layer = int(node // n != nxt // n)
            bridge_counts_flat[node] = bridge_counts_flat[nxt] + crosses_layer
    bridge_counts = bridge_counts_flat.reshape(layer_count, n)

    used = np.zeros(len(bsrc), dtype=np.bool_)
    for layer in range(1, layer_count):
        used |= successor_layers[layer, bsrc] == ((layer - 1) * n + bdst)
    return LayeredPathSolution(
        cost_steps=cost_layers,
        successor=successor_layers,
        bridge_count=bridge_counts,
        used_bridge_mask=used,
    )
