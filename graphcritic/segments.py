"""Exact contraction of deterministic temporal chains between decision states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import EdgeSet, NodeSet, SchemaError, TransitionSet


@dataclass(frozen=True)
class SegmentGraph:
    """Compressed graph whose vertices are original decision-state IDs."""

    state_ids: NDArray[np.int64]
    src: NDArray[np.int64]
    dst: NDArray[np.int64]
    kappa: NDArray[np.int64]
    kind: NDArray[np.int8]
    dist: NDArray[np.float32]
    anchor: NDArray[np.int64]
    weights: NDArray[np.float64]


def contract_segments(
    trans: TransitionSet,
    nodes: NodeSet,
    edges: EdgeSet,
    modes: NDArray[np.int64],
    *,
    max_len: int | None = 1000,
    edge_weights: NDArray[np.float64] | None = None,
) -> SegmentGraph:
    """Contract forced temporal chains while retaining all bridge edges.

    Decisions include episode starts/ends, goals, mode changes, bridge sources
    and landings, and periodic cuts at ``max_len``. Original state IDs remain
    available in ``state_ids`` and bridge anchors retain their original IDs.
    """
    trans.validate()
    nodes.validate(trans)
    edges.validate(nodes.n_nodes)
    mode = np.asarray(modes, dtype=np.int64)
    if mode.shape != (nodes.n_nodes,):
        raise SchemaError("modes must have one label per graph state")
    if max_len is not None and (isinstance(max_len, bool) or max_len < 1):
        raise SchemaError("max_len must be a positive integer or None")

    n = nodes.n_nodes
    original_weights = (
        edges.kappa.astype(np.float64)
        if edge_weights is None else np.asarray(edge_weights, dtype=np.float64)
    )
    if original_weights.shape != (edges.n_edges,) or not np.all(np.isfinite(original_weights)):
        raise SchemaError("edge_weights must be a finite vector aligned with graph edges")
    temporal = edges.kind == 0
    bridge = edges.kind == 1
    next_state = np.full(n, -1, dtype=np.int64)
    temporal_edge = np.full(n, -1, dtype=np.int64)
    for edge_id in np.flatnonzero(temporal):
        u = int(edges.src[edge_id])
        if next_state[u] >= 0:
            raise SchemaError("each occurrence state may have at most one temporal successor")
        next_state[u] = edges.dst[edge_id]
        temporal_edge[u] = edge_id

    decision = nodes.is_goal.copy()
    decision[edges.src[bridge]] = True
    decision[edges.dst[bridge]] = True
    starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
    decision[starts] = True
    decision[next_state < 0] = True
    mode_change = np.zeros(n, dtype=np.bool_)
    has_next = next_state >= 0
    mode_change[has_next] = mode[next_state[has_next]] != mode[has_next]
    decision[mode_change] = True

    if max_len is not None:
        ends = np.r_[starts[1:], trans.n_rows]
        for start, end in zip(starts, ends, strict=True):
            current = int(start)
            steps = 0
            while next_state[current] >= 0:
                nxt = int(next_state[current])
                steps += 1
                if steps >= max_len:
                    decision[nxt] = True
                    steps = 0
                current = nxt

    state_ids = np.flatnonzero(decision).astype(np.int64)
    compressed_id = np.full(n, -1, dtype=np.int64)
    compressed_id[state_ids] = np.arange(state_ids.size, dtype=np.int64)
    srcs: list[int] = []
    dsts: list[int] = []
    kappas: list[int] = []
    kinds: list[int] = []
    distances: list[float] = []
    anchors: list[int] = []
    weights: list[float] = []

    for state in state_ids:
        current = int(state)
        if next_state[current] < 0:
            continue
        length = 0
        segment_weight = 0.0
        while next_state[current] >= 0:
            edge_id = int(temporal_edge[current])
            length += int(edges.kappa[edge_id])
            segment_weight += float(original_weights[edge_id])
            current = int(next_state[current])
            if decision[current]:
                break
        if compressed_id[current] < 0:
            raise SchemaError("contracted temporal chain ended at a non-decision state")
        if current == int(state):
            continue
        srcs.append(int(compressed_id[state]))
        dsts.append(int(compressed_id[current]))
        kappas.append(length)
        kinds.append(0)
        distances.append(0.0)
        anchors.append(-1)
        weights.append(segment_weight)

    for edge_id in np.flatnonzero(bridge):
        u, v = int(edges.src[edge_id]), int(edges.dst[edge_id])
        if compressed_id[u] < 0 or compressed_id[v] < 0:
            raise SchemaError("bridge endpoints must be decision states")
        srcs.append(int(compressed_id[u]))
        dsts.append(int(compressed_id[v]))
        kappas.append(int(edges.kappa[edge_id]))
        kinds.append(1)
        distances.append(float(edges.dist[edge_id]))
        anchors.append(int(edges.anchor[edge_id]))
        weights.append(float(original_weights[edge_id]))

    order = np.lexsort((np.asarray(kinds), np.asarray(dsts), np.asarray(srcs))) if srcs else np.empty(0, dtype=np.int64)
    return SegmentGraph(
        state_ids=state_ids,
        src=np.asarray(srcs, dtype=np.int64)[order],
        dst=np.asarray(dsts, dtype=np.int64)[order],
        kappa=np.asarray(kappas, dtype=np.int64)[order],
        kind=np.asarray(kinds, dtype=np.int8)[order],
        dist=np.asarray(distances, dtype=np.float32)[order],
        anchor=np.asarray(anchors, dtype=np.int64)[order],
        weights=np.asarray(weights, dtype=np.float64)[order],
    )
