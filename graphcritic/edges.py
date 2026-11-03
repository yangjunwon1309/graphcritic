"""Transition-to-edge conversion and run-length edge compression."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import GraphConfig
from graphcritic.types import EdgeSet, NodeSet, TransitionSet


EdgeFilter = Callable[[NDArray, NDArray, NDArray, NDArray], NDArray[np.bool_]]


def build_edges(
    trans: TransitionSet,
    nodes: NodeSet,
    cfg: GraphConfig,
    edge_filter: EdgeFilter | None = None,
) -> EdgeSet:
    """Aggregate changing row-level transitions into unique directed edges."""
    trans.validate()
    nodes.validate(trans)
    cfg.validate()
    change_rows = np.flatnonzero(nodes.assign_cur != nodes.assign_next).astype(np.int64, copy=False)
    if change_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        edges = EdgeSet(
            src=empty, dst=empty, kappa=empty, n_pos=empty, n_neg=empty,
            kind=np.empty(0, dtype=np.int8),
        )
        edges.validate(nodes.n_nodes)
        return edges

    src_occ = nodes.assign_cur[change_rows]
    dst_occ = nodes.assign_next[change_rows]
    if cfg.compress:
        row_idx = np.arange(trans.n_rows, dtype=np.int64)
        episode_start = np.r_[True, trans.episode_id[1:] != trans.episode_id[:-1]]
        episode_start_rows = np.maximum.accumulate(np.where(episode_start, row_idx, 0))
        # Most recent change strictly before each row; shift so a change at i
        # does not shorten its own source-state dwell duration.
        previous_change = np.r_[-1, np.maximum.accumulate(
            np.where(nodes.assign_cur[:-1] != nodes.assign_next[:-1], row_idx[:-1], -1)
        )]
        start_all = np.maximum(episode_start_rows, previous_change + 1)
        start_occ = start_all[change_rows]
        kappa_occ = change_rows - start_occ + 1
    else:
        start_occ = change_rows.copy()
        kappa_occ = np.ones(change_rows.size, dtype=np.int64)

    if edge_filter is not None:
        keep = np.asarray(edge_filter(src_occ.copy(), dst_occ.copy(), start_occ.copy(), change_rows.copy()))
        if keep.shape != change_rows.shape or not np.issubdtype(keep.dtype, np.bool_):
            raise ValueError("edge_filter must return a boolean mask with one value per occurrence")
        src_occ = src_occ[keep]
        dst_occ = dst_occ[keep]
        start_occ = start_occ[keep]
        change_rows = change_rows[keep]
        kappa_occ = kappa_occ[keep]

    if change_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return EdgeSet(empty, empty, empty, empty, empty, np.empty(0, dtype=np.int8))

    order = np.lexsort((dst_occ, src_occ))
    src_sorted = src_occ[order]
    dst_sorted = dst_occ[order]
    key_start = np.r_[True, (src_sorted[1:] != src_sorted[:-1]) | (dst_sorted[1:] != dst_sorted[:-1])]
    starts = np.flatnonzero(key_start)
    counts = np.diff(np.r_[starts, order.size]).astype(np.int64)
    kappa_sorted = kappa_occ[order]
    kappa = np.maximum.reduceat(kappa_sorted, starts).astype(np.int64)
    src = src_sorted[starts].astype(np.int64)
    dst = dst_sorted[starts].astype(np.int64)
    edges = EdgeSet(
        src=src,
        dst=dst,
        kappa=kappa,
        n_pos=counts,
        n_neg=np.zeros(starts.size, dtype=np.int64),
        kind=np.zeros(starts.size, dtype=np.int8),
    )
    edges.validate(nodes.n_nodes)
    return edges
