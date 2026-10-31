"""Typed data containers shared by graph construction and target generation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TransitionSet:
    feat: NDArray[np.float64]  # shape: (N, p)
    next_feat: NDArray[np.float64]  # shape: (N, p)
    episode_id: NDArray[np.int64]  # shape: (N,)
    t: NDArray[np.int64]  # shape: (N,)
    h: NDArray[np.int64]  # shape: (N,)
    success_step: NDArray[np.int64]  # shape: (N,)
    row_idx: NDArray[np.int64]  # shape: (N,)


@dataclass
class NodeSet:
    rep_row: NDArray[np.int64]  # shape: (V,)
    assign: NDArray[np.int64]  # shape: (M,)
    is_goal: NDArray[np.bool_]  # shape: (V,)


@dataclass
class EdgeSet:
    src: NDArray[np.int64]  # shape: (E,)
    dst: NDArray[np.int64]  # shape: (E,)
    kappa: NDArray[np.float64]  # shape: (E,)
    n_pos: NDArray[np.int64]  # shape: (E,)
    n_neg: NDArray[np.int64]  # shape: (E,)


@dataclass
class GraphArtifact:
    nodes: NodeSet
    edges: EdgeSet
    cost: NDArray[np.float64]  # shape: (V,)
    succ: NDArray[np.int64]  # shape: (V,)
    meta: dict[str, Any]
