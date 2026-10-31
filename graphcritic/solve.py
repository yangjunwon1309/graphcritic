"""Shortest-path graph-cost and successor computation."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import EdgeSet, NodeSet


def solve(
    nodes: NodeSet,
    edges: EdgeSet,
    w: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    raise NotImplementedError
