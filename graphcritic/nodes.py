"""Radius-based state merging and representative graph-node construction."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import NodeSet


def build_nodes(
    feat: NDArray[np.float64],
    radius: float,
    goal_mask: NDArray[np.bool_],
    consistency: bool = True,
) -> NodeSet:
    raise NotImplementedError
