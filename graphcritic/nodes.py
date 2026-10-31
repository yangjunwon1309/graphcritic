"""Radius-based state merging and representative graph-node construction."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import GraphConfig
from graphcritic.types import NodeSet, TransitionSet


def build_nodes(
    trans: TransitionSet,
    cfg: GraphConfig,
) -> NodeSet:
    """Build state nodes for every current and episode-final next state."""
    raise NotImplementedError
