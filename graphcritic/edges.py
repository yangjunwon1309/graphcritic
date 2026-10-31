"""Transition-to-edge conversion and run-length edge compression."""

from collections.abc import Callable

from graphcritic.config import GraphConfig
from graphcritic.types import EdgeSet, NodeSet, TransitionSet


def build_edges(
    trans: TransitionSet,
    nodes: NodeSet,
    cfg: GraphConfig,
    edge_filter: Callable[..., object] | None = None,
) -> EdgeSet:
    """Build data edges from row-level transitions and node assignments."""
    raise NotImplementedError
