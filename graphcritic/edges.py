"""Transition-to-edge conversion and run-length edge compression."""

from graphcritic.types import EdgeSet, NodeSet, TransitionSet


def build_edges(
    trans: TransitionSet,
    nodes: NodeSet,
    compress: bool = True,
    edge_filter: object | None = None,
) -> EdgeSet:
    """Build directed graph edges from transitions and node assignments."""
    raise NotImplementedError
