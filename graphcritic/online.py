"""Online graph updates from newly observed transitions."""

from graphcritic.types import GraphArtifact, TransitionSet


def update(artifact: GraphArtifact, new_trans: TransitionSet) -> GraphArtifact:
    """Add transitions to an artifact and recompute graph values."""
    raise NotImplementedError
