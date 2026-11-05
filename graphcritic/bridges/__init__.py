"""Framework-independent graph operations for validated bridges."""

from graphcritic.bridges.check import BridgePlan, make_and_evaluate
from graphcritic.bridges.insert import (
    IterativeBridgeGraph,
    insert_validated_bridges,
    insert_validated_bridges_bulk,
)

__all__ = ["BridgePlan", "IterativeBridgeGraph", "insert_validated_bridges",
           "insert_validated_bridges_bulk", "make_and_evaluate"]
