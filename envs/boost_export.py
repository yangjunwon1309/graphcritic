"""Adapters for transition and state-coordinate exports from BOOST.

BOOST-specific extraction stays in the robot repository. Both projects hand
the graph core the same physical-coordinate names: ``r``, ``o``, ``e``,
``q_p``, ``mode``, and ``q_a``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from graphcritic.state_schema import StateCoordinates
from graphcritic.types import SchemaError


def state_coordinates_from_boost(
    values: Mapping[str, NDArray],
) -> StateCoordinates:
    """Validate a BOOST readout against Graph Critic's shared field contract."""
    required = {"r", "o", "e", "q_p", "mode", "q_a"}
    if set(values) != required:
        raise SchemaError(f"BOOST state readout fields must be exactly {sorted(required)}")
    result = StateCoordinates(
        r=np.asarray(values["r"], dtype=np.float32),
        o=np.asarray(values["o"], dtype=np.float32),
        e=np.asarray(values["e"], dtype=np.float32),
        q_p=np.asarray(values["q_p"], dtype=np.float32),
        mode=np.asarray(values["mode"], dtype=np.int64),
        q_a=np.asarray(values["q_a"], dtype=np.float32),
    )
    n = result.r.shape[0]
    if result.r.ndim != 2 or any(
        getattr(result, key).ndim != 2 or getattr(result, key).shape[0] != n
        for key in ("o", "e", "q_p", "q_a")
    ) or result.mode.shape != (n,):
        raise SchemaError("BOOST r/o/e/q_p/mode/q_a fields must be aligned matrices/vectors")
    if any(not np.all(np.isfinite(getattr(result, key))) for key in ("r", "o", "e", "q_p", "q_a")):
        raise SchemaError("BOOST continuous state coordinates must be finite")
    return result
