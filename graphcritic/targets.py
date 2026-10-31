"""Graph-based critic target construction and cost-unit conversion."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import NodeSet, TransitionSet


def critic_targets(
    trans: TransitionSet,
    nodes: NodeSet,
    cost: NDArray[np.float64],
    gamma: float,
    c_max: float,
    unit: Literal["steps", "cost"] = "steps",
) -> NDArray[np.float64]:
    raise NotImplementedError
