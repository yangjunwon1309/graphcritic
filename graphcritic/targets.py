"""Graph-based critic target construction and cost-unit conversion."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import TargetConfig
from graphcritic.types import NodeSet, TargetSet, TransitionSet


def critic_targets(
    trans: TransitionSet,
    nodes: NodeSet,
    cost: NDArray[np.float64],
    gamma: float,
    c_max_steps: float,
    cfg: TargetConfig,
) -> TargetSet:
    raise NotImplementedError
