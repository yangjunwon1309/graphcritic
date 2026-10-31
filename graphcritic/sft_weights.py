"""Cost-domain advantage weights for BOOST-style weighted SFT."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import NodeSet, TransitionSet


def sft_weights(
    trans: TransitionSet,
    nodes: NodeSet,
    cost: NDArray[np.float64],
    y: NDArray[np.float64],
    beta: float,
    clip: float,
) -> NDArray[np.float64]:
    raise NotImplementedError
