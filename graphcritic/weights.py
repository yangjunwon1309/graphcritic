"""Graph edge-weight policies for hop, time, and LCB costs."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import GraphConfig
from graphcritic.types import EdgeSet


def edge_weights(
    edges: EdgeSet,
    cfg: GraphConfig,
) -> NDArray[np.float64]:
    raise NotImplementedError
