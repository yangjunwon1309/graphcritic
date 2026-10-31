"""Graph edge-weight policies for hop, time, and LCB costs."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import EdgeSet


def edge_weights(
    edges: EdgeSet,
    policy: Literal["hop", "time", "lcb"],
) -> NDArray[np.float64]:
    raise NotImplementedError
