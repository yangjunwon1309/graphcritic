"""Graph-based critic target construction and cost-unit conversion."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import TargetConfig
from graphcritic.types import GraphArtifact, SchemaError, TargetSet, TransitionSet
from graphcritic.weights import step_cost


def critic_targets(
    trans: TransitionSet,
    artifact: GraphArtifact,
    cfg: TargetConfig,
) -> TargetSet:
    trans.validate()
    artifact.validate(trans)
    h = cfg.h
    gamma = float(artifact.config.gamma)
    c = step_cost(gamma)
    c_max_steps = float(artifact.meta["c_max_steps"])
    if not np.isfinite(c_max_steps) or c_max_steps <= 0:
        raise SchemaError("artifact c_max_steps must be finite and positive")

    n_rows = trans.n_rows
    valid = np.zeros(n_rows, dtype=np.bool_)
    source = np.full(n_rows, 3, dtype=np.uint8)
    n_starts = max(0, n_rows - h + 1)
    if n_starts:
        # Match QC sample_sequence: a terminal is allowed at the chunk end,
        # but not before it. Every complete chunk has a valid endpoint.
        prefix = np.r_[0, np.cumsum(trans.terminal, dtype=np.int64)]
        starts = np.arange(n_starts, dtype=np.int64)
        valid[:n_starts] = (
            prefix[starts + h - 1] - prefix[starts] == 0
        )

        success_window = np.stack(
            [trans.success[offset : offset + n_starts] for offset in range(h)], axis=1
        )
        has_success = np.any(success_window, axis=1)
        first_success = np.argmax(success_window, axis=1)
        endpoint_rows = starts + h - 1
        endpoint_nodes = artifact.nodes.assign_next[endpoint_rows]
        remaining = np.minimum(artifact.cost[endpoint_nodes] / c, c_max_steps)
        values = h + remaining
        values[has_success] = first_success[has_success] + 1.0

        # Preserve the explicit current-state success label, including states
        # at a sampler-segment boundary after sticky success rows.
        values[trans.success_cur[:n_starts]] = 0.0
        chunk_success = has_success | trans.success_cur[:n_starts]
        graph_used = (
            ~chunk_success
            & np.isfinite(artifact.cost[endpoint_nodes])
            & (artifact.cost[endpoint_nodes] / c < c_max_steps)
        )
        clipped = valid[:n_starts] & ~chunk_success & ~graph_used
        source[:n_starts][valid[:n_starts] & chunk_success] = 0
        source[:n_starts][graph_used & valid[:n_starts]] = 1
        source[:n_starts][clipped] = 2
        y_steps = np.zeros(n_rows, dtype=np.float64)
        y_steps[:n_starts] = np.where(valid[:n_starts], values, 0.0)
    else:
        y_steps = np.zeros(n_rows, dtype=np.float64)

    if not np.all(np.isfinite(y_steps)) or np.any(y_steps < 0):
        raise SchemaError("critic targets must be finite and non-negative")
    y = y_steps * c if cfg.unit == "nats" else y_steps
    targets = TargetSet(
        y=y.astype(np.float32),
        valid=valid,
        h=h,
        unit=cfg.unit,
        graph_hash=artifact.content_hash,
        dataset_id=str(artifact.meta["dataset_id"]),
        source=source,
    )
    targets.validate(trans)
    return targets
