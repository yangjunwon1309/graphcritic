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
    loss_mask = np.zeros(n_rows, dtype=np.bool_)
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
        start_nodes = artifact.nodes.assign_cur[starts]
        endpoint_nodes = artifact.nodes.assign_next[endpoint_rows]
        start_raw = np.asarray(artifact.cost[start_nodes], dtype=np.float64) / c
        end_raw = np.asarray(artifact.cost[endpoint_nodes], dtype=np.float64) / c
        start_reachable = np.isfinite(start_raw)
        end_reachable = np.isfinite(end_raw)
        start_finite = start_reachable & (start_raw < c_max_steps)
        end_finite = end_reachable & (end_raw < c_max_steps)
        start_cost = np.minimum(start_raw, c_max_steps)
        end_cost = np.minimum(end_raw, c_max_steps)
        start_cost[~np.isfinite(start_cost)] = c_max_steps
        end_cost[~np.isfinite(end_cost)] = c_max_steps

        chunk_success = has_success | trans.success_cur[:n_starts]
        values = np.zeros(n_starts, dtype=np.float64)
        if cfg.kind == "cost":
            values = h + end_cost
            values[has_success] = first_success[has_success] + 1.0
            # Preserve the explicit current-state success label, including
            # states at a sampler-segment boundary after sticky success rows.
            values[trans.success_cur[:n_starts]] = 0.0
            graph_used = ~chunk_success & end_finite
            capped = valid[:n_starts] & ~chunk_success & ~end_finite
            source[:n_starts][valid[:n_starts] & chunk_success] = 0
            source[:n_starts][graph_used & valid[:n_starts]] = 1
            source[:n_starts][capped] = 2
            loss_mask[:n_starts] = valid[:n_starts]
        else:
            # A temporal path of h edges from the chunk start to endpoint
            # makes C(start) <= h + C(end). An unreachable start behind a
            # reachable endpoint is impossible. A *finite* start above the
            # target cap is different: when the endpoint is below the cap,
            # use the uncapped finite costs to retain the exact regret.
            impossible = valid[:n_starts] & ~start_reachable & end_reachable
            if np.any(impossible):
                rows = np.flatnonzero(impossible)[:8].tolist()
                raise SchemaError(
                    "regret property 1 violated: start cost is capped while endpoint is finite "
                    f"(rows {rows})"
                )

            finite_pair = start_reachable & end_finite & ~chunk_success
            capped_end = start_finite & ~end_finite & ~chunk_success
            both_capped = ~start_finite & ~end_finite & ~chunk_success
            success_rows = valid[:n_starts] & chunk_success
            finite_success = success_rows & start_finite
            if np.any(success_rows & ~start_finite):
                rows = np.flatnonzero(success_rows & ~start_finite)[:8].tolist()
                raise SchemaError(
                    "regret property 1 violated: success chunk starts at capped graph cost "
                    f"(rows {rows})"
                )

            values[finite_pair] = h + end_raw[finite_pair] - start_raw[finite_pair]
            values[capped_end] = cfg.regret_max
            success_steps = first_success.astype(np.float64) + 1.0
            success_steps[trans.success_cur[:n_starts]] = 0.0
            values[finite_success] = success_steps[finite_success] - start_cost[finite_success]

            # Allow only floating-point roundoff below zero; anything larger
            # would contradict the temporal shortest-path triangle inequality.
            finite_labels = finite_pair | finite_success
            violations = valid[:n_starts] & finite_labels & (values < -1e-5)
            if np.any(violations):
                rows = np.flatnonzero(violations)[:8].tolist()
                raise SchemaError(f"regret property 1 violated: negative regret at rows {rows}")
            values[finite_labels & (values < 0.0)] = 0.0
            values = np.minimum(values, cfg.regret_max)

            source[:n_starts][finite_success] = 0
            source[:n_starts][valid[:n_starts] & finite_pair] = 1
            source[:n_starts][valid[:n_starts] & capped_end] = 2
            source[:n_starts][valid[:n_starts] & both_capped] = 4
            loss_mask[:n_starts] = valid[:n_starts] & ~both_capped

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
        kind=cfg.kind,
        loss_mask=loss_mask,
        regret_max=cfg.regret_max,
    )
    targets.validate(trans)
    return targets
