"""Cost-domain advantage weights for BOOST-style weighted SFT."""

import numpy as np
from numpy.typing import NDArray

from graphcritic.types import GraphArtifact, SchemaError, TargetSet, TransitionSet
from graphcritic.weights import step_cost


def sft_weights(
    trans: TransitionSet,
    artifact: GraphArtifact,
    targets: TargetSet,
    beta: float,
    w_max: float,
) -> NDArray[np.float64]:
    trans.validate()
    artifact.validate(trans)
    targets.validate(trans)
    if targets.kind != "cost":
        raise SchemaError("SFT cost-advantage weights require kind='cost' targets")
    if targets.graph_hash != artifact.content_hash:
        raise SchemaError("targets were generated from a different graph artifact")
    if targets.dataset_id != artifact.meta["dataset_id"]:
        raise SchemaError("target and graph dataset_id values differ")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(w_max) or w_max <= 0:
        raise ValueError("w_max must be finite and positive")

    c = step_cost(artifact.config.gamma)
    max_cost = c * float(artifact.meta["c_max_steps"])
    current_cost = np.minimum(artifact.cost[artifact.nodes.assign_cur], max_cost)
    target_nats = targets.y.astype(np.float64)
    if targets.unit == "steps":
        target_nats *= c
    advantage = current_cost - target_nats
    # Clipping the exponent before exp avoids overflow while preserving the
    # requested final cap; very negative advantages safely underflow to zero.
    max_log = float(np.log(w_max))
    log_weights = np.minimum(float(beta) * advantage, max_log)
    weights = np.zeros(trans.n_rows, dtype=np.float64)
    weights[targets.valid] = np.exp(log_weights[targets.valid])
    weights = np.clip(weights, 0.0, float(w_max))
    if not np.all(np.isfinite(weights)):
        raise SchemaError("SFT weights must be finite")
    return weights
