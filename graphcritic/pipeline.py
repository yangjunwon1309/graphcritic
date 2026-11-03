"""Deterministic assembly of validated graph artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from graphcritic.config import GraphConfig
from graphcritic.edges import build_edges
from graphcritic.featurize import Standardizer
from graphcritic.nodes import build_nodes
from graphcritic.solve import solve
from graphcritic.types import GraphArtifact, SchemaError, TransitionSet
from graphcritic.weights import edge_weights
from graphcritic import __version__


def _max_episode_len(trans: TransitionSet) -> int:
    if trans.n_rows == 0:
        raise SchemaError("cannot build a graph from an empty transition set")
    # T1 validates local time as a zero-based consecutive sequence.
    return int(np.max(trans.t)) + 1


def _artifact_meta(
    trans: TransitionSet,
    cfg: GraphConfig,
    *,
    dataset_id: str,
    source: str,
    extra_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(dataset_id, str) or not dataset_id:
        raise SchemaError("dataset_id must be a non-empty string")
    if not isinstance(source, str):
        raise SchemaError("source must be a string")
    reserved = {"dataset_id", "source", "core_version", "n_rows", "c_max_steps"}
    extra = dict(extra_meta or {})
    collision = reserved & extra.keys()
    if collision:
        raise SchemaError(f"extra_meta cannot override reserved keys: {sorted(collision)}")
    c_max_steps = cfg.c_max_steps or float(_max_episode_len(trans) + 1)
    return {
        "dataset_id": dataset_id,
        "source": source,
        "core_version": __version__,
        "n_rows": trans.n_rows,
        "c_max_steps": float(c_max_steps),
        **extra,
    }


def build_artifact(
    trans: TransitionSet,
    cfg: GraphConfig,
    *,
    dataset_id: str,
    source: str,
    extra_meta: dict[str, Any] | None = None,
) -> GraphArtifact:
    """Build nodes, edges, weights, shortest-path costs, and provenance."""
    trans.validate()
    cfg.validate()
    stdz = Standardizer.fit(trans, cfg.feature)
    nodes = build_nodes(trans, cfg, stdz)
    edges = build_edges(trans, nodes, cfg)
    weights = edge_weights(edges, cfg)
    cost, succ = solve(nodes, edges, weights)
    artifact = GraphArtifact(
        config=cfg,
        feature_stats=stdz.as_stats(),
        nodes=nodes,
        edges=edges,
        weights=weights,
        cost=cost,
        succ=succ,
        meta=_artifact_meta(
            trans, cfg, dataset_id=dataset_id, source=source, extra_meta=extra_meta
        ),
    )
    artifact.validate(trans)
    return artifact


def rebuild_from_external(
    trans: TransitionSet,
    nodes,
    edges,
    cfg: GraphConfig,
    *,
    dataset_id: str,
    source: str,
    extra_meta: dict[str, Any] | None = None,
) -> GraphArtifact:
    """Assemble a GraphArtifact around externally constructed nodes/edges."""
    trans.validate()
    cfg.validate()
    nodes.validate(trans)
    edges.validate(nodes.n_nodes)
    stdz = Standardizer.fit(trans, cfg.feature)
    weights = edge_weights(edges, cfg)
    cost, succ = solve(nodes, edges, weights)
    artifact = GraphArtifact(
        config=cfg,
        feature_stats=stdz.as_stats(),
        nodes=nodes,
        edges=edges,
        weights=weights,
        cost=cost,
        succ=succ,
        meta=_artifact_meta(
            trans, cfg, dataset_id=dataset_id, source=source, extra_meta=extra_meta
        ),
    )
    artifact.validate(trans)
    return artifact
