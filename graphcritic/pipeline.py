"""Deterministic assembly of validated graph artifacts."""

from __future__ import annotations

import resource
import time
from typing import Any

import numpy as np

from graphcritic.config import GraphConfig
from graphcritic.edges import build_edges
from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.modes import resolve_state_modes
from graphcritic.nodes import build_nodes
from graphcritic.solve import solve
from graphcritic.state_graph import _step_distance, build_state_graph
from graphcritic.types import GraphArtifact, SchemaError, SimStateSet, TransitionSet
from graphcritic.weights import edge_weights
from graphcritic.verification import SimVerifier, evaluate_paths, verify_bridge_edges
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
    sim_state: SimStateSet | None = None,
    verifier: SimVerifier | None = None,
) -> GraphArtifact:
    """Build nodes, edges, weights, shortest-path costs, and provenance."""
    trans.validate()
    cfg.validate()
    started = time.perf_counter()
    stdz = StateFeaturizer.fit(trans, cfg.feature)
    states, _, _, _ = state_table(trans)
    phi = stdz.transform(states)
    build_meta: dict[str, Any] = {"feature": stdz.meta()}
    if cfg.graph_kind == "state":
        nodes, edges, state_meta = build_state_graph(trans, cfg, phi, sim_state=sim_state)
        build_meta.update(state_meta)
        d_step = float(state_meta["d_step"])
        modes = resolve_state_modes(trans, cfg.mode_fn, sim_state=sim_state, min_dwell=cfg.min_dwell)
        if cfg.bridge.cost.policy == "sim_verified":
            if verifier is None or sim_state is None:
                raise SchemaError("sim_verified policy requires both sim_state and a simulator verifier")
            edges, verification_meta = verify_bridge_edges(
                trans, nodes, edges, cfg, stdz, sim_state, modes,
                d_step=d_step, verifier=verifier,
            )
            build_meta.update(verification_meta)
    else:
        modes = resolve_state_modes(trans, cfg.mode_fn, sim_state=sim_state, min_dwell=cfg.min_dwell)
        d_step = _step_distance(phi, trans, modes)
        radius = cfg.radius * d_step if cfg.radius_unit == "step" else cfg.radius
        nodes = build_nodes(
            trans, cfg, stdz, cluster_features=phi, state_modes=modes, radius=radius
        )
        edges = build_edges(trans, nodes, cfg)
        build_meta.update({"graph_kind": "node", "d_step": d_step})
    weights = edge_weights(edges, cfg, d_step=d_step, meta=build_meta)
    cost, succ = solve(nodes, edges, weights)
    if verifier is not None and sim_state is not None and cfg.graph_kind == "state":
        build_meta.update(evaluate_paths(
            trans, nodes, edges, weights, cost, succ, sim_state, verifier
        ))
    build_meta["build_seconds"] = float(time.perf_counter() - started)
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    build_meta["peak_rss_mb"] = rss / (1024.0 if rss < 10**9 else 1024.0**2)
    build_meta["n_states"] = int(states.shape[0])
    artifact = GraphArtifact(
        config=cfg,
        feature_stats=stdz.as_stats(),
        nodes=nodes,
        edges=edges,
        weights=weights,
        cost=cost,
        succ=succ,
        meta=_artifact_meta(
            trans, cfg, dataset_id=dataset_id, source=source,
            extra_meta={**build_meta, **(extra_meta or {})},
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
    stdz = StateFeaturizer.fit(trans, cfg.feature)
    weights = edge_weights(edges, cfg, d_step=(extra_meta or {}).get("d_step"), meta=extra_meta)
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
