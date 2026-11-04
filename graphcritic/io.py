"""Safe NPZ + JSON persistence for schema-versioned graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import GraphConfig
from graphcritic.types import (
    SCHEMA_VERSION,
    EdgeSet,
    GraphArtifact,
    NodeSet,
    SchemaError,
    SimStateSet,
    TargetSet,
    TransitionSet,
    _content_hash,
)


def _pair_paths(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path)
    if requested.suffix in {".npz", ".json"}:
        stem = requested.with_suffix("")
    else:
        stem = requested
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def _array_manifest(arrays: Mapping[str, NDArray[Any]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "dtype": np.asarray(arrays[name]).dtype.str, "shape": list(np.asarray(arrays[name]).shape)}
        for name in sorted(arrays)
    ]


def _write_pair(
    path: str | Path,
    arrays: Mapping[str, NDArray[Any]],
    metadata: dict[str, Any],
) -> None:
    npz_path, json_path = _pair_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    safe_arrays: dict[str, NDArray[Any]] = {}
    for name, value in arrays.items():
        array = np.array(value, copy=True) if np.asarray(value).ndim == 0 else np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise SchemaError(f"array {name!r} has object dtype; pickle is not supported")
        safe_arrays[name] = array
    metadata = dict(metadata)
    metadata["arrays"] = _array_manifest(safe_arrays)
    with npz_path.open("wb") as stream:
        np.savez(stream, **safe_arrays)
    json_path.write_text(
        json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_pair(path: str | Path) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    npz_path, json_path = _pair_paths(path)
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read artifact metadata at {json_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        found = metadata.get("schema_version") if isinstance(metadata, dict) else None
        raise SchemaError(f"unsupported or missing schema_version: {found!r}")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {
                name: (
                    np.array(archive[name], copy=True)
                    if archive[name].ndim == 0
                    else np.ascontiguousarray(archive[name])
                )
                for name in archive.files
            }
    except (OSError, ValueError, EOFError) as exc:
        raise SchemaError(f"cannot read safe NPZ arrays at {npz_path}") from exc
    if any(array.dtype.hasobject for array in arrays.values()):
        raise SchemaError("object arrays are not allowed")

    declared = metadata.get("arrays")
    if not isinstance(declared, list):
        raise SchemaError("metadata is missing the array manifest")
    expected = {entry.get("name"): entry for entry in declared if isinstance(entry, dict)}
    if len(expected) != len(declared) or set(expected) != set(arrays):
        raise SchemaError("NPZ array names do not match the JSON array manifest")
    for name, array in arrays.items():
        entry = expected[name]
        if entry.get("dtype") != array.dtype.str or entry.get("shape") != list(array.shape):
            raise SchemaError(f"NPZ array {name!r} does not match its declared dtype and shape")
    return arrays, metadata


def _require_hash(metadata: Mapping[str, Any], config: Any, arrays: Mapping[str, NDArray[Any]]) -> str:
    stored = metadata.get("content_hash")
    actual = _content_hash(str(metadata["schema_version"]), config, arrays)
    if not isinstance(stored, str) or stored != actual:
        raise SchemaError("artifact content_hash does not match its contents")
    return actual


def _transition_arrays(trans: TransitionSet) -> dict[str, NDArray[Any]]:
    arrays = {
        "feat": trans.feat,
        "next_feat": trans.next_feat,
        "episode_id": trans.episode_id,
        "t": trans.t,
        "success": trans.success,
        "success_cur": trans.success_cur,
        "terminal": trans.terminal,
        "row_idx": trans.row_idx,
        "mode": trans.mode,
        "mode_next": trans.mode_next,
    }
    if trans.action is not None:
        arrays["action"] = trans.action
    return arrays


def save_transitions(
    trans: TransitionSet,
    path: str | Path,
    *,
    dataset_id: str,
    source: str = "",
) -> str:
    """Save transitions to `<stem>.npz` and `<stem>.json`; return content hash."""
    trans.validate()
    if not dataset_id:
        raise SchemaError("dataset_id must be non-empty")
    arrays = _transition_arrays(trans)
    content_hash = _content_hash(SCHEMA_VERSION, {}, arrays)
    _write_pair(
        path,
        arrays,
        {
            "schema_version": SCHEMA_VERSION,
            "config": {},
            "meta": {"dataset_id": dataset_id, "source": source, "n_rows": trans.n_rows},
            "content_hash": content_hash,
        },
    )
    return content_hash


def load_transitions(
    path: str | Path,
    *,
    expected_dataset_id: str | None = None,
) -> tuple[TransitionSet, dict[str, Any]]:
    arrays, metadata = _read_pair(path)
    required = {
        "feat", "next_feat", "episode_id", "t", "success", "success_cur", "terminal",
        "row_idx", "mode", "mode_next",
    }
    if not required.issubset(arrays) or set(arrays) - required - {"action"}:
        raise SchemaError(f"transition arrays must contain {sorted(required)} and optional action")
    _require_hash(metadata, {}, arrays)
    meta = metadata.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("dataset_id"), str):
        raise SchemaError("transition metadata must include a dataset_id")
    if expected_dataset_id is not None and meta["dataset_id"] != expected_dataset_id:
        raise SchemaError("transition dataset_id does not match the expected dataset")
    trans = TransitionSet(**arrays)
    if int(meta.get("n_rows", -1)) != trans.n_rows:
        raise SchemaError("transition metadata n_rows does not match the arrays")
    return trans, meta


def save_sim_state(sim_state: SimStateSet, path: str | Path) -> None:
    """Save an aligned simulator reset-state sidecar as a safe NPZ."""
    sim_state.validate()
    target = Path(path)
    if target.suffix != ".npz":
        target = target / "sim_state.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, NDArray[Any]] = {
        "qpos": sim_state.qpos,
        "qvel": sim_state.qvel,
        "schema_version": np.asarray(SCHEMA_VERSION),
        "state_order": np.asarray("state_table"),
    }
    if sim_state.button_states is not None:
        arrays["button_states"] = sim_state.button_states
    with target.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_sim_state(path: str | Path, *, n_states: int | None = None) -> SimStateSet:
    """Load and validate a simulator sidecar without enabling pickle."""
    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as archive:
            required = {"qpos", "qvel", "schema_version", "state_order"}
            if not required.issubset(archive.files) or set(archive.files) - required - {"button_states"}:
                raise SchemaError("sim_state.npz has unexpected or missing arrays")
            if str(archive["schema_version"].item()) != SCHEMA_VERSION:
                raise SchemaError("sim_state.npz schema_version must be 0.3")
            if str(archive["state_order"].item()) != "state_table":
                raise SchemaError("sim_state.npz state_order must be 'state_table'")
            sim_state = SimStateSet(
                qpos=archive["qpos"],
                qvel=archive["qvel"],
                button_states=archive["button_states"] if "button_states" in archive.files else None,
            )
    except (OSError, ValueError, EOFError) as exc:
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"cannot read simulator sidecar at {target}") from exc
    sim_state.validate(n_states)
    return sim_state


def _graph_arrays(artifact: GraphArtifact) -> dict[str, NDArray[Any]]:
    feature_meta = artifact.meta.get("feature")
    checkpoint_hash = (
        feature_meta.get("checkpoint_sha256") or ""
        if isinstance(feature_meta, Mapping) else ""
    )
    arrays: dict[str, NDArray[Any]] = {
        "rep_feat": artifact.nodes.rep_feat,
        "rep_ref": artifact.nodes.rep_ref,
        "assign_cur": artifact.nodes.assign_cur,
        "assign_next": artifact.nodes.assign_next,
        "is_goal": artifact.nodes.is_goal,
        "src": artifact.edges.src,
        "dst": artifact.edges.dst,
        "kappa": artifact.edges.kappa,
        "n_pos": artifact.edges.n_pos,
        "n_neg": artifact.edges.n_neg,
        "kind": artifact.edges.kind,
        "dist": artifact.edges.dist,
        "anchor": artifact.edges.anchor,
        "weights": artifact.weights,
        "cost": artifact.cost,
        "succ": artifact.succ,
        "dataset_id": np.asarray(artifact.meta["dataset_id"]),
        "feature_checkpoint_sha256": np.asarray(checkpoint_hash),
    }
    tau_hash = artifact.meta.get("tau_model_sha256")
    if tau_hash is not None:
        arrays["tau_model_sha256"] = np.asarray(str(tau_hash))
    arrays.update({f"feature_{name}": value for name, value in artifact.feature_stats.items()})
    return arrays


def save_graph(artifact: GraphArtifact, path: str | Path) -> str:
    """Save a validated GraphArtifact and return its deterministic content hash."""
    artifact.validate()
    content_hash = artifact.compute_hash()
    if content_hash != artifact.content_hash:
        raise SchemaError("GraphArtifact content_hash is stale")
    _write_pair(
        path,
        _graph_arrays(artifact),
        {
            "schema_version": artifact.schema_version,
            "config": artifact.config.to_dict(),
            "meta": artifact.meta,
            "content_hash": content_hash,
        },
    )
    return content_hash


def load_graph(path: str | Path, *, trans: TransitionSet | None = None) -> GraphArtifact:
    arrays, metadata = _read_pair(path)
    config_value = metadata.get("config")
    config = GraphConfig.from_dict(config_value)
    _require_hash(metadata, config, arrays)
    required = {
        "rep_feat", "rep_ref", "assign_cur", "assign_next", "is_goal",
        "src", "dst", "kappa", "n_pos", "n_neg", "kind", "dist", "anchor",
        "weights", "cost", "succ",
        "dataset_id", "feature_checkpoint_sha256",
        "feature_mean", "feature_std",
    }
    optional = {"tau_model_sha256"}
    if not required.issubset(arrays) or set(arrays) - required - optional:
        raise SchemaError(f"graph arrays must be exactly {sorted(required)}")
    if "tau_model_sha256" in arrays:
        meta = metadata.get("meta")
        if not isinstance(meta, dict) or str(arrays["tau_model_sha256"].item()) != meta.get("tau_model_sha256"):
            raise SchemaError("tau model hash array does not match graph metadata")
    nodes = NodeSet(
        rep_feat=arrays["rep_feat"],
        rep_ref=arrays["rep_ref"],
        assign_cur=arrays["assign_cur"],
        assign_next=arrays["assign_next"],
        is_goal=arrays["is_goal"],
    )
    edges = EdgeSet(
        src=arrays["src"], dst=arrays["dst"], kappa=arrays["kappa"],
        n_pos=arrays["n_pos"], n_neg=arrays["n_neg"], kind=arrays["kind"],
        dist=arrays["dist"], anchor=arrays["anchor"],
    )
    artifact = GraphArtifact(
        schema_version=metadata["schema_version"],
        config=config,
        feature_stats={"mean": arrays["feature_mean"], "std": arrays["feature_std"]},
        nodes=nodes,
        edges=edges,
        weights=arrays["weights"],
        cost=arrays["cost"],
        succ=arrays["succ"],
        meta=metadata.get("meta", {}),
        content_hash=metadata.get("content_hash", ""),
    )
    artifact.validate(trans)
    return artifact


def _target_arrays(targets: TargetSet) -> dict[str, NDArray[Any]]:
    return {"y": targets.y, "valid": targets.valid, "source": targets.source}


def _target_config(targets: TargetSet) -> dict[str, Any]:
    return {
        "dataset_id": targets.dataset_id,
        "graph_hash": targets.graph_hash,
        "h": targets.h,
        "unit": targets.unit,
    }


def save_targets(targets: TargetSet, path: str | Path) -> str:
    targets.validate()
    arrays = _target_arrays(targets)
    config = _target_config(targets)
    content_hash = _content_hash(SCHEMA_VERSION, config, arrays)
    _write_pair(
        path,
        arrays,
        {
            "schema_version": SCHEMA_VERSION,
            "config": config,
            "meta": {},
            "content_hash": content_hash,
        },
    )
    npz_path, _ = _pair_paths(path)
    source_path = npz_path.with_name(f"{npz_path.stem}_source.npy")
    with source_path.open("wb") as stream:
        np.save(stream, targets.source, allow_pickle=False)
    return content_hash


def load_targets(
    path: str | Path,
    *,
    graph: GraphArtifact | None = None,
    expected_dataset_id: str | None = None,
) -> TargetSet:
    arrays, metadata = _read_pair(path)
    if set(arrays) != {"y", "valid", "source"}:
        raise SchemaError("target arrays must be exactly ['source', 'valid', 'y']")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise SchemaError("target metadata must include its configuration")
    _require_hash(metadata, config, arrays)
    targets = TargetSet(
        y=arrays["y"],
        valid=arrays["valid"],
        source=arrays["source"],
        h=config.get("h"),
        unit=config.get("unit"),
        graph_hash=config.get("graph_hash"),
        dataset_id=config.get("dataset_id"),
    )
    source_path = _pair_paths(path)[0].with_name(f"{_pair_paths(path)[0].stem}_source.npy")
    if source_path.is_file():
        try:
            with source_path.open("rb") as stream:
                sidecar = np.load(stream, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise SchemaError(f"cannot read target-source sidecar at {source_path}") from exc
        if not np.array_equal(sidecar, targets.source):
            raise SchemaError("target-source sidecar does not match the hashed NPZ source array")
    if expected_dataset_id is not None and targets.dataset_id != expected_dataset_id:
        raise SchemaError("target dataset_id does not match the expected dataset")
    if graph is not None:
        graph.validate()
        if targets.graph_hash != graph.content_hash:
            raise SchemaError("targets were generated from a different graph artifact")
        if targets.dataset_id != graph.meta["dataset_id"]:
            raise SchemaError("target and graph dataset_id values differ")
    return targets
