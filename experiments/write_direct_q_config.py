"""Write the pinned v0.9 direct-Q config after R1 action-head training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows, puzzle_window_candidate_nodes


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_dplus_anchor(graph_meta: dict, checkpoint_meta: dict,
                           checkpoint_sha256: str,
                           build_result: dict | None = None) -> None:
    """Validate d+ provenance from the model and graph, independent of spec labels."""
    provenance_meta = build_result if build_result is not None else graph_meta
    if provenance_meta.get("target_provenance", {}).get("kind") != "learned":
        raise ValueError("dplus anchoring requires a learned L+ graph")
    graph_checkpoint_sha = provenance_meta.get("model_checkpoint_sha256")
    if graph_checkpoint_sha is None:
        graph_checkpoint_sha = graph_meta.get("model_checkpoint_sha256")
    if graph_checkpoint_sha is None:
        graph_checkpoint_sha = graph_meta.get("bridge_model_metadata", {}).get("params_sha256")
    if graph_checkpoint_sha != checkpoint_sha256:
        raise ValueError("L+ graph and dplus checkpoint hashes differ")
    if checkpoint_meta.get("match_constraint") != "object":
        raise ValueError("dplus anchors require an object-match checkpoint")


def _is_v11_same_object_graph(graph_meta: dict, build_result: dict) -> bool:
    provenance = graph_meta.get("target_provenance", {}).get("kind")
    candidate_rule = str(build_result.get("candidate_rule", graph_meta.get("candidate_rule", "")))
    return bool(
        provenance == "learned"
        and graph_meta.get("spec_version") in {None, "v1.1", "v1.2"}
        and candidate_rule.startswith("cross_episode_observed_object_change_v1_")
        and build_result.get("object_change_set_sha256")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--action-head-root", type=Path, required=True)
    parser.add_argument("--anchor-mode", choices=("same_object", "dplus"), default="same_object")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    trans, trans_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    graph = load_graph(args.graph, trans=trans)
    if graph.meta.get("dataset_id") != trans_meta["dataset_id"]:
        raise ValueError("graph and transition dataset IDs differ")
    graph_provenance = graph.meta.get("target_provenance", {}).get("kind")
    result_path = args.graph.parent / "result.json"
    build_result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    v11_same_object = args.anchor_mode == "same_object" and _is_v11_same_object_graph(
        graph.meta, build_result
    )
    if args.anchor_mode == "same_object":
        if not v11_same_object:
            if graph_provenance != "oracle_upper_bound":
                raise ValueError("same_object anchoring requires a learned v1.1 or pinned v0.9 oracle graph")
            if graph.meta.get("target_provenance", {}).get("oracle_np_max") != 1:
                raise ValueError("v0.9 upper-bound graph must use oracle_np_max=1")
            if graph.meta.get("qc_eligible") != "oracle_upper_bound":
                raise ValueError("graph is not marked as an oracle QC upper-bound artifact")
            expected_graph_hash = "bb5b26bf1776785d7196745d97ee1b98e177feeb7112c69ef6344c9c3b3548f9"
            if graph.content_hash != expected_graph_hash:
                raise ValueError("same_object anchoring is pinned to the v0.8.4b O_np1 graph")
    elif graph_provenance != "learned":
        raise ValueError("dplus anchoring requires a learned L+ graph")
    with np.load(args.candidate_cache, allow_pickle=False) as archive:
        if "dataset_id" in archive.files and str(archive["dataset_id"].item()) != trans_meta["dataset_id"]:
            raise ValueError("candidate cache dataset ID mismatch")
        candidate_cache_nodes = np.asarray(archive["node_ids"], dtype=np.int64)
        if int(archive["window"]) != 5 or int(archive["k"]) != 32:
            raise ValueError("v0.9 requires the v0.8 window-5, k=32 candidate cache")
    state_meta = json.loads((args.state_checkpoint / "metadata.json").read_text())
    state_bytes = (args.state_checkpoint / "params.msgpack").read_bytes()
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    if state_sha != state_meta.get("params_sha256"):
        raise ValueError("state checkpoint parameter hash mismatch")
    if args.anchor_mode == "dplus":
        _validate_dplus_anchor(graph.meta, state_meta, state_sha, build_result or None)
    elif v11_same_object:
        if build_result.get("model_checkpoint_sha256") != state_sha:
            raise ValueError("v1.1 graph and same_object state checkpoint hashes differ")
        if graph.meta.get("bridge_model_metadata", {}).get("params_sha256") != state_sha:
            raise ValueError("v1.1 graph metadata and same_object state checkpoint hashes differ")
    embedding_path = args.embedding_cache
    if embedding_path is None:
        candidates = []
        cache_dir = args.candidate_cache.parent / "embeddings"
        if cache_dir.is_dir():
            for metadata_path in cache_dir.glob("*.npy.json"):
                try:
                    value = json.loads(metadata_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if (value.get("params_sha256") == state_sha
                        and value.get("dataset_id") == trans_meta["dataset_id"]):
                    candidates.append(metadata_path.with_suffix(""))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "could not uniquely locate the state embedding cache; pass --embedding-cache explicitly"
            )
        embedding_path = candidates[0]
    cache_meta = json.loads(embedding_path.with_suffix(
        embedding_path.suffix + ".json"
    ).read_text())
    if cache_meta.get("params_sha256") != state_sha:
        raise ValueError("embedding cache does not use the pinned d-theta checkpoint")
    if cache_meta.get("dataset_id") != trans_meta["dataset_id"]:
        raise ValueError("embedding cache dataset ID mismatch")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.dtype != np.float32 or embeddings.shape[1] != 2048:
        raise ValueError("state embedding cache must be float32 with 2048 columns")
    model_data = prepare_puzzle_model_data(trans, sim_state, seed=int(state_meta.get("data_split_seed", 0)))
    frame = np.concatenate((model_data.state_cont, model_data.state_bits), axis=1).astype(np.float32)
    windows = puzzle_state_windows(frame, trans, model_data.terminal_state_index, window=5)
    candidate_ids, _, _ = puzzle_window_candidate_nodes(trans, graph.nodes.is_goal, window=5)
    if not np.array_equal(candidate_cache_nodes, candidate_ids):
        raise ValueError("candidate cache node order does not match the pinned boundary-node rule")
    if len(candidate_ids) != len(embeddings):
        raise ValueError("embedding cache node count does not match the v0.8 candidate-node rule")
    if _array_sha256(candidate_ids) != cache_meta.get("node_ids_sha256"):
        raise ValueError("embedding cache node IDs do not match v0.8 candidate-node order")
    if _array_sha256(windows[candidate_ids]) != cache_meta.get("node_features_sha256"):
        raise ValueError("embedding cache features do not match the fixed d-theta inputs")
    heads = [str((args.action_head_root / f"seed_{seed}" / "step_020000").resolve()) for seed in range(4)]
    missing = [head for head in heads if not (Path(head) / "metadata.json").is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed action-head checkpoints: {missing}")
    head_metas = [json.loads((Path(head) / "metadata.json").read_text()) for head in heads]
    parameterizations = {str(meta.get("parameterization", "free")) for meta in head_metas}
    if len(parameterizations) != 1:
        raise ValueError("action heads use mixed parameterizations")
    action_parameterization = next(iter(parameterizations))
    if action_parameterization not in {"free", "residual"}:
        raise ValueError("action heads have an unsupported parameterization")
    if action_parameterization == "residual":
        if args.anchor_mode != "same_object":
            raise ValueError("v1.2b residual action heads require same_object graph anchors")
        summary_path = args.action_head_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("residual direct-Q config requires the completed action-head summary.json")
        summary = json.loads(summary_path.read_text())
        if summary.get("spec_version") != "v1.2b" or summary.get("parameterization") != "residual":
            raise ValueError("residual action-head summary is not a completed v1.2b run")
        if summary.get("stopped_on_collapse") or summary.get("stopped_on_gate"):
            raise ValueError("v1.2b action-head summary records a failed stop gate")
        head_results = summary.get("heads", {})
        for seed in range(4):
            report = head_results.get(str(seed))
            if not isinstance(report, dict) or int(report.get("step", -1)) != 20_000:
                raise ValueError(f"v1.2b action-head seed {seed} did not complete 20k updates")
            if report.get("collapse_alarm") or not report.get("gate_passed"):
                raise ValueError(f"v1.2b action-head seed {seed} failed its collapse/C1/C2 gate")
            c1 = float(report.get("c1_violation_rate_d_gt_h_plus_epsilon", 1.0))
            v_ref = float(report.get("v_ref_violation_rate_d_theta_gt_h_plus_epsilon", -1.0))
            if c1 > v_ref + 0.05 or float(report.get("c2_max", 1.0)) > 1e-4:
                raise ValueError(f"v1.2b action-head seed {seed} failed its measured C1/C2 gate")
    elif args.anchor_mode == "dplus":
        summary_path = args.action_head_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("dplus direct-Q config requires the completed action-head summary.json")
        summary = json.loads(summary_path.read_text())
        if summary.get("spec_version") != "v1.0" or summary.get("stopped_on_collapse") or summary.get("stopped_on_gate"):
            raise ValueError("v1.0 action-head summary does not show four completed passing heads")
        head_results = summary.get("heads", {})
        for seed in range(4):
            report = head_results.get(str(seed))
            if not isinstance(report, dict) or int(report.get("step", -1)) != 20_000:
                raise ValueError(f"v1.0 action-head seed {seed} did not complete 20k updates")
            if report.get("collapse_alarm") or float(report.get("max_dead_unit_fraction", 1.0)) != 0.0:
                raise ValueError(f"v1.0 action-head seed {seed} failed its F3/dead-unit gate")
            if float(report.get("c1_violation_rate_d_gt_h_plus_epsilon", 1.0)) > 0.05:
                raise ValueError(f"v1.0 action-head seed {seed} failed the held-out C1 gate")
    elif v11_same_object:
        summary_path = args.action_head_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError("v1.1 direct-Q config requires the completed action-head summary.json")
        summary = json.loads(summary_path.read_text())
        if summary.get("stopped_on_collapse") or summary.get("stopped_on_gate"):
            raise ValueError("v1.1 action-head summary records a failed stop gate")
        head_results = summary.get("heads", {})
        for seed in range(4):
            report = head_results.get(str(seed))
            if not isinstance(report, dict) or int(report.get("step", -1)) != 20_000:
                raise ValueError(f"v1.1 action-head seed {seed} did not complete 20k updates")
            if report.get("collapse_alarm"):
                raise ValueError(f"v1.1 action-head seed {seed} triggered F3")
            if float(report.get("c1_violation_rate_d_gt_h_plus_epsilon", 1.0)) > 0.05:
                raise ValueError(f"v1.1 action-head seed {seed} failed held-out C1")
    config = {
        "spec_version": (
            str(graph.meta.get("spec_version", "v1.1")) if v11_same_object
            else "v1.0" if args.anchor_mode == "dplus" else "v0.9"
        ),
        "dataset_id": trans_meta["dataset_id"],
        "graph": str(args.graph.resolve()),
        "graph_hash": graph.content_hash,
        "candidate_cache": str(args.candidate_cache.resolve()),
        "embedding_cache": str(embedding_path.resolve()),
        "state_checkpoint": str(args.state_checkpoint.resolve()),
        "state_checkpoint_sha256": state_sha,
        "action_head_checkpoints": heads,
        "action_parameterization": action_parameterization,
        "anchor_mode": args.anchor_mode,
        "transitions": str(args.transitions.resolve()),
        "sim_state": str(args.sim_state.resolve()),
        "split_seed": int(state_meta.get("data_split_seed", 0)),
        "window": 5,
        "horizon": 5,
        "action_dim": int(trans.action.shape[1]),
        "k": 32,
        "successors": 4,
        "max_local_distance": 20.0,
        "ensemble": "maximum",
        "cost_layer": int(graph.meta.get("cost_layer", graph.meta.get("b_max", 2))),
        "candidate_count": int(len(embeddings)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(args.out), "graph_hash": graph.content_hash,
                      "dataset_id": trans_meta["dataset_id"], "heads": heads}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

