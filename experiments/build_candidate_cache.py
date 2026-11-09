"""Build or validate the exact v0.7/v0.8 cross-episode kNN cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.eval_bridges import (
    _array_sha256,
    load_or_build_candidate_pairs,
    verify_gpu_knn_against_ckdtree,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.state_graph import state_episode_time
from experiments.v08_runtime import provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-graph", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="candidate cache .npz path")
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--parity-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--node-limit", type=int)
    args = parser.parse_args(argv)
    if args.knn < 1 or args.parity_size < 0 or args.batch_size < 1:
        parser.error("knn and batch-size must be positive; parity-size must be non-negative")

    started = time.time()
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    model_data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    graph = load_graph(args.g0_graph, trans=trans)
    if graph.meta["dataset_id"] != transition_meta["dataset_id"]:
        raise ValueError("G0 graph and transition dataset IDs differ")
    if np.any(graph.edges.kind != 0):
        raise ValueError("candidate cache must be generated from temporal-only G0")
    if graph.nodes.n_nodes != len(model_data.state_raw):
        raise ValueError("G0 and model state tables are not aligned")

    episode, _ = state_episode_time(trans)
    features = np.concatenate((model_data.state_cont, model_data.state_bits), axis=1)
    parity = None
    if args.parity_size:
        parity = verify_gpu_knn_against_ckdtree(
            features, episode, sample_size=args.parity_size, k=args.knn,
            seed=args.seed, batch_size=args.batch_size,
        )
        if parity["mismatched_rows"]:
            raise RuntimeError(
                f"GPU kNN parity mismatch on {parity['mismatched_rows']} sampled rows"
            )
    cache_started = time.time()
    pairs = load_or_build_candidate_pairs(
        features, episode, graph.nodes.is_goal,
        dataset_id=transition_meta["dataset_id"], k=args.knn,
        cache_path=args.out, node_limit=args.node_limit,
        batch_size=args.batch_size, progress=True,
    )
    report = {
        **provenance(ROOT),
        "spec_version": "v0.8",
        "dataset_id": transition_meta["dataset_id"],
        "candidate_rule": "cross_episode_incl_goal_v2",
        "k": args.knn,
        "n_nodes": int(graph.nodes.n_nodes),
        "n_pairs": int(len(pairs)),
        "feature_sha256": _array_sha256(features),
        "episode_sha256": _array_sha256(episode),
        "goal_sha256": _array_sha256(graph.nodes.is_goal),
        "success_destination_count": int(np.count_nonzero(graph.nodes.is_goal[pairs[:, 1]])),
        "knn_parity": parity,
        "cache_build_or_validation_seconds": float(time.time() - cache_started),
        "total_seconds": float(time.time() - started),
        "cache_path": str(args.out),
    }
    report_path = args.out.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
