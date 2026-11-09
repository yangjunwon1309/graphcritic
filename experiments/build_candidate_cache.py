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
    load_or_build_window_candidate_pairs,
    verify_gpu_knn_against_ckdtree,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows, puzzle_window_candidate_nodes
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
    parser.add_argument("--window", type=int, default=1,
                        help="use boundary nodes and trailing windows when greater than one")
    args = parser.parse_args(argv)
    if args.knn < 1 or args.parity_size < 0 or args.batch_size < 1:
        parser.error("knn and batch-size must be positive; parity-size must be non-negative")
    if args.window < 1:
        parser.error("window must be positive")

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
    frame_features = np.concatenate((model_data.state_cont, model_data.state_bits), axis=1)
    features = (
        puzzle_state_windows(
            frame_features, trans, model_data.terminal_state_index, window=args.window
        ) if args.window > 1 else frame_features
    )
    if args.window > 1:
        node_ids, candidate_episode, source_mask = puzzle_window_candidate_nodes(
            trans, graph.nodes.is_goal, window=args.window
        )
        if args.node_limit is not None:
            if args.node_limit < 1:
                parser.error("node-limit must be positive")
            node_ids = node_ids[:args.node_limit]
            candidate_episode = candidate_episode[:args.node_limit]
            source_mask = source_mask[:args.node_limit]
        candidate_features = features[node_ids]
        candidate_goals = graph.nodes.is_goal[node_ids]
    else:
        scope = len(features) if args.node_limit is None else min(len(features), args.node_limit)
        if scope < 1:
            parser.error("node-limit must be positive")
        node_ids = np.arange(scope, dtype=np.int64)
        candidate_episode = episode[:scope]
        source_mask = np.ones(scope, dtype=np.bool_)
        candidate_features = features[:scope]
        candidate_goals = graph.nodes.is_goal[:scope]
    parity = None
    if args.parity_size:
        parity = verify_gpu_knn_against_ckdtree(
            candidate_features, candidate_episode, sample_size=args.parity_size, k=args.knn,
            seed=args.seed, batch_size=args.batch_size,
        )
        if parity["mismatched_rows"]:
            raise RuntimeError(
                f"GPU kNN parity mismatch on {parity['mismatched_rows']} sampled rows"
            )
    cache_started = time.time()
    if args.window > 1:
        pairs = load_or_build_window_candidate_pairs(
            features, trans, graph.nodes.is_goal,
            dataset_id=transition_meta["dataset_id"], window=args.window,
            k=args.knn, cache_path=args.out, node_limit=args.node_limit,
            batch_size=args.batch_size, progress=True,
        )
    else:
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
        "candidate_rule": (
            f"cross_episode_incl_goal_v2_boundary_window{args.window}"
            if args.window > 1 else "cross_episode_incl_goal_v2"
        ),
        "window": int(args.window),
        "k": args.knn,
        "n_nodes": int(graph.nodes.n_nodes),
        "n_candidate_nodes": int(len(node_ids)),
        "n_boundary_sources": int(np.count_nonzero(source_mask)),
        "n_pairs": int(len(pairs)),
        "feature_sha256": _array_sha256(candidate_features),
        "episode_sha256": _array_sha256(candidate_episode),
        "goal_sha256": _array_sha256(candidate_goals),
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
