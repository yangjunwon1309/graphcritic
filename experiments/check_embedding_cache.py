"""Check direct and cached IQE distances on a sample of candidate pairs."""

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

from experiments.eval_bridges import _array_sha256, load_or_build_window_candidate_pairs
from experiments.v08_runtime import (
    load_quasimetric_checkpoint,
    make_distance_predictor,
    make_embedding_cache_predictor,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows, puzzle_window_candidate_nodes
from graphcritic.featurize import state_table
from graphcritic.io import load_sim_state, load_transitions


def _random_cross_episode_pairs(
    node_ids: np.ndarray,
    candidate_episodes: np.ndarray,
    source_mask: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    if count < 1:
        raise ValueError("pairs must be positive")
    sources = np.flatnonzero(source_mask)
    if not len(sources) or len(node_ids) < 2:
        raise ValueError("candidate table has too few boundary nodes")
    rng = np.random.default_rng(seed)
    result = np.empty((count, 2), dtype=np.int64)
    filled = 0
    while filled < count:
        batch = max(2 * (count - filled), 1024)
        src_local = rng.choice(sources, size=batch, replace=True)
        dst_local = rng.integers(0, len(node_ids), size=batch)
        valid = candidate_episodes[src_local] != candidate_episodes[dst_local]
        chosen = np.flatnonzero(valid)[: count - filled]
        n = len(chosen)
        result[filled:filled + n, 0] = node_ids[src_local[chosen]]
        result[filled:filled + n, 1] = node_ids[dst_local[chosen]]
        filled += n
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10_000)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)

    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    frame_features = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    raw_metadata = json.loads((args.ckpt / "metadata.json").read_text(encoding="utf-8"))
    metadata_dim = int(raw_metadata.get("input_dim", frame_features.shape[1]))
    model, params, metadata = load_quasimetric_checkpoint(
        args.ckpt, input_dim=metadata_dim,
        expected_dataset_id=transition_meta["dataset_id"],
    )
    window = int(metadata.get("window", 1))
    if window <= 1:
        raise ValueError(
            "embedding-cache check is intended for the windowed candidate graph; "
            "a frame-level cache would require embeddings for every graph node"
        )
    features = (
        puzzle_state_windows(frame_features, trans, data.terminal_state_index, window=window)
        if window > 1 else frame_features
    )
    if features.shape[1] != metadata_dim:
        raise ValueError("checkpoint input_dim does not match reconstructed window features")
    _, _, goals, _ = state_table(trans)
    node_ids, episodes, source_mask = puzzle_window_candidate_nodes(
        trans, goals, window=window
    )

    if args.candidate_cache is not None:
        with np.load(args.candidate_cache, allow_pickle=False) as archive:
            cache_limit = (
                len(archive["node_ids"]) if "node_ids" in archive.files else None
            )
        if cache_limit is not None:
            node_ids, episodes, source_mask = (
                node_ids[:cache_limit], episodes[:cache_limit], source_mask[:cache_limit]
            )
        pairs_all = load_or_build_window_candidate_pairs(
            features, trans, goals,
            dataset_id=transition_meta["dataset_id"], window=window, k=32,
            cache_path=args.candidate_cache, node_limit=cache_limit,
        ).astype(np.int64, copy=False)
        if not len(pairs_all):
            raise ValueError("candidate cache is empty")
        rng = np.random.default_rng(args.seed)
        take = rng.choice(len(pairs_all), size=args.pairs, replace=len(pairs_all) < args.pairs)
        pairs = pairs_all[take]
    else:
        pairs = _random_cross_episode_pairs(
            node_ids, episodes, source_mask, args.pairs, args.seed
        )

    cache_path = args.embedding_cache
    if cache_path is None:
        node_hash = _array_sha256(node_ids)
        cache_path = args.ckpt.parent / "embedding_cache" / (
            f"embeddings-{metadata['params_sha256'][:16]}-{node_hash[:16]}.npy"
        )
    cached_predict, cache_report = make_embedding_cache_predictor(
        model, params, features[node_ids], node_ids,
        params_sha256=str(metadata["params_sha256"]),
        dataset_id=transition_meta["dataset_id"], cache_path=cache_path,
        pair_batch_size=args.batch_size,
    )
    direct_predict = make_distance_predictor(model, params, batch_size=args.batch_size)
    started = time.perf_counter()
    direct = direct_predict(features[pairs[:, 0]], features[pairs[:, 1]])
    direct_seconds = float(time.perf_counter() - started)
    started = time.perf_counter()
    cached = cached_predict(pairs)
    cached_seconds = float(time.perf_counter() - started)
    max_abs = float(np.max(np.abs(direct - cached), initial=0.0))
    mismatch = int(np.count_nonzero((direct <= 20.0) != (cached <= 20.0)))
    report = {
        "checkpoint": str(args.ckpt),
        "window": window,
        "n_pairs": int(len(pairs)),
        "max_absolute_distance_difference": max_abs,
        "bridge_decision_mismatches_at_K20": mismatch,
        "direct_pair_seconds": direct_seconds,
        "cached_pair_seconds": cached_seconds,
        "embedding_cache": cache_report,
        "pass": bool(max_abs <= 1e-4 and mismatch == 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
