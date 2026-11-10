"""Build a v0.8 puzzle graph with either the flow or quasimetric judge.

Both judges share the exact cached candidates, optimistic C0 filter, layered
shortest-path pass, bridge pruning, and regret-target writer. This entry point
does no simulator stepping.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.eval_bridges import (
    CANDIDATE_RULE,
    _array_sha256,
    load_or_build_candidate_pairs,
    load_or_build_window_candidate_pairs,
)
from experiments.v08_runtime import (
    load_quasimetric_checkpoint,
    make_distance_predictor,
    make_embedding_cache_predictor,
    provenance,
)
from graphcritic.bridges.object_state import frame_object_features_from_checkpoint
from graphcritic.bridges.insert import layered_reverse_dijkstra
from graphcritic.bridges.judge_flow import FlowJudge
from graphcritic.bridges.judge_quasimetric import (
    QuasimetricJudge,
    intersect_quasimetric_judges,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows, puzzle_window_candidate_nodes
from graphcritic.config import TargetConfig
from graphcritic.io import load_graph, load_sim_state, load_transitions, save_graph, save_targets
from graphcritic.modes import state_episode_time
from graphcritic.puzzle import minimum_press_solution, toggle_matrix
from graphcritic.targets import critic_targets
from graphcritic.types import EdgeSet, GraphArtifact, SchemaError
from graphcritic.weights import step_cost
from experiments.v07_runtime import load_model_pair, make_bridge_evaluator


def _flow_judge(args, data, features, dataset_id: str) -> FlowJudge:
    model_pair = load_model_pair(
        args.checkpoint, data.n_cont, data.n_bits,
        expected_dataset_id=dataset_id,
    )
    evaluate_plans = make_bridge_evaluator(model_pair[0], model_pair[1])
    batch_seed = [0]

    def evaluate(source: np.ndarray, destination: np.ndarray):
        count = len(source)
        ids = np.arange(count, dtype=np.int64)
        seed = args.seed + batch_seed[0]
        batch_seed[0] += count
        return evaluate_plans(
            source, destination, ids, ids,
            epsilon_z=data.epsilon_z,
            state_std=data.state_std,
            delta_std=data.delta_std,
            n_cont=data.n_cont,
            n_bits=data.n_bits,
            h=5,
            max_chunks=args.max_chunks,
            n_samples=args.plan_samples,
            seed=seed,
            candidate_batch=args.judge_batch_size,
        )

    del features  # the closure receives the already-indexed feature vectors
    return FlowJudge(
        dataset_id, evaluate, horizon_steps=5, max_chunks=args.max_chunks
    )


def _apply_bridge_lambda(cost_steps: np.ndarray, bridge_lambda_steps: float) -> np.ndarray:
    """Add the per-bridge penalty after acceptance, without changing distance."""
    costs = np.asarray(cost_steps, dtype=np.float64)
    if costs.ndim != 1 or np.any(~np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("accepted bridge costs must be a finite non-negative vector")
    if not math.isfinite(bridge_lambda_steps) or bridge_lambda_steps < 0:
        raise ValueError("bridge_lambda_steps must be finite and non-negative")
    if bridge_lambda_steps == 0:
        return costs.copy()
    return costs + float(bridge_lambda_steps)


def _make_artifact(
    base: GraphArtifact,
    bridge_src: np.ndarray,
    bridge_dst: np.ndarray,
    bridge_cost_steps: np.ndarray,
    bridge_distance: np.ndarray,
    cost_steps: np.ndarray,
    successor: np.ndarray,
    *,
    judge_name: str,
    model_metadata: dict,
    candidate_rule: str,
    window_size: int,
    bridge_lambda_steps: float = 0.0,
) -> GraphArtifact:
    src = np.asarray(bridge_src, dtype=np.int64)
    dst = np.asarray(bridge_dst, dtype=np.int64)
    costs = np.asarray(bridge_cost_steps, dtype=np.float64)
    distances = np.asarray(bridge_distance, dtype=np.float32)
    if src.shape != dst.shape or src.shape != costs.shape or src.shape != distances.shape:
        raise ValueError("bridge endpoint, cost, and distance arrays must align")
    if np.any(~np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("bridge costs must be finite and non-negative")
    if not math.isfinite(bridge_lambda_steps) or bridge_lambda_steps < 0:
        raise ValueError("bridge_lambda_steps must be finite and non-negative")

    # kappa is legacy integer step metadata. Graph weights store penalized
    # bridge costs while edge dist preserves the judge's unpenalized distance.
    kappa = np.maximum(1, np.ceil(costs).astype(np.int64))
    old = base.edges
    n_added = len(src)
    edges = EdgeSet(
        src=np.concatenate((old.src, src)),
        dst=np.concatenate((old.dst, dst)),
        kappa=np.concatenate((old.kappa, kappa)),
        n_pos=np.concatenate((old.n_pos, np.zeros(n_added, dtype=np.int64))),
        n_neg=np.concatenate((old.n_neg, np.zeros(n_added, dtype=np.int64))),
        kind=np.concatenate((old.kind, np.ones(n_added, dtype=np.int8))),
        dist=np.concatenate((old.dist, distances)),
        anchor=np.concatenate((old.anchor, dst)),
        bridge_type=np.concatenate((old.bridge_type, np.full(n_added, -1, dtype=np.int8))),
    )
    cost_cfg = replace(
        base.config.bridge.cost,
        policy="time",
        tau_max_steps=int(math.ceil(20.0 + bridge_lambda_steps)),
        lambda0_steps=0.0, lambda1_steps=0.0,
    )
    bridge_cfg = replace(
        base.config.bridge, enabled=True, strategy="puzzle_r",
        state_schema="ogbench_puzzle", cost=cost_cfg,
    )
    config = replace(base.config, bridge=bridge_cfg)
    c = step_cost(config.gamma)
    weights = np.concatenate((base.weights, costs * c))
    meta = dict(base.meta)
    meta.update({
        "spec_version": "v0.8",
        "bridge_judge": judge_name,
        "bridge_strategy": "judge_all_optimistic_candidates_layered_dijkstra",
        "bridge_candidate_rule": candidate_rule,
        "bridge_window_size": int(window_size),
        "bridge_max_added_per_path": 2,
        "bridge_max_judge_distance_steps": 20.0 if judge_name == "quasimetric" else None,
        "bridge_lambda_steps": float(bridge_lambda_steps),
        "bridge_cost_definition": (
            ("quasimetric_distance_steps" if bridge_lambda_steps == 0
             else "quasimetric_distance_steps_plus_bridge_lambda_steps")
            if judge_name == "quasimetric" else
            ("flow_chunk_count_times_5_steps" if bridge_lambda_steps == 0
             else "flow_chunk_count_times_5_steps_plus_bridge_lambda_steps")
        ),
        "bridge_model_metadata": model_metadata,
    })
    return GraphArtifact(
        config=config,
        feature_stats=base.feature_stats,
        nodes=base.nodes,
        edges=edges,
        weights=weights,
        cost=np.asarray(cost_steps, dtype=np.float64) * c,
        succ=np.asarray(successor, dtype=np.int64).copy(),
        meta=meta,
    )


def _parse(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=("flow", "quasimetric"), required=True)
    parser.add_argument("--g0-graph", type=Path, required=True)
    parser.add_argument("--reference-graph", type=Path)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument(
        "--candidate-window", type=int, default=None,
        help="window size used to generate shared candidate pairs (defaults from the quasi checkpoint or cache)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-node-limit", type=int)
    parser.add_argument("--judge-batch-size", type=int, default=1024)
    parser.add_argument("--plan-samples", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--max-bridge-distance", type=float, default=20.0)
    parser.add_argument(
        "--bridge-lambda-steps", type=float, default=0.0,
        help="extra pessimism cost added per accepted bridge, in steps",
    )
    parser.add_argument("--embedding-cache", type=Path,
                        help="float32 .npy cache for per-node IQE embeddings")
    parser.add_argument(
        "--embedding-pair-batch-size", type=int, default=4096,
        help="node pairs per GPU embedding lookup; lower this if GPU memory is limited",
    )
    parser.add_argument("--object-event-checkpoint", type=Path,
                        help="optional object event-count quasimetric checkpoint")
    parser.add_argument("--object-motion-checkpoint", type=Path,
                        help="optional object motion-time quasimetric checkpoint")
    parser.add_argument("--event-max", default="auto",
                        help="event limit: auto=ceil(K/T_I) from event metadata, or a non-negative integer")
    parser.add_argument("--diagnostic-oracle-np-max", type=int,
                        help="Puzzle-only exact GF(2) press-count filter; diagnostic graph, never a QC target")
    parser.add_argument("--oracle-reference-candidates", type=Path,
                        help="D1 judge_candidates.npz used to measure D3 recall on the n_p<=limit, d_theta<=K subset")
    return parser.parse_args(argv)


def _load_object_judge(
    checkpoint: Path,
    *,
    expected_dataset_id: str,
    expected_cost: str,
    expected_window: int,
    expected_seed: int,
    trans,
    sim_state,
    node_ids: np.ndarray,
    candidate_cache: Path,
    pair_batch_size: int,
):
    """Load and cache one object model after validating its data contract."""
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_kind") != "object-quasimetric-v0.8.2":
        raise ValueError(f"{checkpoint} is not a v0.8.2 object quasimetric checkpoint")
    if metadata.get("object_cost") != expected_cost:
        raise ValueError(f"{expected_cost} judge requires an object_cost={expected_cost!r} checkpoint")
    if int(metadata.get("window", -1)) != expected_window:
        raise ValueError("object checkpoint window does not match candidate window")
    if metadata.get("object_state_schema") != "puzzle-button-onehot-depth-velocity-v1":
        raise ValueError("object checkpoint uses an unknown object-state schema")
    if metadata.get("dataset_id") != expected_dataset_id:
        raise ValueError("object checkpoint dataset ID does not match graph-build data")
    if int(metadata.get("data_split_seed", metadata.get("split_seed", -1))) != expected_seed:
        raise ValueError("object checkpoint episode-split seed does not match graph features")
    feature_all = frame_object_features_from_checkpoint(
        trans, sim_state,
        continuous_mean=np.asarray(metadata["continuous_mean"], dtype=np.float32),
        continuous_std=np.asarray(metadata["continuous_std"], dtype=np.float32),
        window=expected_window,
    )
    if feature_all.shape[1] != int(metadata.get("input_dim", -1)):
        raise ValueError("object checkpoint input dimension does not match object features")
    model, params, metadata = load_quasimetric_checkpoint(
        checkpoint, input_dim=feature_all.shape[1],
        expected_dataset_id=expected_dataset_id,
    )
    cache_path = candidate_cache.parent / "embeddings" / (
        f"object-{expected_cost}-{metadata['params_sha256'][:16]}-"
        f"{_array_sha256(node_ids)[:16]}.npy"
    )
    predict_pairs, report = make_embedding_cache_predictor(
        model, params, feature_all[node_ids], node_ids,
        params_sha256=str(metadata["params_sha256"]), dataset_id=expected_dataset_id,
        cache_path=cache_path, pair_batch_size=pair_batch_size,
    )
    return predict_pairs, metadata, report


def _oracle_press_counts(pairs: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Exact minimum GF(2) press count for each graph-node pair."""
    button_count = int(boards.shape[1])
    layout = {9: (3, 3), 16: (4, 4), 20: (4, 5)}
    if button_count not in layout:
        raise SchemaError("oracle press-count filter supports 3x3, 4x4, and 4x5 boards")
    toggle = toggle_matrix(*layout[button_count])
    memo: dict[int, int] = {}
    result = np.empty(len(pairs), dtype=np.int16)
    for i, (src, dst) in enumerate(np.asarray(pairs, dtype=np.int64)):
        difference = np.asarray(boards[int(src)] ^ boards[int(dst)], dtype=np.uint8)
        key = sum(int(bit) << j for j, bit in enumerate(difference))
        count = memo.get(key)
        if count is None:
            try:
                solution = minimum_press_solution(toggle, difference)
            except SchemaError:
                count = -1
            else:
                count = int(np.count_nonzero(solution))
            memo[key] = count
        result[i] = count
    return result


def _oracle_candidate_recall(
    oracle_press_count: np.ndarray,
    oracle_distance: np.ndarray,
    judged_accept: np.ndarray,
    *,
    max_press_count: int,
    max_distance_steps: float = 20.0,
) -> dict:
    """Measure D3 acceptance recall on D1's exact feasible-candidate subset."""
    counts = np.asarray(oracle_press_count, dtype=np.int64)
    distance = np.asarray(oracle_distance, dtype=np.float32)
    accept = np.asarray(judged_accept, dtype=np.bool_)
    if counts.ndim != 1 or distance.shape != counts.shape or accept.shape != counts.shape:
        raise ValueError("D1 oracle counts/distances and D3 acceptance must align")
    eligible = (counts >= 0) & (counts <= int(max_press_count)) & (distance <= float(max_distance_steps))
    denominator = int(np.count_nonzero(eligible))
    retained = int(np.count_nonzero(eligible & accept))
    return {
        "eligible_d1_candidate_count": denominator,
        "retained_by_d3_count": retained,
        "recall": float(retained / denominator) if denominator else None,
        "definition": "D1 n_p<=max_press_count and d_theta<=K, retained by D3",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if (args.knn != 32 or args.delta != 5.0 or args.judge_batch_size < 1
            or args.embedding_pair_batch_size < 1):
        raise SystemExit("v0.8 fixes k=32 and delta=5; judge-batch-size must be positive")
    if not 1 <= args.max_chunks <= 4 or args.plan_samples < 1 or args.max_bridge_distance != 20.0:
        raise SystemExit("v0.8 fixes K=20 and at most four flow chunks")
    if not math.isfinite(args.bridge_lambda_steps) or args.bridge_lambda_steps < 0:
        raise SystemExit("--bridge-lambda-steps must be finite and non-negative")
    if args.diagnostic_oracle_np_max is not None and args.diagnostic_oracle_np_max < 0:
        raise SystemExit("--diagnostic-oracle-np-max must be non-negative")
    if args.object_event_checkpoint is not None or args.object_motion_checkpoint is not None:
        if args.judge != "quasimetric":
            raise SystemExit("object-state filters are only supported with --judge quasimetric")
    if args.diagnostic_oracle_np_max is not None and args.judge != "quasimetric":
        raise SystemExit("--diagnostic-oracle-np-max requires --judge quasimetric")
    if args.diagnostic_oracle_np_max is not None and args.object_event_checkpoint is not None:
        raise SystemExit("diagnostic oracle replaces --object-event-checkpoint")
    if args.event_max != "auto":
        try:
            parsed_event_max = int(args.event_max)
        except (TypeError, ValueError) as exc:
            raise SystemExit("--event-max must be 'auto' or a non-negative integer") from exc
        if parsed_event_max < 0:
            raise SystemExit("--event-max must be 'auto' or a non-negative integer")
    else:
        parsed_event_max = None
    if (parsed_event_max is not None and args.object_event_checkpoint is None
            and args.diagnostic_oracle_np_max is None):
        raise SystemExit("an explicit --event-max requires an event checkpoint or oracle filter")
    if args.diagnostic_oracle_np_max is not None:
        stale_targets = [
            args.out / "targets.npz", args.out / "targets.json",
            args.out / "targets_source.npy",
        ]
        if any(path.exists() for path in stale_targets):
            raise SystemExit(
                "oracle diagnostic output already contains QC targets; choose a fresh --out directory"
            )
    if args.oracle_reference_candidates is not None and (
        args.object_event_checkpoint is None and args.object_motion_checkpoint is None
    ):
        raise SystemExit("--oracle-reference-candidates is for measuring an object-filtered D3 run")
    started = time.time()
    stages: dict[str, float] = {}
    trans, trans_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    base = load_graph(args.g0_graph, trans=trans)
    dataset_id = trans_meta["dataset_id"]
    if base.meta["dataset_id"] != dataset_id:
        raise ValueError("G0 graph and transition dataset IDs differ")
    if np.any(base.edges.kind != 0):
        raise ValueError("judged bridge building must start from temporal-only G0")
    if base.nodes.n_nodes != len(data.state_raw):
        raise ValueError("G0 nodes and puzzle model state vectors are not aligned")
    reference = load_graph(args.reference_graph, trans=trans) if args.reference_graph else None
    if reference is not None and reference.meta["dataset_id"] != dataset_id:
        raise ValueError("reference graph and G0 graph dataset IDs differ")

    frame_features = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    episode, _ = state_episode_time(trans)
    features = frame_features  # judge inputs; Flow consumes the physical frame vector
    model_window = 1
    candidate_window = args.candidate_window
    model = params = model_meta = None
    embedding_node_ids = np.arange(len(frame_features), dtype=np.int64)
    if args.judge == "quasimetric":
        raw_model_meta = json.loads((args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        model_window = int(raw_model_meta.get("window", 1))
        if candidate_window is not None and candidate_window != model_window:
            raise ValueError("candidate window must match the quasimetric checkpoint window")
        candidate_window = model_window
        features = (
            puzzle_state_windows(
                frame_features, trans, data.terminal_state_index, window=model_window
            ) if model_window > 1 else frame_features
        )
        model, params, model_meta = load_quasimetric_checkpoint(
            args.checkpoint, input_dim=features.shape[1], expected_dataset_id=dataset_id,
        )
    elif candidate_window is None:
        # Flow comparisons can consume the same windowed cache as the
        # quasimetric judge. Infer the rule from an existing cache when the
        # caller does not state it explicitly.
        if args.candidate_cache.is_file():
            with np.load(args.candidate_cache, allow_pickle=False) as cache:
                candidate_window = int(cache["window"].item()) if "window" in cache.files else 1
        else:
            candidate_window = 1
    if candidate_window is None or candidate_window < 1:
        raise ValueError("candidate window must be a positive integer")
    if ((args.object_event_checkpoint is not None or args.object_motion_checkpoint is not None)
            and candidate_window != 5):
        raise ValueError("v0.8.2 object checkpoints require the five-frame candidate window")
    if candidate_window > 1:
        candidate_features_all = (
            features if args.judge == "quasimetric" and model_window == candidate_window
            else puzzle_state_windows(
                frame_features, trans, data.terminal_state_index, window=candidate_window
            )
        )
        embedding_node_ids, _, _ = puzzle_window_candidate_nodes(
            trans, base.nodes.is_goal, window=candidate_window
        )
        if args.candidate_node_limit is not None:
            limit = int(args.candidate_node_limit)
            if limit < 1:
                raise ValueError("candidate node limit must be positive")
            embedding_node_ids = embedding_node_ids[:limit]
    else:
        candidate_features_all = frame_features
        embedding_node_ids = np.arange(len(frame_features), dtype=np.int64)
    cache_started = time.time()
    if candidate_window > 1:
        pairs = load_or_build_window_candidate_pairs(
            candidate_features_all, trans, base.nodes.is_goal,
            dataset_id=dataset_id, window=candidate_window, k=args.knn,
            cache_path=args.candidate_cache, node_limit=args.candidate_node_limit,
            progress=True,
        )
        candidate_rule = f"{CANDIDATE_RULE}_boundary_window{candidate_window}"
        candidate_feature_hash = _array_sha256(candidate_features_all[embedding_node_ids])
        candidate_episode_ids = np.concatenate((
            trans.episode_id,
            trans.episode_id[np.flatnonzero(trans.terminal)],
        ))[embedding_node_ids]
        candidate_episode_hash = _array_sha256(candidate_episode_ids)
        candidate_goal_hash = _array_sha256(base.nodes.is_goal[embedding_node_ids])
    else:
        pairs = load_or_build_candidate_pairs(
            frame_features, episode, base.nodes.is_goal,
            dataset_id=dataset_id, k=args.knn, cache_path=args.candidate_cache,
            node_limit=args.candidate_node_limit, progress=True,
        )
        candidate_rule = CANDIDATE_RULE
        candidate_feature_hash = _array_sha256(frame_features)
        candidate_episode_hash = _array_sha256(episode)
        candidate_goal_hash = _array_sha256(base.nodes.is_goal)
    stages["candidate_cache_load_seconds"] = time.time() - cache_started
    base_c = step_cost(base.config.gamma)
    temporal = base.edges.kind == 0
    n_nodes = base.nodes.n_nodes
    goals = np.flatnonzero(base.nodes.is_goal).astype(np.int64)
    base_src = base.edges.src[temporal].astype(np.int64)
    base_dst = base.edges.dst[temporal].astype(np.int64)
    base_weight_steps = (base.weights[temporal] / base_c).astype(np.float64)

    c0_started = time.time()
    c0_solution = layered_reverse_dijkstra(
        n_nodes, goals, base_src, base_dst, base_weight_steps,
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64), b_max=0,
    )
    c0 = c0_solution.cost_steps[0]
    stages["g0_reverse_dijkstra_seconds"] = time.time() - c0_started
    if len(pairs):
        target_finite = np.isfinite(c0[pairs[:, 1]])
        with np.errstate(invalid="ignore"):
            gain = c0[pairs[:, 0]] - c0[pairs[:, 1]]
        optimistic_mask = target_finite & (gain >= 5.0 + args.delta)
        optimistic_pairs = pairs[optimistic_mask]
        excluded_infinite_destinations = int(np.count_nonzero(~target_finite))
    else:
        optimistic_pairs = np.empty((0, 2), dtype=np.int64)
        excluded_infinite_destinations = 0

    embedding_report = None
    judge_pair_batch_size = args.judge_batch_size
    judge_setup_started = time.time()
    if args.judge == "flow":
        judge = _flow_judge(args, data, features, dataset_id)
        model_meta = json.loads((args.checkpoint / "pi" / "metadata.json").read_text())
    else:
        predictor = make_distance_predictor(model, params, batch_size=args.judge_batch_size)
        predict_node_pairs = None
        if model_window > 1:
            judge_pair_batch_size = args.embedding_pair_batch_size
            node_hash = _array_sha256(embedding_node_ids)
            embedding_cache_path = args.embedding_cache
            if embedding_cache_path is None:
                embedding_cache_path = args.candidate_cache.parent / "embeddings" / (
                    f"embeddings-{model_meta['params_sha256'][:16]}-{node_hash[:16]}.npy"
                )
            predict_node_pairs, embedding_report = make_embedding_cache_predictor(
                model, params, features[embedding_node_ids], embedding_node_ids,
                params_sha256=str(model_meta["params_sha256"]), dataset_id=dataset_id,
                cache_path=embedding_cache_path,
                pair_batch_size=judge_pair_batch_size,
            )
        judge = QuasimetricJudge(
            dataset_id, predictor,
            max_distance_steps=args.max_bridge_distance,
            batch_size=judge_pair_batch_size,
            predict_node_pairs=predict_node_pairs,
        )
    object_event_predict = object_motion_predict = None
    event_model_meta = motion_model_meta = None
    object_embedding_reports: dict[str, dict] = {}
    if args.object_event_checkpoint is not None:
        object_event_predict, event_model_meta, object_embedding_reports["events"] = _load_object_judge(
            args.object_event_checkpoint,
            expected_dataset_id=dataset_id, expected_cost="events",
            expected_window=candidate_window, expected_seed=args.seed,
            trans=trans, sim_state=sim_state,
            node_ids=embedding_node_ids, candidate_cache=args.candidate_cache,
            pair_batch_size=judge_pair_batch_size,
        )
        if parsed_event_max is None:
            onset_meta = event_model_meta.get("onset_interval", {})
            interval_p1 = onset_meta.get("p1_frames", event_model_meta.get("T_I"))
            if interval_p1 is None or not math.isfinite(float(interval_p1)) or float(interval_p1) <= 0:
                raise ValueError("event checkpoint lacks a positive T_I p1 interval for --event-max auto")
            event_max = int(math.ceil(args.max_bridge_distance / float(interval_p1)))
        else:
            event_max = parsed_event_max
    else:
        event_max = parsed_event_max
    if args.object_motion_checkpoint is not None:
        object_motion_predict, motion_model_meta, object_embedding_reports["motion"] = _load_object_judge(
            args.object_motion_checkpoint,
            expected_dataset_id=dataset_id, expected_cost="motion",
            expected_window=candidate_window, expected_seed=args.seed,
            trans=trans, sim_state=sim_state,
            node_ids=embedding_node_ids, candidate_cache=args.candidate_cache,
            pair_batch_size=judge_pair_batch_size,
        )
    if int(model_meta.get("data_split_seed", model_meta.get("split_seed", args.seed))) != args.seed:
        raise ValueError("judge checkpoint episode-split seed does not match graph features")
    if "state_mean" in model_meta and not np.allclose(
        np.asarray(model_meta["state_mean"], dtype=np.float32), data.state_mean, atol=1e-6, rtol=0
    ):
        raise ValueError("judge checkpoint state mean does not match graph feature normalization")
    if "state_std" in model_meta and not np.allclose(
        np.asarray(model_meta["state_std"], dtype=np.float32), data.state_std, atol=1e-6, rtol=0
    ):
        raise ValueError("judge checkpoint state std does not match graph feature normalization")

    n_optimistic = len(optimistic_pairs)
    accepted = np.zeros(n_optimistic, dtype=np.bool_)
    judge_cost = np.full(n_optimistic, np.nan, dtype=np.float32)
    judge_distance = np.full(n_optimistic, np.nan, dtype=np.float32)
    event_scores = np.full(n_optimistic, np.nan, dtype=np.float32)
    motion_scores = np.full(n_optimistic, np.nan, dtype=np.float32)
    flow_chunk_count = np.zeros(n_optimistic, dtype=np.uint8)
    flow_sample_index = np.full(n_optimistic, -1, dtype=np.int16)
    flow_endpoint_error = np.full(n_optimistic, np.nan, dtype=np.float32)
    action_offsets = np.full(n_optimistic, -1, dtype=np.int64)
    spool_path = args.out / "accepted_actions.spool.f32"
    args.out.mkdir(parents=True, exist_ok=True)
    action_rows = 0
    judge_started = time.time()
    with spool_path.open("wb") as spool:
        for start in range(0, n_optimistic, judge_pair_batch_size):
            end = min(n_optimistic, start + judge_pair_batch_size)
            chunk_pairs = optimistic_pairs[start:end]
            result = (
                judge.judge_node_pairs(chunk_pairs)
                if args.judge == "quasimetric" and model_window > 1
                else judge.judge(features[chunk_pairs[:, 0]], features[chunk_pairs[:, 1]])
            )
            result.validate(len(chunk_pairs))
            accepted[start:end] = result.accept
            judge_cost[start:end] = result.cost_steps
            if args.judge == "quasimetric":
                judge_distance[start:end] = result.extras["distance_steps"]
                if object_event_predict is not None:
                    event_scores[start:end] = object_event_predict(chunk_pairs)
                if object_motion_predict is not None:
                    motion_scores[start:end] = object_motion_predict(chunk_pairs)
            else:
                flow_chunk_count[start:end] = result.extras["chunk_count"]
                flow_sample_index[start:end] = result.extras["sample_index"]
                flow_endpoint_error[start:end] = result.extras["endpoint_error"]
                chunks = result.extras["action_chunks"]
                for local in np.flatnonzero(result.accept):
                    global_index = start + int(local)
                    n_chunks = int(result.extras["chunk_count"][local])
                    count = n_chunks * 5
                    action_offsets[global_index] = action_rows
                    payload = np.ascontiguousarray(
                        chunks[local, :n_chunks].reshape(count, 5), dtype=np.float32
                    )
                    spool.write(payload.tobytes())
                    action_rows += len(payload)
            if end == n_optimistic or (start // judge_pair_batch_size + 1) % 100 == 0:
                print(
                    f"{args.judge} judge: {end}/{n_optimistic}, accepted={int(np.count_nonzero(accepted[:end]))}",
                    file=sys.stderr, flush=True,
                )
    stages["judge_seconds"] = time.time() - judge_started
    if embedding_report is not None:
        stages["embedding_generation_seconds"] = float(
            embedding_report["embedding_seconds"]
        )
        stages["embedding_pair_distance_seconds"] = float(
            embedding_report["pair_distance_seconds"]
        )
        stages["judge_total_including_embedding_seconds"] = (
            time.time() - judge_setup_started
        )
    else:
        stages["judge_setup_and_evaluation_seconds"] = time.time() - judge_setup_started
    oracle_counts = None
    rejection_report = None
    object_filters_active = (
        args.object_event_checkpoint is not None
        or args.object_motion_checkpoint is not None
        or args.diagnostic_oracle_np_max is not None
    )
    if args.judge == "quasimetric" and object_filters_active:
        if args.diagnostic_oracle_np_max is not None:
            if sim_state.button_states is None:
                raise SchemaError("diagnostic oracle requires the puzzle button_states sidecar")
            oracle_counts = _oracle_press_counts(optimistic_pairs, sim_state.button_states)
        rejection_report = intersect_quasimetric_judges(
            judge_distance,
            max_distance_steps=args.max_bridge_distance,
            m_psi=motion_scores if object_motion_predict is not None else None,
            e_phi=event_scores if object_event_predict is not None else None,
            max_events=event_max if object_event_predict is not None else None,
            oracle_press_count=oracle_counts,
            oracle_max_press_count=args.diagnostic_oracle_np_max,
        )
        accepted = np.asarray(rejection_report["accept"], dtype=np.bool_)
        judge_cost = np.where(accepted, judge_distance, np.nan).astype(np.float32)
    oracle_reference_recall = None
    if args.oracle_reference_candidates is not None:
        with np.load(args.oracle_reference_candidates, allow_pickle=False) as reference_candidates:
            if not {"pairs", "oracle_n_p", "distance_steps", "candidate_sha256"}.issubset(reference_candidates.files):
                raise ValueError("oracle reference file is not a D1 judge_candidates.npz")
            reference_pairs = np.asarray(reference_candidates["pairs"], dtype=np.int64)
            if not np.array_equal(reference_pairs, optimistic_pairs):
                raise ValueError("D1 and D3 optimistic candidate pair arrays differ")
            reference_hash = str(reference_candidates["candidate_sha256"].item())
            if reference_hash != _array_sha256(optimistic_pairs):
                raise ValueError("D1 oracle candidate hash does not match the current candidate cache")
            oracle_reference_recall = _oracle_candidate_recall(
                reference_candidates["oracle_n_p"],
                reference_candidates["distance_steps"],
                accepted,
                max_press_count=(
                    int(reference_candidates["oracle_n_p_max"].item())
                    if "oracle_n_p_max" in reference_candidates.files else 1
                ),
                max_distance_steps=args.max_bridge_distance,
            )
    accepted_indices = np.flatnonzero(accepted).astype(np.int64)
    accepted_pairs = optimistic_pairs[accepted_indices]
    accepted_costs = _apply_bridge_lambda(
        judge_cost[accepted_indices], args.bridge_lambda_steps
    )
    if args.judge == "quasimetric":
        accepted_distances = judge_distance[accepted_indices]
    else:
        accepted_distances = np.zeros(len(accepted_indices), dtype=np.float32)

    dijkstra_started = time.time()
    paths = layered_reverse_dijkstra(
        n_nodes, goals, base_src, base_dst, base_weight_steps,
        accepted_pairs[:, 0] if len(accepted_pairs) else np.empty(0, dtype=np.int64),
        accepted_pairs[:, 1] if len(accepted_pairs) else np.empty(0, dtype=np.int64),
        accepted_costs,
        b_max=2,
    )
    stages["layered_reverse_dijkstra_seconds"] = time.time() - dijkstra_started
    retained_mask = paths.used_bridge_mask
    retained_pairs = accepted_pairs[retained_mask]
    retained_costs = accepted_costs[retained_mask]
    retained_distances = accepted_distances[retained_mask]
    successor = paths.successor[2]
    successor = np.where(successor >= 0, successor % n_nodes, -1).astype(np.int64)

    artifact_started = time.time()
    artifact = _make_artifact(
        base, retained_pairs[:, 0] if len(retained_pairs) else np.empty(0, dtype=np.int64),
        retained_pairs[:, 1] if len(retained_pairs) else np.empty(0, dtype=np.int64),
        retained_costs, retained_distances, paths.cost_steps[2], successor,
        judge_name=args.judge, model_metadata={
            key: value for key, value in model_meta.items()
            if key in {"spec_version", "dataset_id", "params_sha256", "step", "input_dim",
                       "frame_dim", "window", "constraint_horizon", "encoder_widths",
                       "projector_widths", "n_components", "component_dim"}
        }, candidate_rule=candidate_rule, window_size=candidate_window,
        bridge_lambda_steps=args.bridge_lambda_steps,
    )
    if object_filters_active:
        artifact.meta["spec_version"] = "v0.8.2"
        if args.diagnostic_oracle_np_max is not None:
            artifact.meta["oracle_filter"] = {
                "name": "minimum_gf2_press_count",
                "max_press_count": int(args.diagnostic_oracle_np_max),
                "diagnostic_only": True,
            }
            artifact.meta["qc_eligible"] = False
        if args.object_event_checkpoint is not None or args.object_motion_checkpoint is not None:
            artifact.meta["object_judge_rule"] = {
                "acceptance": "d_theta<=K AND m_psi<=K AND e_phi<=E_max+0.5 for enabled judges",
                "K_steps": float(args.max_bridge_distance),
                "event_max": int(event_max) if object_event_predict is not None else None,
                "event_threshold": float(event_max + 0.5) if object_event_predict is not None else None,
                "event_checkpoint_sha256": (
                    event_model_meta.get("params_sha256") if event_model_meta else None
                ),
                "motion_checkpoint_sha256": (
                    motion_model_meta.get("params_sha256") if motion_model_meta else None
                ),
                "event_embedding_cache": object_embedding_reports.get("events"),
                "motion_embedding_cache": object_embedding_reports.get("motion"),
            }
    graph_hash = save_graph(artifact, args.out / "graph.npz")
    targets = None
    if args.diagnostic_oracle_np_max is None:
        targets = critic_targets(
            trans, artifact, TargetConfig(h=5, unit="steps", kind="regret")
        )
        save_targets(targets, args.out / "targets")
    np.savez_compressed(
        args.out / "layered_paths.npz",
        cost_steps=paths.cost_steps, successor=paths.successor,
        bridge_count=paths.bridge_count, b_max=np.asarray(2, dtype=np.int64),
        graph_hash=np.asarray(graph_hash),
    )
    candidate_artifact = {
        "pairs": optimistic_pairs.astype(np.int32), "accept": accepted,
        "cost_steps": judge_cost, "distance_steps": judge_distance,
        "board_changed": np.any(
            data.state_bits[optimistic_pairs[:, 0]] != data.state_bits[optimistic_pairs[:, 1]], axis=1
        ) if n_optimistic else np.empty(0, dtype=np.bool_),
        "candidate_sha256": np.asarray(_array_sha256(optimistic_pairs)),
        "dataset_id": np.asarray(dataset_id), "judge": np.asarray(args.judge),
    }
    if object_event_predict is not None:
        candidate_artifact["e_phi"] = event_scores
    if object_motion_predict is not None:
        candidate_artifact["m_psi"] = motion_scores
    if oracle_counts is not None:
        candidate_artifact["oracle_n_p"] = oracle_counts
        candidate_artifact["oracle_n_p_max"] = np.asarray(
            int(args.diagnostic_oracle_np_max), dtype=np.int16
        )
    np.savez_compressed(args.out / "judge_candidates.npz", **candidate_artifact)
    if args.judge == "flow":
        retained_indices = accepted_indices[retained_mask]
        retained_lengths = flow_chunk_count[retained_indices].astype(np.int64)
        retained_offsets = action_offsets[retained_indices]
        retained_samples = flow_sample_index[retained_indices]
        retained_errors = flow_endpoint_error[retained_indices]
        offsets = np.concatenate((
            np.zeros(1, dtype=np.int64), np.cumsum(retained_lengths * 5, dtype=np.int64)
        ))
        flat = np.memmap(spool_path, dtype=np.float32, mode="r", shape=(action_rows, 5)) if action_rows else None
        saved = np.empty((int(offsets[-1]), 5), dtype=np.float32)
        for ordinal, source_offset in enumerate(retained_offsets):
            length = int(retained_lengths[ordinal]) * 5
            destination_offset = int(offsets[ordinal])
            saved[destination_offset:destination_offset + length] = flat[
                int(source_offset):int(source_offset) + length
            ]
        np.savez_compressed(
            args.out / "accepted_actions.npz", actions=saved, offsets=offsets,
            source=retained_pairs[:, 0], target=retained_pairs[:, 1],
            length_chunks=retained_lengths, sample_index=retained_samples,
            endpoint_error=retained_errors,
        )
        if flat is not None:
            del flat
    spool_path.unlink(missing_ok=True)
    stages["graph_and_target_write_seconds"] = time.time() - artifact_started

    reference_fraction = (
        float(np.mean(np.isfinite(reference.cost))) if reference is not None else None
    )
    final_cost_steps = paths.cost_steps[2]
    original_reachable = np.isfinite(c0)
    decreased = original_reachable & (final_cost_steps < c0 - 1e-8)
    bridge_counts, bridge_frequencies = np.unique(paths.bridge_count[2], return_counts=True)
    result = {
        **provenance(ROOT),
        "spec_version": "v0.8.2" if object_filters_active else "v0.8",
        "dataset_id": dataset_id,
        "judge": args.judge,
        "bridge_lambda_steps": float(args.bridge_lambda_steps),
        "g0_hash": base.content_hash,
        "graph_hash": graph_hash,
        "model_checkpoint": str(args.checkpoint),
        "model_checkpoint_sha256": model_meta.get("params_sha256"),
        "candidate_cache": str(args.candidate_cache),
        "candidate_rule": candidate_rule,
        "window_size": int(candidate_window),
        "model_window_size": int(model_window),
        "candidate_cache_hashes": {
            "feature_sha256": candidate_feature_hash,
            "episode_sha256": candidate_episode_hash,
            "goal_sha256": candidate_goal_hash,
            "node_ids_sha256": _array_sha256(embedding_node_ids),
        },
        "optimistic_candidate_sha256": _array_sha256(optimistic_pairs),
        "n_cached_candidates": int(len(pairs)),
        "n_optimistic_candidates": int(n_optimistic),
        "n_excluded_infinite_destinations": excluded_infinite_destinations,
        "n_accepted_candidates": int(len(accepted_pairs)),
        "n_retained_bridges": int(len(retained_pairs)),
        "n_retained_success_destination_bridges": int(
            np.count_nonzero(base.nodes.is_goal[retained_pairs[:, 1]])
        ) if len(retained_pairs) else 0,
        "g0_reachable_fraction": float(np.mean(np.isfinite(c0))),
        "graph_reachable_fraction": float(np.mean(np.isfinite(final_cost_steps))),
        "reference_reachable_fraction": reference_fraction,
        "reference_reachable_difference": (
            float(np.mean(np.isfinite(final_cost_steps))) - reference_fraction
            if reference_fraction is not None else None
        ),
        "originally_reachable_states_with_reduced_cost_fraction": (
            float(np.count_nonzero(decreased) / np.count_nonzero(original_reachable))
            if np.any(original_reachable) else None
        ),
        "bridge_count_distribution": {
            str(int(count)): int(n) for count, n in zip(bridge_counts, bridge_frequencies, strict=True)
        },
        "h5_regret_valid_fraction": (
            float(np.mean(targets.valid)) if targets is not None else None
        ),
        "quasimetric_distance_summary": (
            {
                "accepted_p50": float(np.quantile(accepted_distances, 0.5)),
                "accepted_p90": float(np.quantile(accepted_distances, 0.9)),
                "accepted_max": float(np.max(accepted_distances)),
            } if args.judge == "quasimetric" and len(accepted_distances) else None
        ),
        "accepted_bridge_cost_summary": (
            {
                "p50": float(np.quantile(accepted_costs, 0.5)),
                "p90": float(np.quantile(accepted_costs, 0.9)),
                "max": float(np.max(accepted_costs)),
            } if len(accepted_costs) else None
        ),
        "embedding_cache": embedding_report,
        "object_embedding_caches": object_embedding_reports if object_embedding_reports else None,
        "stage_seconds": stages,
        "elapsed_seconds": time.time() - started,
        "simulator_q1": "not run per v0.8 minimum implementation",
        "path_q3": "not run per v0.8 minimum implementation",
    }
    if object_filters_active:
        result["condition_rejection_counts"] = rejection_report["individual_rejections"]
        result["condition_rejection_combinations"] = rejection_report["categories"]
        result["object_event_max"] = int(event_max) if object_event_predict is not None else None
        result["object_event_threshold"] = (
            float(event_max + 0.5) if object_event_predict is not None else None
        )
    if oracle_reference_recall is not None:
        result["d3_recall_against_oracle_candidates"] = oracle_reference_recall
        result["oracle_reference_candidates"] = str(args.oracle_reference_candidates)
    if args.diagnostic_oracle_np_max is not None:
        result["oracle_filter"] = {
            "name": "minimum_gf2_press_count",
            "max_press_count": int(args.diagnostic_oracle_np_max),
            "diagnostic_only": True,
            "qc_eligible": False,
            "n_p_rejected": int(np.count_nonzero(oracle_counts > args.diagnostic_oracle_np_max)),
            "n_unsolvable_rejected": int(np.count_nonzero(oracle_counts < 0)),
        }
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
