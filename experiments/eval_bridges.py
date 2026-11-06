"""Build sequentially verified v0.7 bridges and run its simulator witness."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binomtest, spearmanr

from graphcritic.bridges.insert import (
    IterativeBridgeGraph, insert_validated_bridges,
    insert_validated_bridges_bulk,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.config import TargetConfig
from graphcritic.io import load_graph, load_sim_state, load_transitions, save_graph, save_targets
from graphcritic.modes import state_episode_time
from graphcritic.state_schema import OGBenchPuzzleStateSchema
from graphcritic.targets import critic_targets
from graphcritic.types import EdgeSet, GraphArtifact
from graphcritic.verification import OGBenchVerifier
from graphcritic.weights import step_cost
from experiments.v07_runtime import load_model_pair, make_bridge_evaluator


def _json_safe(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def cross_episode_knn_pairs(
    state_features: np.ndarray,
    episode_id: np.ndarray,
    is_goal: np.ndarray,
    *,
    k: int = 32,
    initial_query: int = 128,
    max_query: int | None = None,
    block_size: int = 4096,
    node_limit: int | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Return directed normalized-state kNN pairs from different episodes."""
    x = np.asarray(state_features, dtype=np.float32)
    ep = np.asarray(episode_id, dtype=np.int64)
    goals = np.asarray(is_goal, dtype=np.bool_)
    if x.ndim != 2 or ep.shape != (len(x),) or goals.shape != ep.shape:
        raise ValueError("state features, episode IDs, and goal mask must align")
    n = len(x) if node_limit is None else min(len(x), int(node_limit))
    tree = cKDTree(x)
    query_limit = len(x) if max_query is None else min(len(x), int(max_query))
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for block_number, start in enumerate(range(0, n, block_size), 1):
        rows = np.arange(start, min(n, start + block_size), dtype=np.int64)
        query_k = min(query_limit, max(k + 1, initial_query))
        _, neighbor = tree.query(x[rows], k=query_k, workers=-1)
        neighbor = np.asarray(neighbor, dtype=np.int64).reshape(len(rows), query_k)
        selected: list[np.ndarray] = []
        missing: list[int] = []
        for i, row in enumerate(rows):
            cand = neighbor[i]
            # Successful dataset states are ordinary vertices with zero
            # remaining cost.  They are valid bridge destinations (M1).
            good = (cand != row) & (ep[cand] != ep[row])
            chosen = cand[good][:k]
            selected.append(chosen)
            if chosen.size < k and query_k < query_limit:
                missing.append(i)
        while missing and query_k < query_limit:
            wider = min(query_limit, max(query_k + 1, query_k * 2))
            # Bound temporary neighbor-index storage while widening. In a
            # large episode table, only sources still missing cross-episode
            # neighbors are queried again.
            rows_per_query = max(1, 2_000_000 // wider)
            still_missing = []
            for offset in range(0, len(missing), rows_per_query):
                batch_missing = missing[offset:offset + rows_per_query]
                _, extra = tree.query(x[rows[batch_missing]], k=wider, workers=-1)
                extra = np.asarray(extra, dtype=np.int64).reshape(len(batch_missing), wider)
                for pos, local_i in enumerate(batch_missing):
                    row = int(rows[local_i])
                    cand = extra[pos]
                    good = (cand != row) & (ep[cand] != ep[row])
                    chosen = cand[good][:k]
                    selected[local_i] = chosen
                    if chosen.size < k:
                        still_missing.append(local_i)
            missing = still_missing
            query_k = wider
        for local_i, chosen in enumerate(selected):
            if chosen.size:
                src_parts.append(np.full(chosen.size, rows[local_i], dtype=np.int32))
                dst_parts.append(chosen.astype(np.int32, copy=False))
        if progress and (block_number % 20 == 0 or start + len(rows) == n):
            n_pairs = sum(len(part) for part in src_parts)
            print(
                f"cross-episode kNN: states={min(start + len(rows), n)}/{n}, "
                f"pairs={n_pairs}", file=sys.stderr, flush=True,
            )
    if not src_parts:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.column_stack((np.concatenate(src_parts), np.concatenate(dst_parts)))
    # Each source produces at most k distinct neighbors, so pair deduplication
    # is unnecessary; sort is only for deterministic downstream tie handling.
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    return pairs[order].astype(np.int64, copy=False)


def load_or_build_candidate_pairs(
    features: np.ndarray,
    episode_id: np.ndarray,
    is_goal: np.ndarray,
    *,
    dataset_id: str,
    k: int,
    cache_path: Path | None,
    node_limit: int | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Load a dataset-bound kNN cache or build and save the deterministic pairs."""
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {"pairs", "dataset_id", "k", "n_nodes", "feature_dim", "node_limit"}
            if not required.issubset(cached.files):
                raise ValueError("candidate cache is missing required metadata")
            checks = {
                "dataset_id": (str(cached["dataset_id"].item()), dataset_id),
                "k": (int(cached["k"].item()), int(k)),
                "n_nodes": (int(cached["n_nodes"].item()), int(len(features))),
                "feature_dim": (int(cached["feature_dim"].item()), int(np.asarray(features).shape[1])),
                "node_limit": (
                    int(cached["node_limit"].item()), -1 if node_limit is None else int(node_limit)
                ),
            }
            for name, (actual, expected) in checks.items():
                if actual != expected:
                    raise ValueError(
                        f"candidate cache {name} mismatch: cached={actual!r}, expected={expected!r}"
                    )
            pairs = np.asarray(cached["pairs"], dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("candidate cache pairs must have shape (N, 2)")
        if pairs.size and (np.any(pairs < 0) or np.any(pairs >= len(features))):
            raise ValueError("candidate cache contains an out-of-range state index")
        return pairs

    pairs = cross_episode_knn_pairs(
        features, episode_id, is_goal, k=k, node_limit=node_limit, progress=progress
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            pairs=pairs.astype(np.int32), dataset_id=np.asarray(dataset_id),
            k=np.asarray(k, dtype=np.int64), n_nodes=np.asarray(len(features), dtype=np.int64),
            feature_dim=np.asarray(np.asarray(features).shape[1], dtype=np.int64),
            node_limit=np.asarray(-1 if node_limit is None else node_limit, dtype=np.int64),
        )
    return pairs


def _artifact_with_bridges(base: GraphArtifact, graph: IterativeBridgeGraph,
                           bridge_type: int = 1, max_chunks: int = 4) -> GraphArtifact:
    old = base.edges
    n_added = len(graph.added_src)
    src = np.concatenate((old.src, np.asarray(graph.added_src, dtype=np.int64)))
    dst = np.concatenate((old.dst, np.asarray(graph.added_dst, dtype=np.int64)))
    kappa = np.concatenate((old.kappa, np.asarray(graph.added_weight, dtype=np.int64)))
    edges = EdgeSet(
        src=src, dst=dst, kappa=kappa,
        n_pos=np.concatenate((old.n_pos, np.zeros(n_added, dtype=np.int64))),
        n_neg=np.concatenate((old.n_neg, np.zeros(n_added, dtype=np.int64))),
        kind=np.concatenate((old.kind, np.ones(n_added, dtype=np.int8))),
        dist=np.concatenate((old.dist, np.zeros(n_added, dtype=np.float32))),
        anchor=np.concatenate((old.anchor, np.asarray(graph.added_dst, dtype=np.int64))),
        bridge_type=np.concatenate((old.bridge_type, np.full(n_added, bridge_type, dtype=np.int8))),
    )
    cost_cfg = replace(
        base.config.bridge.cost, policy="time", tau_max_steps=5 * int(max_chunks),
        lambda0_steps=0.0, lambda1_steps=0.0,
    )
    bridge_cfg = replace(
        base.config.bridge, enabled=True, strategy="puzzle_r",
        state_schema="ogbench_puzzle", cost=cost_cfg,
    )
    config = replace(base.config, bridge=bridge_cfg)
    c = step_cost(config.gamma)
    weights = np.concatenate((base.weights, np.asarray(graph.added_weight, dtype=np.float64) * c))
    meta = dict(base.meta)
    meta.update({
        "bridge_v07": True,
        "bridge_strategy": "validated_iterative_action_chunk",
        "bridge_v07_max_added_per_path": 2,
        "bridge_v07_added_edges": n_added,
        "bridge_v07_graph_cost_unit": "steps",
    })
    return GraphArtifact(
        config=config, feature_stats=base.feature_stats, nodes=base.nodes,
        edges=edges, weights=weights, cost=graph.cost * c,
        succ=graph.successor.copy(), meta=meta,
    )


def _simulator_witness(
    plans: list[dict], plan_map: dict[tuple[int, int], object],
    data, sim_state, *, env_id: str, trials: int, seed: int,
) -> dict:
    if not plans:
        return {"n": 0, "landing_rate": None, "passed_gate": False, "reason": "no accepted bridges"}
    import gymnasium as gym
    import ogbench  # noqa: F401

    ids = np.arange(len(plans), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(ids) > trials:
        ids = np.sort(rng.choice(ids, size=trials, replace=False))
    env = gym.make(env_id, max_episode_steps=1_000_000_000)
    verifier = OGBenchVerifier(env.unwrapped)
    schema = OGBenchPuzzleStateSchema()
    results = []
    try:
        for ordinal in ids:
            record = plans[int(ordinal)]
            source, target = int(record["source"]), int(record["target"])
            plan = plan_map[(source, target)]
            payload = {"qpos": sim_state.qpos[source], "qvel": sim_state.qvel[source],
                       "button_states": sim_state.button_states[source]}
            observation = verifier.reset_to(payload)
            done = False
            for action in plan.chunks.reshape(-1, 5):
                observation, reward, terminated, truncated, info = verifier.step(action.astype(np.float32))
                if terminated or truncated:
                    done = True
                    break
            if done:
                hit = False
                cont_ratio = None
                board_exact = False
            else:
                readout = verifier.sim_readout(observation, schema_name="ogbench_puzzle")
                current = np.asarray(readout["r"], dtype=np.float64)
                goal = data.state_raw[target, :data.n_cont].astype(np.float64)
                cont_ratio = float(np.max(np.abs(current - goal) / data.epsilon_raw))
                board_exact = bool(np.array_equal(
                    np.asarray(readout["button_states"], dtype=np.int8),
                    data.state_bits[target].astype(np.int8),
                ))
                hit = bool(cont_ratio <= 1.0 and board_exact)
            results.append({"source": source, "target": target, "hit": hit,
                            "continuous_epsilon_ratio": cont_ratio,
                            "button_state_exact": board_exact,
                            "length_chunks": int(plan.length_chunks)})
            if len(results) % 25 == 0:
                print(f"v0.7 bridge simulator witness: {len(results)}/{len(ids)}", file=sys.stderr, flush=True)
    finally:
        env.close()
    rate = float(np.mean([x["hit"] for x in results]))
    return {"n": len(results), "landing_rate": rate, "gate": 0.8,
            "passed_gate": bool(rate >= 0.8), "results": results}


def _path_witness(
    artifact: GraphArtifact, trans, sim_state, data, plan_map: dict,
    baseline_path: Path, *, env_id: str,
) -> dict:
    """Run 500 graph paths on the baseline's exact fixed start selection."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_outcomes = baseline.get("outcomes", [])
    if len(baseline_outcomes) != 500:
        raise ValueError("Q3 requires a 500-start baseline witness")
    start_selection = np.asarray([
        (int(item["start_state"]), int(bool(item["failed_episode"])))
        for item in baseline_outcomes
    ], dtype="<i8")
    starts = start_selection[:, 0].tolist()
    digest = hashlib.sha256(np.ascontiguousarray(start_selection).tobytes()).hexdigest()
    if digest != baseline.get("start_selection_sha256"):
        raise ValueError("baseline start-selection hash does not match its outcome rows")
    try:
        import gymnasium as gym
        import ogbench  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Q3 path witnessing requires the bench environment") from exc
    env = gym.make(env_id, max_episode_steps=1_000_000_000)
    verifier = OGBenchVerifier(env.unwrapped)
    edge_lookup = {(int(s), int(t)): int(i) for i, (s, t) in enumerate(zip(
        artifact.edges.src, artifact.edges.dst, strict=True
    ))}
    c = step_cost(artifact.config.gamma)
    results = []
    try:
        for number, (start, base_result) in enumerate(zip(starts, baseline_outcomes, strict=True), 1):
            cost_steps = float(artifact.cost[start] / c) if np.isfinite(artifact.cost[start]) else None
            if cost_steps is None:
                results.append({"start_state": start, "success": False, "actual_steps": 0,
                                "graph_cost_steps": None, "failure_reason": "graph_unreachable",
                                "baseline_success": bool(base_result["success"]),
                                "baseline_actual_steps": int(base_result["actual_steps"]),
                                "baseline_graph_cost_steps": float(base_result["graph_cost_steps"])})
                continue
            payload = {"qpos": sim_state.qpos[start], "qvel": sim_state.qvel[start],
                       "button_states": sim_state.button_states[start]}
            observation = verifier.reset_to(payload)
            state = start
            actual_steps = 0
            success = False
            reason = None
            visited = 0
            planned_bridges = 0
            while not artifact.nodes.is_goal[state] and visited <= artifact.nodes.n_nodes:
                target = int(artifact.succ[state])
                edge_id = edge_lookup.get((state, target))
                if target < 0 or edge_id is None:
                    reason = "missing_successor_edge"
                    break
                if artifact.edges.kind[edge_id] == 0:
                    if state >= trans.n_rows:
                        reason = "temporal_action_missing"
                        break
                    action_chunk = np.asarray(trans.action[state], dtype=np.float32).reshape(1, 5)
                else:
                    planned_bridges += 1
                    plan = plan_map.get((state, target))
                    if plan is None:
                        reason = "bridge_action_plan_missing"
                        break
                    action_chunk = plan.chunks.reshape(-1, 5)
                for action in action_chunk:
                    observation, reward, terminated, truncated, info = verifier.step(action)
                    actual_steps += 1
                    if bool(info.get("success", False)) or reward > 0:
                        success = True
                        break
                    if terminated or truncated:
                        reason = "environment_terminated"
                        break
                if success or reason:
                    break
                if artifact.edges.kind[edge_id] == 1:
                    readout = verifier.sim_readout(observation, schema_name="ogbench_puzzle")
                    got = np.asarray(readout["r"], dtype=np.float64)
                    goal = data.state_raw[target, :data.n_cont].astype(np.float64)
                    landed = np.max(np.abs(got - goal) / data.epsilon_raw) <= 1.0
                    board_equal = np.array_equal(
                        np.asarray(readout["button_states"], dtype=np.int8),
                        data.state_bits[target].astype(np.int8),
                    )
                    if not (landed and board_equal):
                        reason = "bridge_landing_failed"
                        break
                state = target
                visited += 1
            if not success and reason is None:
                reason = "reached_goal_without_sim_success" if artifact.nodes.is_goal[state] else "path_limit"
            results.append({"start_state": start, "success": bool(success),
                            "actual_steps": int(actual_steps), "graph_cost_steps": cost_steps,
                            "failure_reason": reason,
                            "baseline_success": bool(base_result["success"]),
                            "baseline_actual_steps": int(base_result["actual_steps"]),
                            "baseline_graph_cost_steps": float(base_result["graph_cost_steps"]),
                            "planned_bridges": planned_bridges})
            if number % 25 == 0:
                print(f"v0.7 500-start path witness: {number}/500", file=sys.stderr, flush=True)
    finally:
        env.close()

    new_success = np.asarray([x["success"] for x in results], dtype=bool)
    base_success = np.asarray([x["baseline_success"] for x in results], dtype=bool)
    wins = int(np.count_nonzero(new_success & ~base_success))
    losses = int(np.count_nonzero(~new_success & base_success))
    p_success = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if wins + losses else 1.0
    common = new_success & base_success
    rho_new = rho_base = None
    p_rank = None
    rank_ci = None
    if np.count_nonzero(common) >= 3:
        ng = np.asarray([x["graph_cost_steps"] for x in results])[common]
        nt = np.asarray([x["actual_steps"] for x in results])[common]
        bg = np.asarray([x["baseline_graph_cost_steps"] for x in results])[common]
        bt = np.asarray([x["baseline_actual_steps"] for x in results])[common]
        rho_new = float(spearmanr(ng, nt).statistic)
        rho_base = float(spearmanr(bg, bt).statistic)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(2000):
            sample = rng.integers(0, len(ng), len(ng))
            r_new = spearmanr(ng[sample], nt[sample]).statistic
            r_base = spearmanr(bg[sample], bt[sample]).statistic
            if np.isfinite(r_new) and np.isfinite(r_base):
                diffs.append(float(r_new - r_base))
        if diffs:
            p_rank = float((1 + np.count_nonzero(np.asarray(diffs) <= 0.0)) / (len(diffs) + 1))
            rank_ci = np.quantile(diffs, [0.025, 0.975]).tolist()
    success_rate = float(np.mean(new_success))
    baseline_rate = float(np.mean(base_success))
    p_rate = float(binomtest(
        int(np.count_nonzero(new_success)), len(new_success), baseline_rate,
        alternative="greater",
    ).pvalue)
    accepted_report = []
    for item in accepted:
        row = dict(item)
        row["source_cost_initially_unreachable"] = not bool(
            np.isfinite(base.cost[int(item["source"])])
        )
        accepted_report.append(row)
    result = {
        "n": 500,
        "start_selection_sha256": digest,
        "baseline_start_selection_sha256": baseline.get("start_selection_sha256"),
        "success_rate": success_rate,
        "baseline_success_rate": baseline_rate,
        "paired_success_wins": wins,
        "paired_success_losses": losses,
        "success_one_sided_binomial_p_vs_baseline_rate": p_rate,
        "success_mcnemar_one_sided_p": p_success,
        "success_cost_time_spearman_common_success": rho_new,
        "baseline_cost_time_spearman_common_success": rho_base,
        "rank_difference_bootstrap_95ci": rank_ci,
        "rank_difference_bootstrap_one_sided_p": p_rank,
        "passed_q3_success_gate": bool(success_rate > baseline_rate and p_rate < 0.05),
        "outcomes": results,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-graph", type=Path, required=True)
    parser.add_argument("--reference-graph", type=Path,
                        help="optional G-R rho=0.9 graph for Q2 reachability comparison")
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="step_NNNNNN model checkpoint directory containing pi/ and f/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env", default="puzzle-4x4-singletask-task2-v0")
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--max-added-per-path", type=int, default=2)
    parser.add_argument("--max-candidates-per-round", type=int, default=2048)
    parser.add_argument("--insertion-mode", choices=("sequential", "bulk"), default="sequential")
    parser.add_argument("--plan-samples", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--witness-trials", type=int, default=200)
    parser.add_argument("--path-baseline", type=Path,
                        help="matched 500-start G-R rho=0.9 servo JSON for Q3 path comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-node-limit", type=int,
                        help="smoke-test cap; omit for full graph")
    parser.add_argument("--candidate-cache", type=Path,
                        help="optional dataset-bound NPZ cache shared by sequential and bulk builds")
    parser.add_argument("--skip-simulator", action="store_true")
    args = parser.parse_args()

    started = time.time()
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    base = load_graph(args.g0_graph, trans=trans)
    if base.meta["dataset_id"] != transition_meta["dataset_id"]:
        raise ValueError("G0 graph and transition dataset IDs differ")
    if np.any(base.edges.kind != 0):
        raise ValueError("v0.7 insertion must start from temporal-only G0")
    if base.nodes.n_nodes != len(data.state_raw):
        raise ValueError("model state table and G0 state nodes are not aligned")
    model_pair = load_model_pair(
        args.checkpoint, data.n_cont, data.n_bits,
        expected_dataset_id=transition_meta["dataset_id"],
    )
    bridge_evaluator = make_bridge_evaluator(model_pair[0], model_pair[1])
    reference = load_graph(args.reference_graph, trans=trans) if args.reference_graph else None
    if reference is not None and reference.meta["dataset_id"] != base.meta["dataset_id"]:
        raise ValueError("reference graph and G0 graph dataset IDs differ")

    episode, _ = state_episode_time(trans)
    features = np.concatenate((data.state_cont, data.state_bits), axis=1)
    pairs = load_or_build_candidate_pairs(
        features, episode, base.nodes.is_goal,
        dataset_id=transition_meta["dataset_id"], k=args.knn,
        cache_path=args.candidate_cache, node_limit=args.candidate_node_limit,
        progress=True,
    )
    if pairs.size:
        existing = np.column_stack((base.edges.src, base.edges.dst))
        if existing.size:
            pair_order = np.lexsort((existing[:, 1], existing[:, 0]))
            existing = existing[pair_order]
            candidate_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            candidates_sorted = pairs[candidate_order]
            key_dtype = np.dtype([("s", np.int64), ("t", np.int64)])
            ex_key = np.empty(len(existing), dtype=key_dtype)
            ex_key["s"], ex_key["t"] = existing[:, 0], existing[:, 1]
            can_key = np.empty(len(candidates_sorted), dtype=key_dtype)
            can_key["s"], can_key["t"] = candidates_sorted[:, 0], candidates_sorted[:, 1]
            pos = np.searchsorted(ex_key, can_key)
            in_bounds = pos < len(ex_key)
            same = np.zeros(len(pos), dtype=bool)
            same[in_bounds] = ex_key[pos[in_bounds]] == can_key[in_bounds]
            pairs = candidates_sorted[~same]

    c = step_cost(base.config.gamma)
    temporal = base.edges.kind == 0
    graph = IterativeBridgeGraph(
        n_nodes=base.nodes.n_nodes,
        goals=np.flatnonzero(base.nodes.is_goal),
        base_src=base.edges.src[temporal], base_dst=base.edges.dst[temporal],
        base_weight=base.weights[temporal] / c,
    )
    plan_map: dict[tuple[int, int], object] = {}
    validation_count = 0

    def validate_batch(batch):
        nonlocal validation_count
        plans = bridge_evaluator(
            features[batch[:, 0]], features[batch[:, 1]], batch[:, 0], batch[:, 1],
            epsilon_z=data.epsilon_z, state_std=data.state_std,
            delta_std=data.delta_std, n_cont=data.n_cont, n_bits=data.n_bits,
            h=5, max_chunks=args.max_chunks, n_samples=args.plan_samples,
            seed=args.seed + validation_count,
        )
        validation_count += len(batch)
        for pair, plan in zip(batch, plans, strict=True):
            if plan is not None:
                plan_map[(int(pair[0]), int(pair[1]))] = plan
        return plans

    if args.insertion_mode == "sequential":
        accepted = insert_validated_bridges(
            graph, pairs, validate_batch, min_edge_cost=5.0,
            delta=args.delta, b_max=args.max_added_per_path,
            max_candidates_per_round=args.max_candidates_per_round,
            lambda_cost=0.0, progress=True,
        )
    else:
        accepted = insert_validated_bridges_bulk(
            graph, pairs, validate_batch, min_edge_cost=5.0,
            delta=args.delta, b_max=args.max_added_per_path,
            max_candidates=args.max_candidates_per_round,
            lambda_cost=0.0, progress=True,
        )
    artifact = _artifact_with_bridges(base, graph, max_chunks=args.max_chunks)
    args.out.mkdir(parents=True, exist_ok=True)
    graph_path = args.out / "graph.npz"
    graph_hash = save_graph(artifact, graph_path)
    targets = critic_targets(
        trans, artifact, TargetConfig(h=5, unit="steps", kind="regret")
    )
    save_targets(targets, args.out / "targets")

    action_pieces, offsets = [], [0]
    for item in accepted:
        plan = plan_map[(int(item["source"]), int(item["target"]))]
        flat = plan.chunks.reshape(-1, 5).astype(np.float32)
        action_pieces.append(flat)
        offsets.append(offsets[-1] + len(flat))
        item["action_offset"] = int(offsets[-2])
        item["action_steps"] = int(len(flat))
    actions = np.concatenate(action_pieces, axis=0) if action_pieces else np.empty((0, 5), dtype=np.float32)
    np.savez_compressed(args.out / "accepted_actions.npz", actions=actions,
                         offsets=np.asarray(offsets, dtype=np.int64))

    witness = None
    if not args.skip_simulator:
        witness = _simulator_witness(
            accepted, plan_map, data, sim_state, env_id=args.env,
            trials=args.witness_trials, seed=args.seed,
        )
    path_witness = None
    if args.path_baseline is not None and not args.skip_simulator:
        path_witness = _path_witness(
            artifact, trans, sim_state, data, plan_map, args.path_baseline,
            env_id=args.env,
        )
    reachable_frac = float(np.mean(np.isfinite(graph.cost)))
    reference_reachable_frac = (
        float(np.mean(np.isfinite(reference.cost))) if reference is not None else None
    )
    per_source_counts = np.bincount(pairs[:, 0], minlength=base.nodes.n_nodes) if pairs.size else np.zeros(base.nodes.n_nodes, dtype=np.int64)
    queried_sources = per_source_counts[:base.nodes.n_nodes if args.candidate_node_limit is None
                                      else min(base.nodes.n_nodes, args.candidate_node_limit)]
    round_summary: dict[str, int] = {}
    for item in accepted:
        label = str(int(item["round"]))
        round_summary[label] = round_summary.get(label, 0) + 1
    bridge_count_values, bridge_count_frequencies = np.unique(graph.bridge_count, return_counts=True)
    result = {
        "spec": "Graph Critic v0.7",
        "dataset_id": transition_meta["dataset_id"],
        "g0_hash": base.content_hash,
        "graph_hash": graph_hash,
        "model_checkpoint": str(args.checkpoint),
        "model_data_schema": data.metadata["schema"],
        "epsilon_raw": data.epsilon_raw.tolist(),
        "epsilon_z": data.epsilon_z.tolist(),
        "candidate": {"metric": "Euclidean on z-normalized continuous coordinates concatenated with binary button coordinates",
                      "k": args.knn, "cross_episode_only": True,
                      "success_states_are_valid_destinations": True,
                      "success_destination_candidate_count": int(np.count_nonzero(base.nodes.is_goal[pairs[:, 1]])) if len(pairs) else 0,
                      "count": int(len(pairs)),
                      "smoke_source_limit": args.candidate_node_limit,
                      "source_count": int(np.count_nonzero(queried_sources)),
                      "queried_source_count": int(len(queried_sources)),
                      "source_degree_fraction_at_k": float(np.mean(queried_sources == args.knn)) if len(queried_sources) else 0.0,
                      "source_degree_min_all": int(np.min(queried_sources)) if len(queried_sources) else 0,
                      "source_degree_p50": float(np.median(queried_sources[queried_sources > 0])) if np.any(queried_sources) else 0.0,
                      "source_degree_min_positive": int(np.min(queried_sources[queried_sources > 0])) if np.any(queried_sources) else 0},
        "insertion": {"mode": args.insertion_mode,
                      "delta_steps": args.delta, "Lambda_steps": 0.0,
                      "B_max": args.max_added_per_path, "h": 5,
                      "max_chunks": args.max_chunks, "N": args.plan_samples,
                      "M_per_round": args.max_candidates_per_round,
                      "rounds_and_edges": accepted_report,
                      "accepted_by_round": round_summary,
                      "accepted_count": int(len(accepted)),
                      "validated_candidate_count": int(validation_count),
                      "acceptance_rate_among_validated": (
                          float(len(accepted) / validation_count) if validation_count else None
                      ),
                      "acceptance_rate_among_knn_pairs": (
                          float(len(accepted) / len(pairs)) if len(pairs) else None
                      ),
                      "success_destination_edges": int(sum(bool(base.nodes.is_goal[int(item["target"])]) for item in accepted)),
                      "unreachable_sources_initially_allowed": True},
        "cost_steps": {"finite_fraction": float(np.mean(np.isfinite(graph.cost))),
                       "reachable_nodes": int(np.count_nonzero(np.isfinite(graph.cost))),
                       "path_bridge_count_max": int(np.max(graph.bridge_count, initial=0)),
                       "path_bridge_count_distribution": {
                           str(int(value)): int(count)
                           for value, count in zip(bridge_count_values, bridge_count_frequencies, strict=True)
                       },
                       "cost_p50_finite": float(np.median(graph.cost[np.isfinite(graph.cost)])) if np.any(np.isfinite(graph.cost)) else None},
        "q2_reachability": {"graph_fraction": reachable_frac,
                            "reference_fraction": reference_reachable_frac,
                            "difference": (reachable_frac - reference_reachable_frac) if reference_reachable_frac is not None else None,
                            "reference_graph": str(args.reference_graph) if args.reference_graph else None},
        "simulator_q1": witness,
        "path_q3": path_witness,
        "elapsed_seconds": float(time.time() - started),
    }
    result = _json_safe(result)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "insertion"}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
