"""v1.8 cube-double object-threshold bridges and task-specific targets.

The candidate cache is task-shared.  Graph vertices are five-frame decision
states, task goals, and episode terminal states; the per-row target writer
projects every transition endpoint onto the next graph vertex along its
recorded episode.  This keeps exact QC/DQC row alignment without expanding a
five-frame graph into one million decision vertices.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from graphcritic.bridges.insert import layered_reverse_dijkstra, used_bridge_mask_for_budget
from graphcritic.bridges.object_state import object_distance
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.config import TargetConfig
from graphcritic.envs.ogbench_cube import (
    archive_dataset_id,
    cube_double_goals,
    make_cube_transition_set,
    split_observation,
)
from graphcritic.featurize import state_table
from graphcritic.io import save_targets
from graphcritic.types import TargetSet


ENV_BASE = "cube-double-play-v0"
WINDOW = 5
K_NEIGHBORS = 32
MAX_DISTANCE_STEPS = 20.0
BRIDGE_LAMBDA_STEPS = 10.5
DELTA_ROT_DEG = 15.0
DELTA_POS_CANDIDATES_M = (0.005, 0.01, 0.02, 0.03, 0.04)


def _next_state_indices(terminal: np.ndarray) -> np.ndarray:
    terminal = np.asarray(terminal, dtype=bool)
    terminal_rows = np.flatnonzero(terminal)
    result = np.empty(len(terminal), dtype=np.int64)
    nonterminal_rows = np.flatnonzero(~terminal)
    result[nonterminal_rows] = nonterminal_rows + 1
    result[terminal_rows] = len(terminal) + np.arange(len(terminal_rows), dtype=np.int64)
    return result


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        required = {"observations", "actions", "terminals"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"cube-double archive is missing required fields: {sorted(missing)}")
        # Standard OGBench archives can also contain qpos/qvel and other
        # unused fields. Avoid decompressing and copying those large arrays.
        return {key: np.array(archive[key], copy=True) for key in required}


def load_cube_context(dataset_path: str | Path) -> dict[str, Any]:
    """Load one standard archive and derive task-independent aligned states."""
    path = Path(dataset_path).expanduser().resolve()
    archive = _load_archive(path)
    observations = archive["observations"]
    actions = archive["actions"]
    terminals = archive["terminals"].astype(bool)
    goals = cube_double_goals()
    transitions = make_cube_transition_set(
        observations, actions, terminals, goals=goals[1]
    )
    states, references, _, _ = state_table(transitions)
    robot, objects = split_observation(states)
    terminal_rows = np.flatnonzero(transitions.terminal)
    state_episode = np.concatenate((transitions.episode_id, transitions.episode_id[terminal_rows]))
    state_time = np.concatenate((transitions.t, transitions.t[terminal_rows] + 1))
    state_success_by_task: dict[int, np.ndarray] = {}
    position = objects[:, :, :3]
    for task, task_goals in goals.items():
        state_success_by_task[task] = np.all(
            np.linalg.norm(position - task_goals[None, :, :], axis=2) <= 0.04,
            axis=1,
        )
    boundary = state_time % WINDOW == WINDOW - 1
    goal_union = np.logical_or.reduce(tuple(state_success_by_task.values()))
    candidate_node_ids = np.flatnonzero(boundary | goal_union).astype(np.int64)
    terminal_state_ids = transitions.n_rows + np.arange(len(terminal_rows), dtype=np.int64)
    graph_node_state_ids = np.unique(np.concatenate((candidate_node_ids, terminal_state_ids)))
    order = np.lexsort((graph_node_state_ids, state_time[graph_node_state_ids], state_episode[graph_node_state_ids]))
    graph_node_state_ids = graph_node_state_ids[order]
    return {
        "path": path,
        "dataset_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_id_base": archive_dataset_id(path, ENV_BASE),
        "observations": observations,
        "actions": actions,
        "raw_terminals": terminals,
        "transitions": transitions,
        "states": states,
        "references": references,
        "robot": robot,
        "objects": objects,
        "grasp_state": (observations[:, 18] > 0.5) | np.any(objects[:, :, 2] > 0.04, axis=1),
        "goals": goals,
        "state_episode": state_episode.astype(np.int64),
        "state_time": state_time.astype(np.int64),
        "state_success_by_task": state_success_by_task,
        "goal_union": goal_union,
        "boundary": boundary,
        "candidate_node_ids": candidate_node_ids,
        "graph_node_state_ids": graph_node_state_ids,
        "terminal_state_ids": terminal_state_ids,
        "terminal_state_index": _next_state_indices(transitions.terminal),
    }


def _robot_windows(ctx: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    robot = np.asarray(ctx["robot"], dtype=np.float32)
    if mean.shape != (19,) or std.shape != (19,) or np.any(std <= 0):
        raise ValueError("robot quasimetric checkpoint must record 19D state mean/std")
    normalized = ((robot.astype(np.float64) - mean) / std).astype(np.float32)
    return puzzle_state_windows(
        normalized,
        ctx["transitions"],
        ctx["terminal_state_index"],
        window=WINDOW,
    )


def build_cube_candidate_cache(
    dataset_path: str | Path,
    checkpoint: str | Path,
    out: str | Path,
    *,
    delta_pos: float,
    delta_rot_deg: float = DELTA_ROT_DEG,
    knn: int = K_NEIGHBORS,
    source_limit: int | None = None,
) -> dict[str, Any]:
    """Create cross-episode object-matched robot-window kNN candidates."""
    if not math.isfinite(delta_pos) or delta_pos < 0:
        raise ValueError("delta_pos must be finite and non-negative, in meters")
    if not math.isfinite(delta_rot_deg) or delta_rot_deg < 0:
        raise ValueError("delta_rot_deg must be finite and non-negative, in degrees")
    if knn != K_NEIGHBORS:
        raise ValueError("v1.8 fixes robot feature kNN to 32")
    checkpoint_path = Path(checkpoint)
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("input_part") != "robot" or int(metadata.get("frame_dim", -1)) != 19:
        raise ValueError("v1.8 candidate kNN requires a robot-only 19D frame checkpoint")
    if int(metadata.get("window", -1)) != WINDOW or int(metadata.get("constraint_horizon", -1)) != WINDOW:
        raise ValueError("v1.8 candidate cache requires a five-frame, five-step checkpoint")
    ctx = load_cube_context(dataset_path)
    if metadata.get("dataset_id") != ctx["dataset_id_base"]:
        raise ValueError("robot checkpoint was trained on a different cube-double archive")
    windows = _robot_windows(
        ctx, np.asarray(metadata["state_mean"], dtype=np.float32),
        np.asarray(metadata["state_std"], dtype=np.float32),
    )
    node_ids = ctx["candidate_node_ids"]
    source_ids = np.flatnonzero(ctx["boundary"]).astype(np.int64)
    if source_limit is not None:
        if source_limit < 1:
            raise ValueError("source_limit must be positive")
        source_ids = source_ids[:source_limit]
    node_objects = ctx["objects"][node_ids]
    node_episode = ctx["state_episode"][node_ids]
    node_positions = node_objects[:, :, :3].reshape(len(node_ids), 6)
    tree = cKDTree(node_positions)
    local_of_state = np.full(len(ctx["states"]), -1, dtype=np.int64)
    local_of_state[node_ids] = np.arange(len(node_ids), dtype=np.int64)
    source_local = local_of_state[source_ids]
    if np.any(source_local < 0):
        raise RuntimeError("every chunk boundary must be in the candidate destination table")
    radius = math.sqrt(2.0) * float(delta_pos) + 1e-12
    angle_limit = math.radians(float(delta_rot_deg))
    pair_src: list[np.ndarray] = []
    pair_dst: list[np.ndarray] = []
    pair_pos: list[np.ndarray] = []
    pair_rot: list[np.ndarray] = []
    position_tolerance_pairs = 0
    rotation_pairs_checked = 0
    # Layouts repeat heavily. Bound query_ball_point's Python result size even
    # at the largest δ candidate.
    query_batch = 32
    robot_features = windows[node_ids]
    for batch_start in range(0, len(source_ids), query_batch):
        batch_src_ids = source_ids[batch_start : batch_start + query_batch]
        batch_src_local = source_local[batch_start : batch_start + query_batch]
        neighbor_lists = tree.query_ball_point(
            node_positions[batch_src_local], radius, workers=1,
        )
        for src_id, src_local, candidates in zip(
            batch_src_ids, batch_src_local, neighbor_lists, strict=True
        ):
            candidate_local = np.asarray(candidates, dtype=np.int64)
            if not len(candidate_local):
                continue
            candidate_local = candidate_local[node_episode[candidate_local] != ctx["state_episode"][src_id]]
            if not len(candidate_local):
                continue
            source_xyz = ctx["objects"][src_id, :, :3]
            candidate_xyz = node_objects[candidate_local, :, :3]
            candidate_dpos = np.max(np.linalg.norm(candidate_xyz - source_xyz[None, :, :], axis=2), axis=1)
            position_local = candidate_local[candidate_dpos <= delta_pos]
            position_dpos = candidate_dpos[candidate_dpos <= delta_pos]
            if not len(position_local):
                continue
            position_tolerance_pairs += len(position_local)
            diff = robot_features[position_local].astype(np.float64) - robot_features[src_local].astype(np.float64)
            feature_dist = np.sqrt(np.einsum("ij,ij->i", diff, diff, optimize=True))
            order = np.lexsort((node_ids[position_local], feature_dist))
            ranked_local = position_local[order]
            ranked_dpos = position_dpos[order]
            chosen_local: list[np.ndarray] = []
            chosen_dpos: list[np.ndarray] = []
            chosen_drot: list[np.ndarray] = []
            chosen_count = 0
            # Position filtering is cheap and broad. Check the more expensive
            # 24-way rotational quotient in robot-distance order, stopping as
            # soon as the exact nearest 32 eligible neighbors are known.
            for offset in range(0, len(ranked_local), 256):
                local = ranked_local[offset : offset + 256]
                drot = np.asarray(object_distance(
                    np.broadcast_to(ctx["objects"][src_id], (len(local), 2, 7)),
                    node_objects[local],
                )[1])
                rotation_pairs_checked += len(local)
                good = drot <= angle_limit
                if np.any(good):
                    chosen_local.append(local[good])
                    chosen_dpos.append(ranked_dpos[offset : offset + len(local)][good])
                    chosen_drot.append(drot[good])
                    chosen_count += int(np.count_nonzero(good))
                if chosen_count >= knn:
                    break
            if not chosen_local:
                continue
            valid_local = np.concatenate(chosen_local)[:knn]
            valid_pos = np.concatenate(chosen_dpos)[:knn]
            valid_rot = np.concatenate(chosen_drot)[:knn]
            order = np.arange(len(valid_local), dtype=np.int64)
            pair_src.append(np.full(len(order), int(src_id), dtype=np.int64))
            pair_dst.append(node_ids[valid_local[order]].astype(np.int64))
            pair_pos.append(valid_pos[order].astype(np.float32))
            pair_rot.append(valid_rot[order].astype(np.float32))
        done = batch_start + len(batch_src_ids)
        if done == len(source_ids) or done % 20_000 == 0:
            print(
                f"cube object kNN δp={delta_pos * 100:.1f}cm: sources={done}/{len(source_ids)}",
                file=sys.stderr, flush=True,
            )
    if pair_src:
        src = np.concatenate(pair_src)
        dst = np.concatenate(pair_dst)
        dpos = np.concatenate(pair_pos)
        drot = np.concatenate(pair_rot)
        order = np.lexsort((dst, src))
        pairs = np.column_stack((src[order], dst[order])).astype(np.int32)
        dpos, drot = dpos[order], drot[order]
    else:
        pairs = np.empty((0, 2), dtype=np.int32)
        dpos = drot = np.empty(0, dtype=np.float32)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pairs=pairs,
        object_distance_pos_m=dpos,
        object_distance_rot_rad=drot,
        node_ids=node_ids.astype(np.int32),
        dataset_hash=np.asarray(ctx["dataset_hash"]),
        dataset_id_base=np.asarray(ctx["dataset_id_base"]),
        delta_pos_m=np.asarray(delta_pos, dtype=np.float64),
        delta_rot_rad=np.asarray(angle_limit, dtype=np.float64),
        window=np.asarray(WINDOW, dtype=np.int64),
        knn=np.asarray(knn, dtype=np.int64),
    )
    report = {
        "spec_version": "v1.8",
        "dataset_id_base": ctx["dataset_id_base"],
        "dataset_hash": ctx["dataset_hash"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": str(metadata.get("params_sha256")),
        "delta_pos_m": float(delta_pos),
        "delta_rot_deg": float(delta_rot_deg),
        "window": WINDOW,
        "knn": knn,
        "n_graph_candidates": int(len(node_ids)),
        "n_boundary_sources": int(len(source_ids)),
        "n_position_tolerance_pairs_before_rotation_and_knn": int(position_tolerance_pairs),
        "n_rotation_pairs_checked_in_robot_rank_order": int(rotation_pairs_checked),
        "n_candidate_pairs": int(len(pairs)),
        "candidate_pair_hash": _sha256_array(pairs),
        "node_ids_hash": _sha256_array(node_ids),
        "candidate_cache": str(out_path.resolve()),
    }
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def score_cube_candidate_cache(
    dataset_path: str | Path,
    checkpoint: str | Path,
    candidate_cache: str | Path,
    out: str | Path,
    *,
    pair_batch_size: int = 4096,
) -> dict[str, Any]:
    """Evaluate dθ on every cached pair using per-node IQE embedding reuse."""
    from experiments.v08_runtime import (
        load_quasimetric_checkpoint,
        make_embedding_cache_predictor,
    )

    ctx = load_cube_context(dataset_path)
    cache_path = Path(candidate_cache)
    with np.load(cache_path, allow_pickle=False) as cache:
        pairs = np.asarray(cache["pairs"], dtype=np.int64)
        node_ids = np.asarray(cache["node_ids"], dtype=np.int64)
        candidate_hash = str(cache["dataset_hash"].item())
    if candidate_hash != ctx["dataset_hash"]:
        raise ValueError("cube candidate cache and dataset archive hashes differ")
    checkpoint_path = Path(checkpoint)
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("input_part") != "robot" or int(metadata.get("window", -1)) != WINDOW:
        raise ValueError("dθ checkpoint must be the v1.8 robot-only window-5 model")
    windows = _robot_windows(
        ctx, np.asarray(metadata["state_mean"], dtype=np.float32),
        np.asarray(metadata["state_std"], dtype=np.float32),
    )
    model, params, metadata = load_quasimetric_checkpoint(
        checkpoint_path,
        input_dim=WINDOW * 19,
        expected_dataset_id=ctx["dataset_id_base"],
    )
    embedding_path = Path(out).parent / "embeddings" / (
        f"cube-{metadata['params_sha256'][:16]}-{_sha256_array(node_ids)[:16]}.npy"
    )
    predict_pairs, embedding_report = make_embedding_cache_predictor(
        model,
        params,
        windows[node_ids],
        node_ids,
        params_sha256=str(metadata["params_sha256"]),
        dataset_id=ctx["dataset_id_base"],
        cache_path=embedding_path,
        pair_batch_size=pair_batch_size,
    )
    distances = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), pair_batch_size):
        end = min(start + pair_batch_size, len(pairs))
        distances[start:end] = predict_pairs(pairs[start:end])
        if end == len(pairs) or end % (pair_batch_size * 100) == 0:
            print(f"cube dθ judge: {end}/{len(pairs)}", file=sys.stderr, flush=True)
    cache_out = Path(out)
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_out,
        pairs=pairs.astype(np.int32),
        distance_steps=distances,
        accept_dtheta=(distances <= MAX_DISTANCE_STEPS),
        candidate_pair_hash=np.asarray(_sha256_array(pairs)),
        checkpoint_sha256=np.asarray(metadata["params_sha256"]),
        dataset_hash=np.asarray(ctx["dataset_hash"]),
    )
    report = {
        "spec_version": "v1.8",
        "candidate_cache": str(cache_path.resolve()),
        "candidate_pair_hash": _sha256_array(pairs),
        "checkpoint_sha256": str(metadata["params_sha256"]),
        "dataset_hash": ctx["dataset_hash"],
        "n_candidate_pairs": int(len(pairs)),
        "n_dtheta_accepted": int(np.count_nonzero(distances <= MAX_DISTANCE_STEPS)),
        "dtheta_p50": float(np.median(distances)) if len(distances) else None,
        "dtheta_p90": float(np.quantile(distances, 0.9)) if len(distances) else None,
        "embedding_report": embedding_report,
        "score_cache": str(cache_out.resolve()),
    }
    cache_out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _task_graph_inputs(ctx: dict[str, Any], task: int):
    state_to_node = np.full(len(ctx["states"]), -1, dtype=np.int64)
    state_to_node[ctx["graph_node_state_ids"]] = np.arange(len(ctx["graph_node_state_ids"]), dtype=np.int64)
    node_states = ctx["graph_node_state_ids"]
    node_episode = ctx["state_episode"][node_states]
    node_time = ctx["state_time"][node_states]
    node_goals = ctx["state_success_by_task"][task][node_states]
    base_src: list[int] = []
    base_dst: list[int] = []
    base_weight: list[int] = []
    starts = np.r_[0, np.flatnonzero(node_episode[1:] != node_episode[:-1]) + 1]
    ends = np.r_[starts[1:], len(node_states)]
    for start, end in zip(starts, ends, strict=True):
        # Goal nodes remain vertices but have no outgoing continuation in the
        # target graph. The incoming temporal edge is still retained.
        full = np.arange(int(start), int(end), dtype=np.int64)
        if len(full) < 2:
            continue
        for first, second in zip(full[:-1], full[1:], strict=True):
            if node_goals[first]:
                continue
            gap = int(node_time[second] - node_time[first])
            if gap <= 0:
                raise ValueError("cube graph temporal node times must strictly increase per episode")
            base_src.append(int(first))
            base_dst.append(int(second))
            base_weight.append(gap)
    return (
        state_to_node,
        node_states,
        node_episode,
        node_time,
        node_goals,
        np.asarray(base_src, dtype=np.int64),
        np.asarray(base_dst, dtype=np.int64),
        np.asarray(base_weight, dtype=np.float64),
    )


def _base_costs(n: int, goals: np.ndarray, src: np.ndarray, dst: np.ndarray, weight: np.ndarray) -> np.ndarray:
    reverse = csr_matrix((weight, (dst, src)), shape=(n, n))
    costs = np.asarray(dijkstra(reverse, directed=True, indices=goals, min_only=True), dtype=np.float64)
    costs[goals] = 0.0
    return costs


def budget_cost_sweep(
    n_nodes: int,
    goals: np.ndarray,
    base_src: np.ndarray,
    base_dst: np.ndarray,
    base_weight: np.ndarray,
    bridge_src: np.ndarray,
    bridge_dst: np.ndarray,
    bridge_weight: np.ndarray,
    *,
    max_budget: int = 32,
) -> np.ndarray:
    """Exact C_b recurrence, equivalent to layered reverse Dijkstra.

    Each layer is a multi-source reverse Dijkstra over only recorded data
    edges. Bridge sources are seeded with ``w_bridge + C_(b-1)(dst)`` and the
    previous-layer cost is also a seed, enforcing an at-most-b budget.
    """
    n = int(n_nodes)
    reverse = csr_matrix((base_weight, (base_dst, base_src)), shape=(n, n))
    current = np.full(n, np.inf, dtype=np.float64)
    if len(goals):
        current = np.asarray(dijkstra(reverse, directed=True, indices=goals, min_only=True), dtype=np.float64)
        current[goals] = 0.0
    all_costs = [current]
    bsrc = np.asarray(bridge_src, dtype=np.int64)
    bdst = np.asarray(bridge_dst, dtype=np.int64)
    bweight = np.asarray(bridge_weight, dtype=np.float64)
    for _budget in range(1, int(max_budget) + 1):
        seeds = current.copy()
        finite_target = np.isfinite(current[bdst])
        if np.any(finite_target):
            proposed = bweight[finite_target] + current[bdst[finite_target]]
            np.minimum.at(seeds, bsrc[finite_target], proposed)
        finite_seed = np.flatnonzero(np.isfinite(seeds))
        super_node = n
        augmented = csr_matrix(
            (
                np.concatenate((base_weight, seeds[finite_seed])),
                (
                    np.concatenate((base_dst, np.full(len(finite_seed), super_node, dtype=np.int64))),
                    np.concatenate((base_src, finite_seed)),
                ),
            ),
            shape=(n + 1, n + 1),
        )
        current = np.asarray(dijkstra(augmented, directed=True, indices=super_node), dtype=np.float64)[:n]
        current[np.asarray(goals, dtype=np.int64)] = 0.0
        all_costs.append(current)
    return np.stack(all_costs, axis=0)


def _lift_node_costs(ctx: dict[str, Any], task: int, node_costs: np.ndarray) -> np.ndarray:
    """Recover exact per-state costs between graph decision vertices."""
    node_states = ctx["graph_node_state_ids"]
    node_ep = ctx["state_episode"][node_states]
    state_t = ctx["state_time"]
    successes = ctx["state_success_by_task"][task]
    result = np.full(len(ctx["states"]), np.inf, dtype=np.float64)
    result[successes] = 0.0
    starts = np.r_[0, np.flatnonzero(node_ep[1:] != node_ep[:-1]) + 1]
    ends = np.r_[starts[1:], len(node_states)]
    trans_ep = ctx["transitions"].episode_id
    transition_starts = np.r_[0, np.flatnonzero(trans_ep[1:] != trans_ep[:-1]) + 1]
    transition_ends = np.r_[transition_starts[1:], ctx["transitions"].n_rows]
    for (start, end), (row_start, row_end) in zip(
        zip(starts, ends, strict=True),
        zip(transition_starts, transition_ends, strict=True),
        strict=True,
    ):
        # state_table stores each episode's transition states contiguously,
        # followed by its terminal next-state occurrence in the side table.
        state_ids = np.concatenate((
            np.arange(int(row_start), int(row_end), dtype=np.int64),
            np.asarray([ctx["terminal_state_index"][int(row_end) - 1]], dtype=np.int64),
        ))
        graph_states = node_states[start:end]
        graph_times = state_t[graph_states]
        positions = np.searchsorted(graph_times, state_t[state_ids], side="left")
        has_endpoint = positions < len(graph_states)
        keep = has_endpoint & ~successes[state_ids]
        covered = state_ids[keep]
        positions = positions[keep]
        endpoint_ids = graph_states[positions]
        offset = state_t[endpoint_ids] - state_t[covered]
        base_cost = node_costs[start + positions]
        finite = np.isfinite(base_cost)
        result[covered[finite]] = offset[finite] + base_cost[finite]
    return result


def _target_set(
    ctx: dict[str, Any], task: int, state_cost: np.ndarray, graph_hash: str,
    dataset_id: str, *, kind: str, h: int = 5, gamma: float = 0.99,
) -> TargetSet:
    trans = ctx["transitions"]
    n = trans.n_rows
    valid = np.zeros(n, dtype=bool)
    loss_mask = np.zeros(n, dtype=bool)
    source = np.full(n, 3, dtype=np.uint8)
    n_starts = max(0, n - h + 1)
    values = np.zeros(n, dtype=np.float64)
    if n_starts:
        prefix = np.r_[0, np.cumsum(trans.terminal, dtype=np.int64)]
        starts = np.arange(n_starts, dtype=np.int64)
        valid[:n_starts] = prefix[starts + h - 1] - prefix[starts] == 0
        success_windows = np.stack([trans.success[offset : offset + n_starts] for offset in range(h)], axis=1)
        has_success = np.any(success_windows, axis=1)
        first_success = np.argmax(success_windows, axis=1).astype(np.int64) + 1
        endpoint_states = starts + h
        # Index h-step endpoints with the same complete-chunk rule as QC and
        # DQC, including the terminal-only endpoint after a final action.
        terminal_state_index = ctx["terminal_state_index"]
        episode_starts = np.r_[0, np.flatnonzero(trans.episode_id[1:] != trans.episode_id[:-1]) + 1]
        episode_ends = np.r_[episode_starts[1:] - 1, n - 1]
        ends = np.empty(n, dtype=np.int64)
        for start, end in zip(episode_starts, episode_ends, strict=True):
            ends[int(start) : int(end) + 1] = int(end)
        end_rows = ends[starts]
        endpoint_ids = endpoint_states.copy()
        at_terminal_endpoint = (endpoint_states == end_rows + 1) & valid[:n_starts]
        if np.any((endpoint_states > end_rows + 1) & valid[:n_starts]):
            raise ValueError("valid chunk endpoint crossed an episode boundary")
        endpoint_ids[at_terminal_endpoint] = terminal_state_index[end_rows[at_terminal_endpoint]]
        endpoint_cost = state_cost[endpoint_ids]
        chunk_success = has_success | trans.success_cur[:n_starts]
        if kind == "discounted":
            values[:n_starts] = (1.0 - np.power(gamma, h + endpoint_cost)) / (1.0 - gamma)
            values[:n_starts][~np.isfinite(endpoint_cost)] = 1.0 / (1.0 - gamma)
            values[:n_starts][has_success] = (1.0 - np.power(gamma, first_success[has_success])) / (1.0 - gamma)
            values[:n_starts][trans.success_cur[:n_starts]] = 0.0
        elif kind == "gamma_pow":
            values[:n_starts] = np.power(gamma, h + endpoint_cost)
            values[:n_starts][~np.isfinite(endpoint_cost)] = 0.0
            values[:n_starts][has_success] = np.power(gamma, first_success[has_success])
            values[:n_starts][trans.success_cur[:n_starts]] = 1.0
        else:
            raise ValueError("cube v1.8 target kind must be 'discounted' or 'gamma_pow'")
        values[:n_starts][~valid[:n_starts]] = 0.0
        source[:n_starts][valid[:n_starts] & chunk_success] = 0
        source[:n_starts][valid[:n_starts] & ~chunk_success & np.isfinite(endpoint_cost)] = 1
        source[:n_starts][valid[:n_starts] & ~chunk_success & ~np.isfinite(endpoint_cost)] = 2
        loss_mask[:n_starts] = valid[:n_starts]
    return TargetSet(
        y=values.astype(np.float32),
        valid=valid,
        h=h,
        unit="steps",
        graph_hash=graph_hash,
        dataset_id=dataset_id,
        source=source,
        kind=kind,
        loss_mask=loss_mask,
        discount_gamma=gamma,
    )


def _graph_hash(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _summarize_cost(cost: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(cost)[np.isfinite(cost)]
    return {
        "reachable_fraction": float(len(finite) / len(cost)) if len(cost) else 0.0,
        "n_reachable": int(len(finite)),
        "p50": float(np.quantile(finite, 0.5)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
    }


def task_budget_sweep(
    ctx: dict[str, Any], task: int, pairs_state: np.ndarray, distances: np.ndarray,
    *, max_budget: int = 32,
) -> dict[str, Any]:
    inputs = _task_graph_inputs(ctx, task)
    state_to_node, node_states, node_ep, node_time, goals_mask, base_src, base_dst, base_weight = inputs
    pairs = np.asarray(pairs_state, dtype=np.int64)
    dtheta = np.asarray(distances, dtype=np.float32)
    if pairs.shape != (len(dtheta), 2):
        raise ValueError("cube candidate pair scores do not align")
    local_src = state_to_node[pairs[:, 0]]
    local_dst = state_to_node[pairs[:, 1]]
    if np.any(local_src < 0) or np.any(local_dst < 0):
        raise ValueError("candidate endpoint is absent from the task graph node set")
    c0 = _base_costs(len(node_states), np.flatnonzero(goals_mask), base_src, base_dst, base_weight)
    target_finite = np.isfinite(c0[local_dst])
    gain = c0[local_src] - c0[local_dst]
    optimistic = target_finite & (gain >= 10.0)
    accepted = optimistic & (dtheta <= MAX_DISTANCE_STEPS)
    bsrc = local_src[accepted]
    bdst = local_dst[accepted]
    bweight = dtheta[accepted].astype(np.float64) + BRIDGE_LAMBDA_STEPS
    costs = budget_cost_sweep(
        len(node_states), np.flatnonzero(goals_mask), base_src, base_dst, base_weight,
        bsrc, bdst, bweight, max_budget=max_budget,
    )
    budget_rows = []
    for budget, node_cost in enumerate(costs):
        summary = _summarize_cost(node_cost)
        budget_rows.append({"b": budget, **summary})
    sat = float(budget_rows[-1]["reachable_fraction"])
    b_star = next(
        (int(row["b"]) for row in budget_rows if float(row["reachable_fraction"]) >= sat - 0.005),
        max_budget,
    )
    final = costs[b_star]
    reduced = np.isfinite(c0) & np.isfinite(final) & (final < c0 - 1e-9)
    reductions = c0[reduced] - final[reduced]
    return {
        "task": task,
        "n_candidate_pairs": int(len(pairs)),
        "n_optimistic_candidates": int(np.count_nonzero(optimistic)),
        "n_accepted_bridges": int(np.count_nonzero(accepted)),
        "accepted_candidate_indices": np.flatnonzero(accepted),
        "accepted_bridge_src": bsrc,
        "accepted_bridge_dst": bdst,
        "accepted_bridge_weight_steps": bweight,
        "accepted_bridge_distance_steps": dtheta[accepted],
        "accepted_pair_pos_m": None,
        "accepted_pair_rot_rad": None,
        "c0_node": c0,
        "cost_by_budget": costs,
        "budget_sweep": budget_rows,
        "r_sat": sat,
        "b_star": b_star,
        "c_reduced_node_count": int(np.count_nonzero(reduced)),
        "c_reduced_node_fraction_of_all": float(np.mean(reduced)) if len(reduced) else 0.0,
        "c_reduction_p50_steps": float(np.quantile(reductions, 0.5)) if len(reductions) else None,
        "c_reduction_p90_steps": float(np.quantile(reductions, 0.9)) if len(reductions) else None,
        "newly_reachable_node_count": int(np.count_nonzero(~np.isfinite(c0) & np.isfinite(final))),
        "node_states": node_states,
        "node_episode": node_ep,
        "node_time": node_time,
        "node_goals": goals_mask,
        "state_to_node": state_to_node,
        "base_src": base_src,
        "base_dst": base_dst,
        "base_weight_steps": base_weight,
    }


def build_cube_judged_graph(
    dataset_path: str | Path,
    candidate_cache: str | Path,
    score_cache: str | Path,
    out: str | Path,
    *,
    task: int,
    b_max: int,
    target_kind: str = "discounted",
    gamma: float = 0.99,
    delta_pos: float,
    delta_rot_deg: float = DELTA_ROT_DEG,
    bridge_lambda_steps: float = BRIDGE_LAMBDA_STEPS,
    max_budget_for_sweep: int = 32,
) -> dict[str, Any]:
    """Build one task's pruned graph and QC/DQC target artifact."""
    if task not in range(1, 6) or b_max < 0 or b_max > max_budget_for_sweep:
        raise ValueError("task must be 1..5 and b_max within the configured sweep")
    if bridge_lambda_steps != BRIDGE_LAMBDA_STEPS:
        raise ValueError("v1.8 fixes bridge lambda to 10.5 steps")
    ctx = load_cube_context(dataset_path)
    with np.load(candidate_cache, allow_pickle=False) as cache:
        candidate_pairs = np.asarray(cache["pairs"], dtype=np.int64)
        cached_dataset_hash = str(cache["dataset_hash"].item())
        cached_delta_pos = float(cache["delta_pos_m"].item())
        cached_delta_rot = float(cache["delta_rot_rad"].item())
        cached_node_ids = np.asarray(cache["node_ids"], dtype=np.int64)
        pair_pos = np.asarray(cache["object_distance_pos_m"], dtype=np.float32)
        pair_rot = np.asarray(cache["object_distance_rot_rad"], dtype=np.float32)
    with np.load(score_cache, allow_pickle=False) as cache:
        score_pairs = np.asarray(cache["pairs"], dtype=np.int64)
        distance = np.asarray(cache["distance_steps"], dtype=np.float32)
        score_hash = str(cache["dataset_hash"].item())
    if cached_dataset_hash != ctx["dataset_hash"] or score_hash != ctx["dataset_hash"]:
        raise ValueError("candidate/score cache dataset hash mismatch")
    if not np.array_equal(candidate_pairs, score_pairs) or pair_pos.shape != distance.shape or pair_rot.shape != distance.shape:
        raise ValueError("candidate and dθ score caches are not aligned")
    if not np.array_equal(cached_node_ids, ctx["candidate_node_ids"]):
        raise ValueError("candidate node table does not match cube state ordering")
    if not math.isclose(cached_delta_pos, delta_pos) or not math.isclose(cached_delta_rot, math.radians(delta_rot_deg)):
        raise ValueError("requested δ differs from candidate cache tolerances")
    sweep = task_budget_sweep(ctx, task, candidate_pairs, distance, max_budget=max_budget_for_sweep)
    inputs = _task_graph_inputs(ctx, task)
    state_to_node, node_states, node_ep, node_time, node_goals, base_src, base_dst, base_weight = inputs
    if b_max > sweep["cost_by_budget"].shape[0] - 1:
        raise ValueError("requested b_max exceeds the cached sweep horizon")
    accepted_idxs = sweep["accepted_candidate_indices"]
    accepted_src = sweep["accepted_bridge_src"]
    accepted_dst = sweep["accepted_bridge_dst"]
    accepted_weight = sweep["accepted_bridge_weight_steps"]
    # Match the recorded decision for this task and budget; only edges selected
    # by a shortest path in some layer survive the final graph.
    final_paths = layered_reverse_dijkstra(
        len(node_states), np.flatnonzero(node_goals), base_src, base_dst, base_weight,
        accepted_src, accepted_dst, accepted_weight, b_max=b_max,
    )
    retained_mask = used_bridge_mask_for_budget(
        final_paths, accepted_src, accepted_dst, n_nodes=len(node_states), budget=b_max,
    )
    retained_idxs = accepted_idxs[retained_mask]
    retained_src = accepted_src[retained_mask]
    retained_dst = accepted_dst[retained_mask]
    retained_weight = accepted_weight[retained_mask]
    final_node_cost = final_paths.cost_steps[b_max]
    successor_layer = final_paths.successor[b_max]
    successor = np.where(successor_layer >= 0, successor_layer % len(node_states), -1).astype(np.int64)
    final_state_cost = _lift_node_costs(ctx, task, final_node_cost)
    task_env = f"cube-double-play-singletask-task{task}-v0"
    dataset_id = archive_dataset_id(dataset_path, task_env)
    graph_arrays = {
        "node_state_ids": node_states.astype(np.int32),
        "node_episode_id": node_ep.astype(np.int32),
        "node_time": node_time.astype(np.int32),
        "node_is_goal": node_goals.astype(bool),
        "temporal_src": base_src.astype(np.int32),
        "temporal_dst": base_dst.astype(np.int32),
        "temporal_weight_steps": base_weight.astype(np.float32),
        "bridge_src": retained_src.astype(np.int32),
        "bridge_dst": retained_dst.astype(np.int32),
        "bridge_distance_steps": distance[retained_idxs].astype(np.float32),
        "bridge_cost_steps": retained_weight.astype(np.float32),
        "cost_steps": final_node_cost.astype(np.float32),
        "successor": successor.astype(np.int32),
    }
    graph_meta = {
        "schema_version": "cube-graph-v1.8",
        "spec_version": "v1.8",
        "dataset_id": dataset_id,
        "dataset_hash": ctx["dataset_hash"],
        "task": task,
        "object_match": "cube-indexed-position-and-cube-symmetry-rotation-threshold",
        "delta_pos_m": float(delta_pos),
        "delta_rot_deg": float(delta_rot_deg),
        "judge": "robot-window-5 IQE d_theta",
        "dtheta_max_steps": MAX_DISTANCE_STEPS,
        "bridge_lambda_steps": float(bridge_lambda_steps),
        "b_max": int(b_max),
        "source_candidate_cache_hash": _sha256_array(candidate_pairs),
        "source_score_checkpoint_sha256": str(json.loads((Path(score_cache).with_suffix('.json')).read_text()).get("checkpoint_sha256", "")),
        "n_graph_nodes": int(len(node_states)),
        "n_temporal_edges": int(len(base_src)),
        "n_candidate_pairs": int(len(candidate_pairs)),
        "n_optimistic_candidates": int(sweep["n_optimistic_candidates"]),
        "n_accepted_bridges": int(sweep["n_accepted_bridges"]),
        "n_retained_bridges": int(len(retained_src)),
        "candidate_delta_pos_m": cached_delta_pos,
        "candidate_delta_rot_rad": cached_delta_rot,
    }
    graph_hash = _graph_hash(graph_meta, graph_arrays)
    graph_meta["graph_hash"] = graph_hash
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path / "graph.npz", **graph_arrays)
    (out_path / "graph.json").write_text(json.dumps(graph_meta, indent=2, sort_keys=True) + "\n")
    target_kinds = ("discounted", "gamma_pow")
    targets_by_kind: dict[str, TargetSet] = {}
    target_hashes: dict[str, str] = {}
    for kind in target_kinds:
        target = _target_set(
            ctx, task, final_state_cost, graph_hash, dataset_id,
            kind=kind, h=5, gamma=gamma,
        )
        filename = "targets.npz" if kind == target_kind else f"targets_{kind}.npz"
        target_hashes[kind] = save_targets(
            target, out_path / filename,
            target_provenance={"kind": "learned", "spec_version": "v1.8", "graph_hash": graph_hash},
        )
        targets_by_kind[kind] = target
    target = targets_by_kind[target_kind]
    target_hash = target_hashes[target_kind]
    target_stats = target.y[target.loss_mask]
    result = {
        **graph_meta,
        "target_kind": target_kind,
        "target_gamma": gamma,
        "target_hash": target_hash,
        "target_hashes": target_hashes,
        "target_artifacts": {
            "discounted": "targets.npz" if target_kind == "discounted" else "targets_discounted.npz",
            "gamma_pow": "targets.npz" if target_kind == "gamma_pow" else "targets_gamma_pow.npz",
        },
        "target_valid_fraction": float(np.mean(target.valid)),
        "target_loss_mask_fraction": float(np.mean(target.loss_mask)),
        "target_capped_loss_mask_fraction": float(np.mean(target.source[target.loss_mask] == 2)),
        "target_mean": float(np.mean(target_stats)) if len(target_stats) else None,
        "target_p50": float(np.quantile(target_stats, 0.5)) if len(target_stats) else None,
        "target_p90": float(np.quantile(target_stats, 0.9)) if len(target_stats) else None,
        "target_max": float(np.max(target_stats)) if len(target_stats) else None,
        "r_sat": sweep["r_sat"],
        "b_star": sweep["b_star"],
        "graph_reachable_fraction": _summarize_cost(final_state_cost)["reachable_fraction"],
        "graph_node_reachable_fraction": _summarize_cost(final_node_cost)["reachable_fraction"],
        "c_reduced_node_count": sweep["c_reduced_node_count"],
        "c_reduced_node_fraction_of_all": sweep["c_reduced_node_fraction_of_all"],
        "c_reduction_p50_steps": sweep["c_reduction_p50_steps"],
        "c_reduction_p90_steps": sweep["c_reduction_p90_steps"],
        "newly_reachable_node_count": sweep["newly_reachable_node_count"],
        "graph_bridge_count_distribution": {
            str(count): float(np.mean(final_paths.bridge_count[b_max, np.isfinite(final_paths.cost_steps[b_max])] == count))
            if np.any(np.isfinite(final_paths.cost_steps[b_max])) else 0.0
            for count in range(b_max + 1)
        },
        "accepted_object_distance_pos_p50_cm": float(np.median(pair_pos[accepted_idxs]) * 100) if len(accepted_idxs) else None,
        "accepted_object_distance_pos_p90_cm": float(np.quantile(pair_pos[accepted_idxs], 0.9) * 100) if len(accepted_idxs) else None,
        "accepted_object_distance_rot_p50_deg": float(np.median(pair_rot[accepted_idxs]) * 180 / math.pi) if len(accepted_idxs) else None,
        "accepted_object_distance_rot_p90_deg": float(np.quantile(pair_rot[accepted_idxs], 0.9) * 180 / math.pi) if len(accepted_idxs) else None,
        "retained_object_distance_pos_violation_count": int(np.count_nonzero(pair_pos[retained_idxs] > delta_pos)),
        "accepted_grasp_mismatch_fraction": compute_grasp_mismatch(ctx, candidate_pairs[accepted_idxs]),
        "accepted_grasp_mismatch_count": int(round(
            compute_grasp_mismatch(ctx, candidate_pairs[accepted_idxs]) * len(accepted_idxs)
        )),
        "target_rows": int(len(target.y)),
    }
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        out_path / "sweep.npz",
        cost_by_budget=sweep["cost_by_budget"].astype(np.float32),
        accepted_candidate_indices=accepted_idxs.astype(np.int64),
        retained_candidate_indices=retained_idxs.astype(np.int64),
        b_star=np.asarray(sweep["b_star"], dtype=np.int64),
    )
    return result


def compute_grasp_mismatch(ctx: dict[str, Any], pair_indices: np.ndarray) -> float:
    """Fraction of accepted bridges disagreeing on contact/elevated grasp state."""
    pairs = np.asarray(pair_indices, dtype=np.int64)
    if not len(pairs):
        return 0.0
    grasped = np.asarray(ctx.get("grasp_state"))
    if grasped.shape != (len(ctx["states"]),):
        observations = ctx["states"]
        _, objects = split_observation(observations)
        contact = observations[:, 18] > 0.5
        elevated = np.any(objects[:, :, 2] > 0.04, axis=1)
        grasped = contact | elevated
    return float(np.mean(grasped[pairs[:, 0]] != grasped[pairs[:, 1]]))


def summarize_delta_sweep(
    ctx: dict[str, Any], candidate_paths: dict[float, Path], score_paths: dict[float, Path],
    *, max_budget: int = 32,
) -> tuple[dict[str, Any], float]:
    """Run task-specific C sweeps and select the strictest δp within 0.5 pp."""
    table: dict[str, Any] = {"spec_version": "v1.8", "rows": {}, "tasks": {}}
    for delta_pos in DELTA_POS_CANDIDATES_M:
        with np.load(candidate_paths[delta_pos], allow_pickle=False) as cand:
            pairs = np.asarray(cand["pairs"], dtype=np.int64)
        with np.load(score_paths[delta_pos], allow_pickle=False) as scores:
            scored_pairs = np.asarray(scores["pairs"], dtype=np.int64)
            distances = np.asarray(scores["distance_steps"], dtype=np.float32)
        if not np.array_equal(pairs, scored_pairs):
            raise ValueError("candidate and score arrays differ in δ sweep")
        task_sweeps = {}
        for task in range(1, 6):
            task_sweeps[task] = task_budget_sweep(ctx, task, pairs, distances, max_budget=max_budget)
        mean_reachability = float(np.mean([task_sweeps[t]["r_sat"] for t in range(1, 6)]))
        table["rows"][f"{delta_pos:.3f}"] = {
            "delta_pos_m": delta_pos,
            "candidate_pairs": int(len(pairs)),
            "mean_accepted_bridges_over_tasks": float(np.mean([task_sweeps[t]["n_accepted_bridges"] for t in range(1, 6)])),
            "mean_reachability_b32": mean_reachability,
            "per_task": {
                str(task): {
                    "n_optimistic_candidates": task_sweeps[task]["n_optimistic_candidates"],
                    "n_accepted_bridges": task_sweeps[task]["n_accepted_bridges"],
                    "b_star": task_sweeps[task]["b_star"],
                    "r_sat": task_sweeps[task]["r_sat"],
                    "budget_sweep": task_sweeps[task]["budget_sweep"],
                }
                for task in range(1, 6)
            },
        }
        del task_sweeps
    max_delta = max(DELTA_POS_CANDIDATES_M)
    reference_reach = float(table["rows"][f"{max_delta:.3f}"]["mean_reachability_b32"])
    selected = next(
        delta for delta in DELTA_POS_CANDIDATES_M
        if float(table["rows"][f"{delta:.3f}"]["mean_reachability_b32"]) >= reference_reach - 0.005
    )
    table["selected_delta_pos_m"] = float(selected)
    table["selection_reference_reachability_b32"] = reference_reach
    table["selection_tolerance_fraction"] = 0.005
    return table, float(selected)


def write_cube_graph_report(out: str | Path, result: dict[str, Any]) -> None:
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
