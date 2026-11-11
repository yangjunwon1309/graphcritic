"""Build or validate the exact v0.7/v0.8 cross-episode kNN cache."""

from __future__ import annotations

import argparse
import json
import hashlib
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
from graphcritic.bridges.object_state import pack_button_states
from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.state_graph import state_episode_time
from experiments.v08_runtime import provenance


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _robot_only_windows(windows: np.ndarray, *, frame_dim: int = 28,
                        robot_dim: int = 19) -> np.ndarray:
    """Select robot coordinates from each frame, preserving time-major order."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % frame_dim:
        raise ValueError("flattened windows do not match the Puzzle frame schema")
    shaped = values.reshape(len(values), values.shape[1] // frame_dim, frame_dim)
    return np.ascontiguousarray(shaped[:, :, :robot_dim].reshape(len(values), -1))


def select_object_candidate_pairs(
    features: np.ndarray,
    node_ids: np.ndarray,
    episodes: np.ndarray,
    board_codes: np.ndarray,
    source_mask: np.ndarray,
    allowed_changes: np.ndarray,
    *,
    knn: int = 32,
    knn_changed: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Small/reference implementation of grouped object-constrained kNN.

    ``features`` must already contain only robot coordinates. The production
    builder uses blocked GPU distances; this exact routine is also the simple
    executable contract used by tests and sampled parity checks.
    """
    x = np.asarray(features, dtype=np.float32)
    ids = np.asarray(node_ids, dtype=np.int64)
    ep = np.asarray(episodes, dtype=np.int64)
    codes = np.asarray(board_codes, dtype=np.uint64)
    sources = np.asarray(source_mask, dtype=np.bool_)
    allowed = np.asarray(allowed_changes, dtype=np.uint64)
    if x.ndim != 2 or not (len(x) == len(ids) == len(ep) == len(codes) == len(sources)):
        raise ValueError("candidate node arrays must align")
    if knn < 1 or knn_changed < 1:
        raise ValueError("both object-change k values must be positive")
    rows: list[tuple[int, int, int]] = []
    for src_local in np.flatnonzero(sources):
        delta = np.bitwise_xor(codes, codes[src_local])
        valid = (ep != ep[src_local]) & np.isin(delta, allowed)
        for changed, limit in ((False, knn), (True, knn_changed)):
            category = valid & ((delta != 0) if changed else (delta == 0))
            targets = np.flatnonzero(category)
            if not len(targets):
                continue
            diff = x[targets].astype(np.float64) - x[src_local].astype(np.float64)
            distance = np.sqrt(np.einsum("ij,ij->i", diff, diff, optimize=True))
            order = np.lexsort((ids[targets], distance))[:limit]
            rows.extend(
                (int(ids[src_local]), int(ids[targets[index]]), int(delta[targets[index]]))
                for index in order
            )
        # The changed flag is encoded by its nonzero XOR pattern.
    rows.sort(key=lambda item: (item[0], item[2] != 0, item[1]))
    if not rows:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.uint16)
    packed = np.asarray(rows, dtype=np.int64)
    return packed[:, :2].astype(np.int32), packed[:, 2].astype(np.uint16)


def _gpu_object_candidate_pairs(
    features: np.ndarray,
    node_ids: np.ndarray,
    episodes: np.ndarray,
    board_codes: np.ndarray,
    source_mask: np.ndarray,
    allowed_changes: np.ndarray,
    *,
    knn: int,
    knn_changed: int,
    batch_size: int,
    refine_extra: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Blocked GPU top-k for same-board and observed-changed-board groups."""
    import jax
    import jax.numpy as jnp

    try:
        device = jax.devices("gpu")[0]
    except (IndexError, RuntimeError) as exc:
        raise RuntimeError("object-change candidate building requires a JAX GPU") from exc
    from graphcritic.state_graph import _build_knn_index

    x = np.ascontiguousarray(features, dtype=np.float32)
    ids = np.asarray(node_ids, dtype=np.int64)
    ep = np.asarray(episodes, dtype=np.int32)
    codes = np.asarray(board_codes, dtype=np.uint64)
    source_rows = np.flatnonzero(source_mask).astype(np.int64)
    if not len(source_rows) or len(ids) < 2:
        raise ValueError("object-change kNN needs source and destination candidate nodes")
    index, backend = _build_knn_index(x, "jax_gpu")
    if backend != "jax_gpu":
        raise RuntimeError(f"expected jax_gpu backend, got {backend!r}")
    reference = index["reference"]
    reference_norm = index["reference_norm"]
    ref_ep = jax.device_put(ep, device)
    ref_codes = jax.device_put(codes.astype(np.int32), device)
    membership = np.zeros(1 << 20, dtype=np.bool_)
    if np.any(allowed_changes >= len(membership)):
        raise ValueError("object change pattern exceeds 20 buttons")
    membership[np.asarray(allowed_changes, dtype=np.int64)] = True
    membership_device = jax.device_put(membership, device)
    query_count_same = min(len(ids), knn + max(0, int(refine_extra)))
    query_count_changed = min(len(ids), knn_changed + max(0, int(refine_extra)))

    @jax.jit
    def topk(query_device, query_ep, query_codes):
        query_norm = jnp.sum(query_device * query_device, axis=1, keepdims=True)
        squared = jnp.maximum(
            query_norm + reference_norm[None, :]
            - 2.0 * jnp.matmul(query_device, reference.T, precision=jax.lax.Precision.HIGHEST),
            0.0,
        )
        xor = jnp.bitwise_xor(query_codes[:, None], ref_codes[None, :])
        common = (query_ep[:, None] != ref_ep[None, :]) & membership_device[xor]
        same = common & (xor == 0)
        changed = common & (xor != 0)
        same_scores = jnp.where(same, -squared, -jnp.inf)
        changed_scores = jnp.where(changed, -squared, -jnp.inf)
        same_ids = jax.lax.top_k(same_scores, query_count_same)[1]
        changed_ids = jax.lax.top_k(changed_scores, query_count_changed)[1]
        return same_ids, changed_ids

    result_src: list[np.ndarray] = []
    result_dst: list[np.ndarray] = []
    result_delta: list[np.ndarray] = []
    for start in range(0, len(source_rows), batch_size):
        rows = source_rows[start : start + batch_size]
        same_ids, changed_ids = topk(
            jax.device_put(x[rows], device),
            jax.device_put(ep[rows], device),
            jax.device_put(codes[rows].astype(np.int32), device),
        )
        category_ids = [np.asarray(same_ids, dtype=np.int64), np.asarray(changed_ids, dtype=np.int64)]
        category_limits = [knn, knn_changed]
        for category_index, (candidate_ids, limit) in enumerate(
            zip(category_ids, category_limits, strict=True)
        ):
            valid = (candidate_ids >= 0) & (candidate_ids < len(ids))
            safe = np.clip(candidate_ids, 0, len(ids) - 1)
            diffs = np.bitwise_xor(codes[rows, None], codes[safe])
            valid &= ep[safe] != ep[rows, None]
            valid &= np.isin(diffs, allowed_changes)
            valid &= (diffs == 0) if category_index == 0 else (diffs != 0)
            delta = x[rows, None, :].astype(np.float64) - x[safe].astype(np.float64)
            distances = np.sqrt(np.sum(delta * delta, axis=2))
            distances[~valid] = np.inf
            order = np.argsort(distances, axis=1, kind="stable")[:, :limit]
            nearest = np.take_along_axis(candidate_ids, order, axis=1)
            nearest_dist = np.take_along_axis(distances, order, axis=1)
            keep = np.isfinite(nearest_dist)
            src_grid = np.broadcast_to(rows[:, None], nearest.shape)
            src_take, dst_take = src_grid[keep], nearest[keep]
            delta_take = np.bitwise_xor(codes[src_take], codes[np.clip(dst_take, 0, len(ids) - 1)])
            # Same and changed groups are processed in fixed order; the top-k
            # output itself is category-specific, so assert the group contract.
            if category_index == 0:
                if np.any(delta_take != 0):
                    raise RuntimeError("same-object GPU neighborhood returned a changed board")
            result_src.append(ids[src_take].astype(np.int32))
            result_dst.append(ids[dst_take].astype(np.int32))
            result_delta.append(delta_take.astype(np.uint16))
        if start == 0 or (start + len(rows)) % (batch_size * 100) == 0 or start + len(rows) == len(source_rows):
            print(f"object-change GPU kNN: sources={start + len(rows)}/{len(source_rows)}", file=sys.stderr, flush=True)
    src = np.concatenate(result_src) if result_src else np.empty(0, dtype=np.int32)
    dst = np.concatenate(result_dst) if result_dst else np.empty(0, dtype=np.int32)
    delta = np.concatenate(result_delta) if result_delta else np.empty(0, dtype=np.uint16)
    order = np.lexsort((dst, delta != 0, src))
    return np.column_stack((src[order], dst[order])).astype(np.int32), delta[order]


def _verify_object_knn_parity(
    features: np.ndarray,
    node_ids: np.ndarray,
    episodes: np.ndarray,
    board_codes: np.ndarray,
    source_mask: np.ndarray,
    allowed_changes: np.ndarray,
    pairs: np.ndarray,
    pair_delta: np.ndarray,
    *,
    sample_size: int,
    knn: int,
    knn_changed: int,
    seed: int,
) -> dict[str, object]:
    """Compare sampled constrained GPU neighbors with an exact cKDTree query."""
    from scipy.spatial import cKDTree

    x = np.asarray(features, dtype=np.float32)
    ids = np.asarray(node_ids, dtype=np.int64)
    ep = np.asarray(episodes, dtype=np.int64)
    codes = np.asarray(board_codes, dtype=np.uint64)
    source_ids = np.flatnonzero(source_mask)
    if sample_size <= 0:
        return {"sampled_sources": 0, "compared_groups": 0, "mismatched_groups": 0,
                "ties_excluded": 0, "passed": True}
    if sample_size > len(source_ids):
        raise ValueError(f"parity-size {sample_size} exceeds {len(source_ids)} boundary sources")
    sampled = np.sort(np.random.default_rng(seed).choice(source_ids, size=sample_size, replace=False))
    pair_lookup: dict[tuple[int, int], np.ndarray] = {}
    for index, source_node in enumerate(np.asarray(pairs)[:, 0]):
        pair_lookup.setdefault((int(source_node), int(pair_delta[index] != 0)), []).append(
            int(pairs[index, 1])
        )
    pair_lookup = {key: np.asarray(value, dtype=np.int64) for key, value in pair_lookup.items()}
    allowed = np.asarray(allowed_changes, dtype=np.uint64)
    allowed_nonzero = allowed[allowed != 0]
    compared = mismatched = ties_excluded = 0
    for source_code in np.unique(codes[sampled]):
        rows = sampled[codes[sampled] == source_code]
        target_specs = []
        if np.any(allowed == 0):
            target_specs.append((0, codes == source_code, knn))
        if len(allowed_nonzero):
            targets_codes = np.bitwise_xor(source_code, allowed_nonzero)
            target_specs.append((1, np.isin(codes, targets_codes), knn_changed))
        for changed, target_mask, limit in target_specs:
            target_local = np.flatnonzero(target_mask)
            if not len(target_local):
                for local in rows:
                    if len(pair_lookup.get((int(ids[local]), changed), ())):
                        mismatched += 1
                continue
            tree = cKDTree(x[target_local])
            missing = rows.copy()
            query_count = min(len(target_local), max(limit + 64, limit + 1))
            expected: dict[int, np.ndarray] = {}
            boundary_ties: set[int] = set()
            while len(missing):
                dists, local_neighbors = tree.query(
                    x[missing], k=query_count, workers=-1,
                )
                dists = np.asarray(dists).reshape(len(missing), query_count)
                local_neighbors = np.asarray(local_neighbors, dtype=np.int64).reshape(
                    len(missing), query_count
                )
                global_neighbors = target_local[np.minimum(local_neighbors, len(target_local) - 1)]
                next_missing = []
                for pos, local in enumerate(missing):
                    valid = ep[global_neighbors[pos]] != ep[local]
                    candidates = global_neighbors[pos, valid]
                    if len(candidates) < limit and query_count < len(target_local):
                        next_missing.append(int(local))
                        continue
                    exact_delta = x[candidates].astype(np.float64) - x[local].astype(np.float64)
                    exact_dist = np.sqrt(np.einsum("ij,ij->i", exact_delta, exact_delta))
                    order = np.lexsort((ids[candidates], exact_dist))
                    chosen = candidates[order[:limit]]
                    if len(order) > limit:
                        d0, d1 = exact_dist[order[limit - 1]], exact_dist[order[limit]]
                        if abs(float(d0 - d1)) <= 1e-12 * max(1.0, float(d1)):
                            boundary_ties.add(int(local))
                    expected[int(local)] = ids[chosen]
                if not next_missing or query_count == len(target_local):
                    break
                missing = np.asarray(next_missing, dtype=np.int64)
                query_count = min(len(target_local), max(query_count + 1, query_count * 2))
            for local in rows:
                if int(local) in boundary_ties:
                    ties_excluded += 1
                    continue
                expected_nodes = expected.get(int(local), np.empty(0, dtype=np.int64))
                actual_nodes = pair_lookup.get((int(ids[local]), changed), np.empty(0, dtype=np.int64))
                # Compare ranked rows as well as the neighbor set; distances
                # are stable except for explicitly excluded boundary ties.
                if not np.array_equal(np.sort(expected_nodes), np.sort(actual_nodes)):
                    mismatched += 1
                compared += 1
    return {
        "sampled_sources": int(len(sampled)), "compared_groups": int(compared),
        "mismatched_groups": int(mismatched), "ties_excluded": int(ties_excluded),
        "passed": bool(mismatched == 0),
    }


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
    parser.add_argument("--object-change-set", type=Path,
                        help="restrict candidates to changes observed in the dataset")
    parser.add_argument("--feature-mode", choices=("all", "robot"), default="all",
                        help="robot uses only continuous robot coordinates in window kNN")
    parser.add_argument("--knn-changed", type=int, default=32,
                        help="neighbors per source for nonzero object changes")
    args = parser.parse_args(argv)
    if args.knn < 1 or args.parity_size < 0 or args.batch_size < 1:
        parser.error("knn and batch-size must be positive; parity-size must be non-negative")
    if args.window < 1:
        parser.error("window must be positive")
    if args.knn_changed < 1:
        parser.error("knn-changed must be positive")
    if args.object_change_set is not None:
        if args.window != 5 or args.feature_mode != "robot":
            parser.error("object-change mode requires --window 5 --feature-mode robot")
        if args.knn != 32 or args.knn_changed != 32:
            parser.error("v1.1 object-change mode fixes --knn and --knn-changed at 32")

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
    if args.object_change_set is not None:
        change_report = json.loads(args.object_change_set.read_text(encoding="utf-8"))
        if change_report.get("dataset_id") != transition_meta["dataset_id"]:
            raise ValueError("object-change set dataset ID does not match transitions")
        if int(change_report.get("button_count", -1)) != model_data.state_bits.shape[1]:
            raise ValueError("object-change set button count does not match the sidecar")
        patterns = change_report.get("patterns")
        if not isinstance(patterns, list):
            raise ValueError("object-change set must contain a patterns list")
        allowed_changes = np.asarray([int(item["code"]) for item in patterns], dtype=np.uint64)
        if len(np.unique(allowed_changes)) != len(allowed_changes) or not len(allowed_changes):
            raise ValueError("object-change set must contain unique non-empty patterns")
        if not np.any(allowed_changes == 0):
            raise ValueError("object-change set must retain the zero-change pattern")
        robot_candidate_features = _robot_only_windows(candidate_features)
        board_codes = pack_button_states(model_data.state_bits[node_ids])
        candidate_pairs, pair_delta = _gpu_object_candidate_pairs(
            robot_candidate_features, node_ids, candidate_episode, board_codes, source_mask,
            allowed_changes, knn=args.knn, knn_changed=args.knn_changed,
            batch_size=args.batch_size,
        )
        parity = _verify_object_knn_parity(
            robot_candidate_features, node_ids, candidate_episode, board_codes, source_mask,
            allowed_changes, candidate_pairs, pair_delta,
            sample_size=args.parity_size, knn=args.knn, knn_changed=args.knn_changed,
            seed=args.seed,
        )
        if not parity["passed"]:
            raise RuntimeError(f"object-change GPU/cKDTree parity failed: {parity}")
        change_set_sha = _sha256_file(args.object_change_set)
        rule = "cross_episode_observed_object_change_v1_robot_window5_same32_changed32"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out, pairs=candidate_pairs, pair_delta=pair_delta,
            pair_changed=(pair_delta != 0), dataset_id=np.asarray(transition_meta["dataset_id"]),
            k=np.asarray(args.knn, dtype=np.int64), knn_changed=np.asarray(args.knn_changed, dtype=np.int64),
            window=np.asarray(args.window, dtype=np.int64), candidate_rule=np.asarray(rule),
            object_change_set_sha256=np.asarray(change_set_sha),
            allowed_change_codes=allowed_changes, node_ids=node_ids.astype(np.int32),
            node_ids_sha256=np.asarray(_array_sha256(node_ids)),
            feature_sha256=np.asarray(_array_sha256(robot_candidate_features)),
            episode_sha256=np.asarray(_array_sha256(candidate_episode)),
            goal_sha256=np.asarray(_array_sha256(candidate_goals)),
        )
        cache_report = {
            **provenance(ROOT), "spec_version": "v1.1",
            "dataset_id": transition_meta["dataset_id"], "candidate_rule": rule,
            "object_change_set_sha256": change_set_sha,
            "window": int(args.window), "k_same": int(args.knn),
            "k_changed": int(args.knn_changed), "feature_mode": "robot",
            "n_nodes": int(graph.nodes.n_nodes), "n_candidate_nodes": int(len(node_ids)),
            "n_boundary_sources": int(np.count_nonzero(source_mask)),
            "n_pairs": int(len(candidate_pairs)),
            "n_unchanged_pairs": int(np.count_nonzero(pair_delta == 0)),
            "n_changed_pairs": int(np.count_nonzero(pair_delta != 0)),
            "feature_sha256": _array_sha256(robot_candidate_features),
            "episode_sha256": _array_sha256(candidate_episode),
            "goal_sha256": _array_sha256(candidate_goals),
            "node_ids_sha256": _array_sha256(node_ids),
            "knn_parity": parity, "cache_path": str(args.out),
        }
        args.out.with_suffix(".json").write_text(
            json.dumps(cache_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(cache_report, indent=2, sort_keys=True))
        return 0
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
