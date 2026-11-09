"""Build model-validated v0.7 bridges with all-pairs judging and witness them."""

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
from scipy.stats import binomtest, spearmanr

from graphcritic.bridges.check import BridgePlan
from graphcritic.bridges.insert import layered_reverse_dijkstra
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_window_candidate_nodes
from graphcritic.config import TargetConfig
from graphcritic.featurize import state_table
from graphcritic.io import load_graph, load_sim_state, load_transitions, save_graph, save_targets
from graphcritic.modes import state_episode_time
from graphcritic.path import PathStep
from graphcritic.physical_time import estimate_realized_speed
from graphcritic.servo import ServoTrackingPolicy
from graphcritic.state_schema import OGBenchPuzzleStateSchema
from graphcritic.targets import critic_targets
from graphcritic.types import EdgeSet, GraphArtifact
from graphcritic.verification import OGBenchVerifier
from graphcritic.weights import step_cost
from experiments.v07_runtime import load_model_pair, make_bridge_evaluator

CANDIDATE_RULE = "cross_episode_incl_goal_v2"


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


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _gpu_cross_episode_topk(
    features: np.ndarray,
    episode_id: np.ndarray,
    *,
    k: int,
    batch_size: int,
    refine_extra: int = 64,
    progress: bool = False,
) -> np.ndarray:
    """Exact cross-episode neighbors using blocked JAX GPU distance matrices.

    Same-episode distances are masked before top-k, so the returned k are the
    nearest *eligible* states. A small extra shortlist is re-ranked in float64
    on CPU to remove float32 dot-product cancellation at the top-k boundary.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - bench environment only
        raise RuntimeError("GPU kNN requires JAX in the bench environment") from exc
    try:
        device = jax.devices("gpu")[0]
    except (IndexError, RuntimeError) as exc:  # pragma: no cover - hardware dependent
        raise RuntimeError("GPU kNN requires an available JAX GPU") from exc

    x = np.ascontiguousarray(features, dtype=np.float32)
    ep = np.asarray(episode_id, dtype=np.int32)
    n, dim = x.shape
    if n < 2 or not 1 <= k < n or batch_size < 1:
        raise ValueError("GPU kNN requires 1 <= k < n and a positive batch size")

    # Reuse state_graph's backend/index setup (GPU selection and device-resident
    # reference matrix) rather than maintaining a second device policy.
    from graphcritic.state_graph import _build_knn_index

    index, backend = _build_knn_index(x, "jax_gpu")
    if backend != "jax_gpu":
        raise RuntimeError(f"expected jax_gpu backend, got {backend!r}")
    ref_ep = jax.device_put(ep, device)
    query_count = min(n, k + max(0, int(refine_extra)))
    reference = index["reference"]
    reference_norm = index["reference_norm"]

    @jax.jit
    def topk(query_device, query_ep_device):
        query_norm = jnp.sum(query_device * query_device, axis=1, keepdims=True)
        squared = jnp.maximum(
            query_norm + reference_norm[None, :]
            - 2.0 * jnp.matmul(
                query_device, reference.T, precision=jax.lax.Precision.HIGHEST
            ),
            0.0,
        )
        eligible = query_ep_device[:, None] != ref_ep[None, :]
        squared = jnp.where(eligible, squared, jnp.inf)
        negative, ids = jax.lax.top_k(-squared, query_count)
        return -negative, ids

    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        rows = np.arange(start, end, dtype=np.int64)
        query_device = jax.device_put(x[start:end], device)
        query_ep_device = jax.device_put(ep[start:end], device)
        _, candidate_ids = topk(query_device, query_ep_device)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        valid = (candidate_ids >= 0) & (candidate_ids < n)
        valid &= ep[np.maximum(candidate_ids, 0)] != ep[rows, None]

        # Recompute shortlisted distances in float64 and stably order them.
        safe_ids = np.maximum(candidate_ids, 0)
        delta = x[start:end, None, :].astype(np.float64) - x[safe_ids].astype(np.float64)
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        distances[~valid] = np.inf
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        nearest = np.take_along_axis(candidate_ids, order, axis=1)
        nearest_dist = np.take_along_axis(distances, order, axis=1)
        keep = np.isfinite(nearest_dist)
        row_grid = np.broadcast_to(rows[:, None], nearest.shape)
        source_parts.append(row_grid[keep].astype(np.int32, copy=False))
        target_parts.append(nearest[keep].astype(np.int32, copy=False))
        if progress and (end % (batch_size * 20) == 0 or end == n):
            count = sum(part.size for part in source_parts)
            print(
                f"GPU cross-episode kNN: states={end}/{n}, pairs={count}",
                file=sys.stderr, flush=True,
            )
    if not source_parts or not any(part.size for part in source_parts):
        return np.empty((0, 2), dtype=np.int64)
    # Rows are emitted in source order already. Keep each row's distance rank
    # order and avoid a 32M-pair global sort and its multi-gigabyte temporary.
    return np.column_stack((np.concatenate(source_parts), np.concatenate(target_parts)))


def _cross_episode_knn_pairs_ckdtree_reference(
    features: np.ndarray,
    episode_id: np.ndarray,
    *,
    k: int,
    initial_query: int = 128,
    block_size: int = 4096,
) -> np.ndarray:
    """CPU exact-kNN reference used only by the M3 parity experiment/tests."""
    from scipy.spatial import cKDTree

    x = np.ascontiguousarray(features, dtype=np.float32)
    ep = np.asarray(episode_id, dtype=np.int64)
    if x.ndim != 2 or ep.shape != (len(x),):
        raise ValueError("features and episode IDs must align")
    n = len(x)
    if not 1 <= k < n:
        raise ValueError("k must be in [1, n)")
    tree = cKDTree(x)
    result = np.full((n, k), -1, dtype=np.int32)
    for start in range(0, n, block_size):
        rows = np.arange(start, min(n, start + block_size), dtype=np.int64)
        query_k = min(n, max(k + 1, int(initial_query)))
        missing = np.arange(len(rows), dtype=np.int64)
        while missing.size:
            _, ids = tree.query(x[rows[missing]], k=query_k, workers=-1)
            ids = np.asarray(ids, dtype=np.int64).reshape(len(missing), query_k)
            valid = ep[ids] != ep[rows[missing], None]
            rank = np.cumsum(valid, axis=1)
            enough = valid.sum(axis=1) >= k
            enough_rows = np.flatnonzero(enough)
            if enough_rows.size:
                keep = valid[enough_rows] & (rank[enough_rows] <= k)
                out_rows, out_cols = np.nonzero(keep)
                result[rows[missing[enough_rows[out_rows]]], rank[enough_rows[out_rows], out_cols] - 1] = ids[enough_rows[out_rows], out_cols]
            missing = missing[~enough]
            if not missing.size or query_k == n:
                break
            query_k = min(n, max(query_k + 1, query_k * 2))
        # Rows with fewer than k eligible states retain their actual neighbors.
    keep = result >= 0
    sources = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], result.shape)
    pairs = np.column_stack((sources[keep], result[keep]))
    order = np.lexsort((pairs[:, 1], pairs[:, 0])) if len(pairs) else np.empty(0, dtype=np.int64)
    return pairs[order].astype(np.int64, copy=False)


def verify_gpu_knn_against_ckdtree(
    features: np.ndarray,
    episode_id: np.ndarray,
    *,
    sample_size: int = 50_000,
    k: int = 32,
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, object]:
    """Compare GPU and cKDTree neighbor sets on one deterministic subset."""
    x = np.asarray(features, dtype=np.float32)
    ep = np.asarray(episode_id, dtype=np.int64)
    if x.ndim != 2 or ep.shape != (len(x),):
        raise ValueError("features and episode IDs must align")
    if sample_size > len(x) or sample_size <= k:
        raise ValueError("sample_size must be greater than k and no larger than the dataset")
    ids = np.sort(np.random.default_rng(seed).choice(len(x), size=sample_size, replace=False))
    subset, subset_ep = x[ids], ep[ids]
    gpu_pairs = _gpu_cross_episode_topk(
        subset, subset_ep, k=k + 1, batch_size=batch_size, progress=True,
    )
    cpu_pairs = _cross_episode_knn_pairs_ckdtree_reference(
        subset, subset_ep, k=k + 1,
    )

    def group(pairs: np.ndarray) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        if len(pairs):
            rows, counts = np.unique(pairs[:, 0], return_counts=True)
            offset = 0
            for row, count in zip(rows, counts, strict=True):
                result[int(row)] = pairs[offset:offset + int(count), 1]
                offset += int(count)
        return result

    gpu_group, cpu_group = group(gpu_pairs), group(cpu_pairs)
    matched = compared = ties_excluded = mismatch = 0
    for row in range(sample_size):
        gpu_ids = gpu_group.get(row, np.empty(0, dtype=np.int64))
        cpu_ids = cpu_group.get(row, np.empty(0, dtype=np.int64))
        if len(gpu_ids) < k or len(cpu_ids) < k:
            continue
        gpu_dist = np.linalg.norm(
            subset[gpu_ids].astype(np.float64) - subset[row].astype(np.float64), axis=1
        )
        cpu_dist = np.linalg.norm(
            subset[cpu_ids].astype(np.float64) - subset[row].astype(np.float64), axis=1
        )
        gpu_ranked = gpu_ids[np.argsort(gpu_dist, kind="stable")]
        cpu_order = np.argsort(cpu_dist, kind="stable")
        cpu_ranked = cpu_ids[cpu_order]
        tie = False
        if len(cpu_ranked) > k:
            d_k, d_next = cpu_dist[cpu_order[k - 1]], cpu_dist[cpu_order[k]]
            tie = bool(abs(float(d_k - d_next)) <= 1e-12 * max(1.0, float(d_next)))
        if tie:
            ties_excluded += 1
            continue
        compared += 1
        if np.array_equal(np.sort(gpu_ranked[:k]), np.sort(cpu_ranked[:k])):
            matched += 1
        else:
            mismatch += 1
    return {
        "backend": "jax_gpu_blocked_exact",
        "reference": "scipy_cKDTree_exact",
        "sample_size": int(sample_size),
        "k": int(k),
        "seed": int(seed),
        "rows_compared_excluding_boundary_ties": int(compared),
        "boundary_tie_rows_excluded": int(ties_excluded),
        "matched_rows": int(matched),
        "mismatched_rows": int(mismatch),
        "match_fraction": float(matched / compared) if compared else None,
    }


def cross_episode_knn_pairs(
    state_features: np.ndarray,
    episode_id: np.ndarray,
    is_goal: np.ndarray,
    *,
    k: int = 32,
    node_limit: int | None = None,
    batch_size: int = 256,
    progress: bool = False,
) -> np.ndarray:
    """Return exact GPU kNN pairs from different episodes, including goals."""
    x = np.asarray(state_features, dtype=np.float32)
    ep = np.asarray(episode_id, dtype=np.int64)
    goals = np.asarray(is_goal, dtype=np.bool_)
    if x.ndim != 2 or ep.shape != (len(x),) or goals.shape != ep.shape:
        raise ValueError("state features, episode IDs, and goal mask must align")
    if k < 1:
        raise ValueError("k must be positive")
    n = len(x) if node_limit is None else min(len(x), int(node_limit))
    return _gpu_cross_episode_topk(
        x[:n], ep[:n], k=k, batch_size=batch_size, progress=progress,
    )


def load_or_build_candidate_pairs(
    features: np.ndarray,
    episode_id: np.ndarray,
    is_goal: np.ndarray,
    *,
    dataset_id: str,
    k: int,
    cache_path: Path | None,
    node_limit: int | None = None,
    batch_size: int = 256,
    progress: bool = False,
) -> np.ndarray:
    """Load a dataset-bound kNN cache or build and save the deterministic pairs."""
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {
                "pairs", "dataset_id", "k", "n_nodes", "feature_dim", "node_limit",
                "candidate_rule", "feature_sha256", "episode_sha256", "goal_sha256",
            }
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
                "candidate_rule": (str(cached["candidate_rule"].item()), CANDIDATE_RULE),
                "feature_sha256": (str(cached["feature_sha256"].item()), _array_sha256(features)),
                "episode_sha256": (str(cached["episode_sha256"].item()), _array_sha256(episode_id)),
                "goal_sha256": (str(cached["goal_sha256"].item()), _array_sha256(is_goal)),
            }
            for name, (actual, expected) in checks.items():
                if actual != expected:
                    raise ValueError(
                        f"candidate cache {name} mismatch: cached={actual!r}, expected={expected!r}"
                    )
            pairs = np.asarray(cached["pairs"], dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("candidate cache pairs must have shape (N, 2)")
        if pairs.size and (np.any(pairs < 0) or np.any(pairs >= len(features))):
            raise ValueError("candidate cache contains an out-of-range state index")
        return pairs

    pairs = cross_episode_knn_pairs(
        features, episode_id, is_goal, k=k, node_limit=node_limit,
        batch_size=batch_size, progress=progress,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            pairs=pairs.astype(np.int32), dataset_id=np.asarray(dataset_id),
            k=np.asarray(k, dtype=np.int64), n_nodes=np.asarray(len(features), dtype=np.int64),
            feature_dim=np.asarray(np.asarray(features).shape[1], dtype=np.int64),
            node_limit=np.asarray(-1 if node_limit is None else node_limit, dtype=np.int64),
            candidate_rule=np.asarray(CANDIDATE_RULE),
            feature_sha256=np.asarray(_array_sha256(features)),
            episode_sha256=np.asarray(_array_sha256(episode_id)),
            goal_sha256=np.asarray(_array_sha256(is_goal)),
        )
    return pairs


def load_or_build_window_candidate_pairs(
    window_features: np.ndarray,
    trans,
    is_goal: np.ndarray,
    *,
    dataset_id: str,
    window: int = 5,
    k: int = 32,
    cache_path: Path | None,
    node_limit: int | None = None,
    batch_size: int = 256,
    progress: bool = False,
) -> np.ndarray:
    """Cache cross-episode kNN pairs over chunk-boundary window nodes.

    Boundary states are the only sources. All boundary nodes and all success
    nodes are eligible destinations, so a success node remains reachable even
    when its time is not aligned to the chunk boundary.
    """
    features = np.ascontiguousarray(window_features, dtype=np.float32)
    goals = np.asarray(is_goal, dtype=np.bool_)
    if features.ndim != 2 or goals.shape != (len(features),):
        raise ValueError("window features and goal labels must align")
    node_ids, episodes, source_mask = puzzle_window_candidate_nodes(
        trans, goals, window=window
    )
    if node_limit is not None:
        limit = int(node_limit)
        if limit < 1:
            raise ValueError("node_limit must be positive")
        node_ids, episodes, source_mask = node_ids[:limit], episodes[:limit], source_mask[:limit]
    if len(node_ids) <= k:
        raise ValueError(f"window candidate table has {len(node_ids)} nodes; need more than k={k}")
    local_features = np.ascontiguousarray(features[node_ids])
    local_goals = goals[node_ids]
    node_hash = _array_sha256(node_ids)
    feature_hash = _array_sha256(local_features)
    episode_hash = _array_sha256(episodes)
    goal_hash = _array_sha256(local_goals)
    rule = f"{CANDIDATE_RULE}_boundary_window{int(window)}"

    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            required = {
                "pairs", "dataset_id", "k", "window", "candidate_rule",
                "node_ids_sha256", "feature_sha256", "episode_sha256", "goal_sha256",
            }
            if not required.issubset(cached.files):
                raise ValueError("window candidate cache is missing required metadata")
            checks = {
                "dataset_id": (str(cached["dataset_id"].item()), dataset_id),
                "k": (int(cached["k"].item()), int(k)),
                "window": (int(cached["window"].item()), int(window)),
                "candidate_rule": (str(cached["candidate_rule"].item()), rule),
                "node_ids_sha256": (str(cached["node_ids_sha256"].item()), node_hash),
                "feature_sha256": (str(cached["feature_sha256"].item()), feature_hash),
                "episode_sha256": (str(cached["episode_sha256"].item()), episode_hash),
                "goal_sha256": (str(cached["goal_sha256"].item()), goal_hash),
            }
            for name, (actual, expected) in checks.items():
                if actual != expected:
                    raise ValueError(
                        f"window candidate cache {name} mismatch: "
                        f"cached={actual!r}, expected={expected!r}"
                    )
            pairs = np.asarray(cached["pairs"], dtype=np.int32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("window candidate pairs must have shape (N,2)")
        if pairs.size:
            src_local = np.searchsorted(node_ids, pairs[:, 0])
            dst_local = np.searchsorted(node_ids, pairs[:, 1])
            src_safe = np.minimum(src_local, len(node_ids) - 1)
            dst_safe = np.minimum(dst_local, len(node_ids) - 1)
            invalid = (
                np.any(pairs < 0) or np.any(pairs >= len(features))
                or np.any(src_local >= len(node_ids)) or np.any(dst_local >= len(node_ids))
                or np.any(node_ids[src_safe] != pairs[:, 0])
                or np.any(node_ids[dst_safe] != pairs[:, 1])
                or np.any(~source_mask[src_safe])
            )
            if invalid:
                raise ValueError("window candidate cache contains an invalid source or node index")
        return pairs

    local_pairs = _gpu_cross_episode_topk(
        local_features, episodes, k=k, batch_size=batch_size, progress=progress,
    )
    local_pairs = local_pairs[source_mask[local_pairs[:, 0]]]
    pairs = node_ids[local_pairs].astype(np.int32, copy=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            pairs=pairs,
            dataset_id=np.asarray(dataset_id), k=np.asarray(k, dtype=np.int64),
            window=np.asarray(window, dtype=np.int64), candidate_rule=np.asarray(rule),
            node_ids_sha256=np.asarray(node_hash), feature_sha256=np.asarray(feature_hash),
            episode_sha256=np.asarray(episode_hash), goal_sha256=np.asarray(goal_hash),
            node_ids=node_ids.astype(np.int32),
        )
    return pairs


class _ActionPlanStore:
    """Lazily read accepted action plans from a compact disk-backed spool."""

    def __init__(self, pairs, lengths, errors, samples, offsets, action_rows):
        self.pairs = np.asarray(pairs)
        self.lengths = np.asarray(lengths)
        self.errors = np.asarray(errors)
        self.samples = np.asarray(samples)
        self.offsets = np.asarray(offsets)
        self.actions = action_rows
        self.sources = self.pairs[:, 0]
        if self.sources.size > 1 and np.any(self.sources[1:] < self.sources[:-1]):
            raise ValueError("optimistic candidate pairs must be grouped by source state")

    def __getitem__(self, key):
        source, target = map(int, key)
        left = int(np.searchsorted(self.sources, source, side="left"))
        right = int(np.searchsorted(self.sources, source, side="right"))
        if left == right:
            raise KeyError((source, target))
        matches = np.flatnonzero(self.pairs[left:right, 1] == target)
        if not matches.size:
            raise KeyError((source, target))
        index = left + int(matches[0])
        length = int(self.lengths[index])
        offset = int(self.offsets[index])
        if length <= 0 or offset < 0 or self.actions is None:
            raise KeyError((source, target))
        chunks = np.asarray(
            self.actions[offset:offset + length * 5], dtype=np.float32
        ).reshape(length, 5, 5).copy()
        return BridgePlan(
            source=source, target=target, chunks=chunks, length_chunks=length,
            endpoint_error=float(self.errors[index]), sample_index=int(self.samples[index]),
        )

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _artifact_with_bridges(
    base: GraphArtifact,
    bridge_src: np.ndarray,
    bridge_dst: np.ndarray,
    bridge_weight_steps: np.ndarray,
    cost_steps: np.ndarray,
    successor: np.ndarray,
    *,
    max_chunks: int = 4,
) -> GraphArtifact:
    old = base.edges
    bridge_src = np.asarray(bridge_src, dtype=np.int64)
    bridge_dst = np.asarray(bridge_dst, dtype=np.int64)
    bridge_weight_steps = np.asarray(bridge_weight_steps, dtype=np.int64)
    n_added = len(bridge_src)
    src = np.concatenate((old.src, bridge_src))
    dst = np.concatenate((old.dst, bridge_dst))
    kappa = np.concatenate((old.kappa, bridge_weight_steps))
    edges = EdgeSet(
        src=src, dst=dst, kappa=kappa,
        n_pos=np.concatenate((old.n_pos, np.zeros(n_added, dtype=np.int64))),
        n_neg=np.concatenate((old.n_neg, np.zeros(n_added, dtype=np.int64))),
        kind=np.concatenate((old.kind, np.ones(n_added, dtype=np.int8))),
        dist=np.concatenate((old.dist, np.zeros(n_added, dtype=np.float32))),
        anchor=np.concatenate((old.anchor, bridge_dst)),
        bridge_type=np.concatenate((old.bridge_type, np.full(n_added, -1, dtype=np.int8))),
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
    weights = np.concatenate((base.weights, bridge_weight_steps.astype(np.float64) * c))
    meta = dict(base.meta)
    meta.update({
        "bridge_v07": True,
        "bridge_strategy": "validated_action_chunk_layered_dijkstra",
        "bridge_v07_max_added_per_path": 2,
        "bridge_v07_added_edges": n_added,
        "bridge_v07_graph_cost_unit": "steps",
        "bridge_v07_candidate_rule": CANDIDATE_RULE,
        "bridge_v07_successor_budget_layers": 3,
    })
    return GraphArtifact(
        config=config, feature_stats=base.feature_stats, nodes=base.nodes,
        edges=edges, weights=weights, cost=np.asarray(cost_steps) * c,
        succ=np.asarray(successor, dtype=np.int64).copy(), meta=meta,
    )


def _simulator_witness(
    bridge_src: np.ndarray, bridge_dst: np.ndarray, plan_store: _ActionPlanStore,
    data, sim_state, *, env_id: str, trials: int, seed: int,
) -> dict:
    if not len(bridge_src):
        return {"n": 0, "landing_rate": None, "passed_gate": False, "reason": "no accepted bridges"}
    import gymnasium as gym
    import ogbench  # noqa: F401

    ids = np.arange(len(bridge_src), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(ids) > trials:
        ids = np.sort(rng.choice(ids, size=trials, replace=False))
    env = gym.make(env_id, max_episode_steps=1_000_000_000)
    verifier = OGBenchVerifier(env.unwrapped)
    schema = OGBenchPuzzleStateSchema()
    results = []
    try:
        for ordinal in ids:
            source, target = int(bridge_src[ordinal]), int(bridge_dst[ordinal])
            plan = plan_store[(source, target)]
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
    artifact: GraphArtifact, trans, sim_state, data, plan_store: _ActionPlanStore,
    baseline_path: Path, path_solution, *, env_id: str, b_max: int = 2,
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
    if path_solution.successor.shape[0] != b_max + 1:
        raise ValueError("layered path solution has an unexpected bridge budget")
    n_nodes = artifact.nodes.n_nodes
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
                                "baseline_graph_cost_steps": base_result.get("graph_cost_steps")})
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
            remaining_bridges = b_max
            while not artifact.nodes.is_goal[state] and visited <= artifact.nodes.n_nodes:
                flat_target = int(path_solution.successor[remaining_bridges, state])
                target = flat_target % n_nodes if flat_target >= 0 else -1
                next_budget = flat_target // n_nodes if flat_target >= 0 else -1
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
                    plan = plan_store.get((state, target))
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
                    if next_budget != remaining_bridges - 1:
                        reason = "layered_bridge_budget_mismatch"
                        break
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
                elif next_budget != remaining_bridges:
                    reason = "layered_temporal_budget_mismatch"
                    break
                remaining_bridges = next_budget
                state = target
                visited += 1
            if not success and reason is None:
                reason = "reached_goal_without_sim_success" if artifact.nodes.is_goal[state] else "path_limit"
            results.append({"start_state": start, "success": bool(success),
                            "actual_steps": int(actual_steps), "graph_cost_steps": cost_steps,
                            "failure_reason": reason,
                            "baseline_success": bool(base_result["success"]),
                            "baseline_actual_steps": int(base_result["actual_steps"]),
                            "baseline_graph_cost_steps": (
                                float(base_result["graph_cost_steps"])
                                if base_result.get("graph_cost_steps") is not None else None
                            ),
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
    reachable_mask = np.asarray(
        [item["failure_reason"] != "graph_unreachable" for item in results], dtype=bool
    )
    reachable_count = int(np.count_nonzero(reachable_mask))
    unreachable_count = int(len(results) - reachable_count)
    reachable_success_rate = (
        float(np.mean(new_success[reachable_mask])) if reachable_count else None
    )
    p_rate = float(binomtest(
        int(np.count_nonzero(new_success)), len(new_success), baseline_rate,
        alternative="greater",
    ).pvalue)
    result = {
        "n": 500,
        "start_selection_sha256": digest,
        "baseline_start_selection_sha256": baseline.get("start_selection_sha256"),
        "success_rate": success_rate,
        "reachable_start_count": reachable_count,
        "unreachable_start_count": unreachable_count,
        "success_rate_among_reachable_starts": reachable_success_rate,
        "baseline_success_rate": baseline_rate,
        "paired_success_wins": wins,
        "paired_success_losses": losses,
        "success_one_sided_binomial_p_vs_baseline_rate": p_rate,
        "success_mcnemar_one_sided_p": p_success,
        "success_cost_time_spearman_common_success": rho_new,
        "baseline_cost_time_spearman_common_success": rho_base,
        "rank_difference_bootstrap_95ci": rank_ci,
        "rank_difference_bootstrap_one_sided_p": p_rank,
        "passed_q3_success_gate": bool(success_rate > baseline_rate and p_success < 0.05),
        "outcomes": results,
    }
    return result


def _layered_path(artifact: GraphArtifact, path_solution, start: int, b_max: int):
    """Materialize a path and edge IDs while retaining the remaining budget."""
    n_nodes = artifact.nodes.n_nodes
    edge_lookup = {
        (int(source), int(target)): int(edge_id)
        for edge_id, (source, target) in enumerate(zip(
            artifact.edges.src, artifact.edges.dst, strict=True
        ))
    }
    state, budget = int(start), int(b_max)
    path: list[PathStep] = []
    edge_ids: list[int] = []
    seen: set[tuple[int, int]] = set()
    while not artifact.nodes.is_goal[state]:
        layer_state = (budget, state)
        if layer_state in seen:
            raise ValueError("layered shortest-path successor contains a cycle")
        seen.add(layer_state)
        flattened = int(path_solution.successor[budget, state])
        if flattened < 0:
            raise ValueError(f"reachable state {state} has no layered successor")
        target, next_budget = flattened % n_nodes, flattened // n_nodes
        edge_id = edge_lookup.get((state, target))
        if edge_id is None:
            raise ValueError(f"layered successor {state}->{target} is missing from graph")
        kind = int(artifact.edges.kind[edge_id])
        expected_budget = budget - int(kind == 1)
        if next_budget != expected_budget:
            raise ValueError("layered successor consumes an invalid bridge budget")
        path.append(PathStep(state, kind, int(artifact.edges.kappa[edge_id])))
        edge_ids.append(edge_id)
        state, budget = target, next_budget
        if len(path) > (b_max + 1) * n_nodes:
            raise ValueError("layered path exceeded the expanded graph size")
    path.append(PathStep(state, None, 0))
    return path, edge_ids


def _servo_path_witness(
    artifact: GraphArtifact, trans, sim_state, baseline_path: Path,
    path_solution, *, env_id: str, b_max: int = 2,
) -> dict:
    """Evaluate layered paths with the calibrated closed-loop pose servo."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_outcomes = baseline.get("outcomes", [])
    if len(baseline_outcomes) != 500:
        raise ValueError("Q3 requires a 500-start servo baseline witness")
    start_selection = np.asarray([
        (int(item["start_state"]), int(bool(item["failed_episode"])))
        for item in baseline_outcomes
    ], dtype="<i8")
    starts = start_selection[:, 0].tolist()
    digest = hashlib.sha256(np.ascontiguousarray(start_selection).tobytes()).hexdigest()
    if digest != baseline.get("start_selection_sha256"):
        raise ValueError("servo baseline start-selection hash does not match its outcomes")
    try:
        import gymnasium as gym
        import ogbench  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Q3 servo witnessing requires the bench environment") from exc

    state_features, _, _, _ = state_table(trans)
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    schema = OGBenchPuzzleStateSchema()
    coordinates = schema.coordinates(state_features, aux)
    calibration = estimate_realized_speed(
        trans, coordinates, coordinates.mode == 0, schema
    )
    policy = ServoTrackingPolicy(
        artifact, trans, schema, coordinates, mode="servo", window=10,
        epsilon_p=np.asarray([0.5]), feedback_gain=1.0,
        response_gain=np.asarray(calibration["feedback_gain_by_dimension"], dtype=np.float64),
        contact_gain=0.3, free_distance=artifact.config.bridge.free_distance,
    )
    env = gym.make(env_id, max_episode_steps=1_000_000_000)
    verifier = OGBenchVerifier(env.unwrapped)
    c = step_cost(artifact.config.gamma)
    results = []
    try:
        for number, (start, baseline_outcome) in enumerate(
            zip(starts, baseline_outcomes, strict=True), 1
        ):
            if not np.isfinite(artifact.cost[start]):
                results.append({
                    "start_state": start, "success": False, "actual_steps": 0,
                    "graph_cost_steps": None, "failure_reason": "graph_unreachable",
                    "baseline_success": bool(baseline_outcome["success"]),
                    "baseline_actual_steps": int(baseline_outcome["actual_steps"]),
                    "baseline_graph_cost_steps": baseline_outcome.get("graph_cost_steps"),
                    "planned_bridges": None,
                })
                continue
            path, edge_ids = _layered_path(artifact, path_solution, start, b_max)
            policy.reset(start, path_override=path, edge_indices=edge_ids)
            graph_steps = float(artifact.cost[start] / c)
            max_steps = int(np.ceil(1.5 * graph_steps) + 20)
            payload = {
                "qpos": sim_state.qpos[start], "qvel": sim_state.qvel[start],
            }
            if sim_state.button_states is not None:
                payload["button_states"] = sim_state.button_states[start]
            observation = verifier.reset_to(payload)
            success = False
            failure_reason = None
            steps = 0
            while steps < max_steps:
                action = policy.act(verifier.sim_readout(
                    observation, schema_name="ogbench_puzzle"
                ))
                observation, reward, terminated, truncated, info = verifier.step(action)
                steps += 1
                success = bool(info.get("success", False)) or reward > 0
                status = policy.status()
                failure_reason = status["failure_reason"]
                if success or terminated or truncated or failure_reason:
                    if not success and (terminated or truncated):
                        failure_reason = "timeout"
                    break
            if not success and failure_reason is None and steps >= max_steps:
                failure_reason = "timeout"
            results.append({
                "start_state": start, "success": bool(success),
                "actual_steps": int(steps), "graph_cost_steps": graph_steps,
                "failure_reason": failure_reason,
                "baseline_success": bool(baseline_outcome["success"]),
                "baseline_actual_steps": int(baseline_outcome["actual_steps"]),
                "baseline_graph_cost_steps": baseline_outcome.get("graph_cost_steps"),
                "planned_bridges": int(sum(step.kind == 1 for step in path)),
                "bridge_landing": status.get("bridge_landing", []),
            })
            if number % 25 == 0:
                print(f"v0.7 servo 500-start witness: {number}/500", file=sys.stderr, flush=True)
    finally:
        env.close()

    new_success = np.asarray([item["success"] for item in results], dtype=bool)
    baseline_success = np.asarray([item["baseline_success"] for item in results], dtype=bool)
    wins = int(np.count_nonzero(new_success & ~baseline_success))
    losses = int(np.count_nonzero(~new_success & baseline_success))
    paired_p = float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue) if wins + losses else 1.0
    baseline_rate = float(np.mean(baseline_success))
    success_rate = float(np.mean(new_success))
    reachable = np.asarray([item["failure_reason"] != "graph_unreachable" for item in results])
    reachable_count = int(np.count_nonzero(reachable))
    return {
        "controller": "calibrated_pose_servo",
        "n": 500,
        "start_selection_sha256": digest,
        "baseline_start_selection_sha256": baseline.get("start_selection_sha256"),
        "success_rate": success_rate,
        "baseline_success_rate": baseline_rate,
        "reachable_start_count": reachable_count,
        "unreachable_start_count": int(len(results) - reachable_count),
        "success_rate_among_reachable_starts": (
            float(np.mean(new_success[reachable])) if reachable_count else None
        ),
        "paired_success_wins": wins,
        "paired_success_losses": losses,
        "success_mcnemar_one_sided_p": paired_p,
        "passed_q3_success_gate": bool(success_rate > baseline_rate and paired_p < 0.05),
        "outcomes": results,
    }


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
    parser.add_argument("--env", default="puzzle-4x4-play-singletask-task2-v0")
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--max-added-per-path", type=int, default=2)
    parser.add_argument("--plan-samples", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--witness-trials", type=int, default=200)
    parser.add_argument("--path-baseline", type=Path,
                        help="matched 500-start G-R rho=0.9 servo JSON for Q3 comparison")
    parser.add_argument("--replay-path-baseline", type=Path,
                        help="matched 500-start G-R rho=0.9 replay JSON for Q3 replay comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-node-limit", type=int,
                        help="smoke-test cap; omit for full graph")
    parser.add_argument("--candidate-cache", type=Path,
                        help="feature-hashed exact GPU candidate cache")
    parser.add_argument("--knn-batch-size", type=int, default=256)
    parser.add_argument("--knn-parity-size", type=int, default=50_000,
                        help="compare this many sampled rows against exact cKDTree; set 0 to skip")
    parser.add_argument("--skip-simulator", action="store_true")
    args = parser.parse_args()

    started = time.time()
    stage_seconds: dict[str, float] = {}
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
    parity_report = None
    if args.knn_parity_size:
        parity_started = time.time()
        parity_report = verify_gpu_knn_against_ckdtree(
            features, episode, sample_size=args.knn_parity_size,
            k=args.knn, seed=args.seed, batch_size=args.knn_batch_size,
        )
        stage_seconds["knn_parity_seconds"] = float(time.time() - parity_started)
        if parity_report["mismatched_rows"]:
            raise RuntimeError(
                "GPU kNN disagrees with cKDTree on the tie-free parity subset: "
                f"{parity_report['mismatched_rows']} / "
                f"{parity_report['rows_compared_excluding_boundary_ties']} rows"
            )

    knn_started = time.time()
    pairs = load_or_build_candidate_pairs(
        features, episode, base.nodes.is_goal,
        dataset_id=transition_meta["dataset_id"], k=args.knn,
        cache_path=args.candidate_cache, node_limit=args.candidate_node_limit,
        batch_size=args.knn_batch_size, progress=True,
    )
    stage_seconds["knn_search_or_cache_seconds"] = float(time.time() - knn_started)
    # Candidate pairs cross episodes, while every G0 edge is an in-episode
    # temporal transition. Therefore candidate/temporal duplicate keys are
    # impossible and no global pair sort is needed here.

    c = step_cost(base.config.gamma)
    temporal = base.edges.kind == 0
    n_nodes = base.nodes.n_nodes
    goals = np.flatnonzero(base.nodes.is_goal).astype(np.int64)
    base_src = base.edges.src[temporal].astype(np.int64, copy=False)
    base_dst = base.edges.dst[temporal].astype(np.int64, copy=False)
    base_weight = (base.weights[temporal] / c).astype(np.float64, copy=False)

    c0_started = time.time()
    c0_solution = layered_reverse_dijkstra(
        n_nodes, goals, base_src, base_dst, base_weight,
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64), b_max=0,
    )
    c0 = c0_solution.cost_steps[0]
    stage_seconds["g0_reverse_dijkstra_seconds"] = float(time.time() - c0_started)

    knn_count = int(len(pairs))
    per_source_counts = np.bincount(pairs[:, 0], minlength=n_nodes) if pairs.size else np.zeros(n_nodes, dtype=np.int64)
    success_destination_candidate_count = (
        int(np.count_nonzero(base.nodes.is_goal[pairs[:, 1]])) if len(pairs) else 0
    )
    c0_target_finite = np.isfinite(c0[pairs[:, 1]]) if knn_count else np.empty(0, dtype=bool)
    excluded_target_infinite = int(np.count_nonzero(~c0_target_finite))
    if knn_count:
        with np.errstate(invalid="ignore"):
            optimistic_gain = c0[pairs[:, 0]] - c0[pairs[:, 1]]
        optimistic = c0_target_finite & (optimistic_gain >= 5.0 + args.delta)
        optimistic_pairs = pairs[optimistic]
    else:
        optimistic_pairs = np.empty((0, 2), dtype=np.int64)
    optimistic_source_unreachable = int(
        np.count_nonzero(~np.isfinite(c0[optimistic_pairs[:, 0]]))
    ) if len(optimistic_pairs) else 0
    del c0_target_finite
    del pairs

    args.out.mkdir(parents=True, exist_ok=True)
    action_spool_path = args.out / "accepted_actions.spool.f32"
    accepted_lengths = np.zeros(len(optimistic_pairs), dtype=np.uint8)
    accepted_errors = np.zeros(len(optimistic_pairs), dtype=np.float32)
    accepted_samples = np.zeros(len(optimistic_pairs), dtype=np.uint8)
    accepted_action_offsets = np.full(len(optimistic_pairs), -1, dtype=np.int64)
    action_row_count = 0
    validation_count = 0
    judge_started = time.time()
    judge_batch_size = 1024
    action_spool_path.parent.mkdir(parents=True, exist_ok=True)
    with action_spool_path.open("wb") as action_spool:
        for start in range(0, len(optimistic_pairs), judge_batch_size):
            batch = optimistic_pairs[start:start + judge_batch_size]
            plans = bridge_evaluator(
                features[batch[:, 0]], features[batch[:, 1]], batch[:, 0], batch[:, 1],
                epsilon_z=data.epsilon_z, state_std=data.state_std,
                delta_std=data.delta_std, n_cont=data.n_cont, n_bits=data.n_bits,
                h=5, max_chunks=args.max_chunks, n_samples=args.plan_samples,
                seed=args.seed + start, candidate_batch=judge_batch_size,
            )
            validation_count += len(batch)
            batch_payload: list[bytes] = []
            for local, plan in enumerate(plans):
                if plan is None:
                    continue
                global_index = start + local
                accepted_lengths[global_index] = int(plan.length_chunks)
                accepted_errors[global_index] = float(plan.endpoint_error)
                accepted_samples[global_index] = int(plan.sample_index)
                accepted_action_offsets[global_index] = action_row_count
                flat = np.ascontiguousarray(plan.chunks, dtype=np.float32).reshape(-1, 5)
                batch_payload.append(flat.tobytes())
                action_row_count += len(flat)
            if batch_payload:
                action_spool.write(b"".join(batch_payload))
            if ((start // judge_batch_size + 1) % 20 == 0
                    or start + len(batch) == len(optimistic_pairs)):
                action_spool.flush()
                print(
                    f"JIT bridge judge: candidates={min(start + len(batch), len(optimistic_pairs))}/"
                    f"{len(optimistic_pairs)}, accepted={np.count_nonzero(accepted_lengths)}",
                    file=sys.stderr, flush=True,
                )
    stage_seconds["batched_policy_transition_judging_seconds"] = float(time.time() - judge_started)
    judge_throughput = (
        float(validation_count / stage_seconds["batched_policy_transition_judging_seconds"])
        if stage_seconds["batched_policy_transition_judging_seconds"] > 0 else None
    )

    accepted_indices = np.flatnonzero(accepted_lengths).astype(np.int64)
    accepted_count = int(len(accepted_indices))
    accepted_src = optimistic_pairs[accepted_indices, 0].astype(np.int64)
    accepted_dst = optimistic_pairs[accepted_indices, 1].astype(np.int64)
    accepted_weight = accepted_lengths[accepted_indices].astype(np.float64) * 5.0
    dijkstra_started = time.time()
    path_solution = layered_reverse_dijkstra(
        n_nodes, goals, base_src, base_dst, base_weight,
        accepted_src, accepted_dst, accepted_weight,
        b_max=args.max_added_per_path,
    )
    stage_seconds["layered_reverse_dijkstra_seconds"] = float(time.time() - dijkstra_started)
    retained_mask = path_solution.used_bridge_mask
    retained_indices = accepted_indices[retained_mask]
    retained_src = accepted_src[retained_mask]
    retained_dst = accepted_dst[retained_mask]
    retained_weight = accepted_weight[retained_mask]
    retained_lengths = accepted_lengths[retained_indices]
    retained_errors = accepted_errors[retained_indices]
    retained_samples = accepted_samples[retained_indices]
    retained_offsets = accepted_action_offsets[retained_indices]
    retained_count = int(len(retained_indices))
    action_store = (
        np.memmap(action_spool_path, dtype=np.float32, mode="r", shape=(action_row_count, 5))
        if action_row_count else None
    )
    plan_store = _ActionPlanStore(
        optimistic_pairs, accepted_lengths, accepted_errors, accepted_samples,
        accepted_action_offsets, action_store,
    )
    retained_successor = path_solution.successor[args.max_added_per_path]
    retained_successor = np.where(
        retained_successor >= 0, retained_successor % n_nodes, -1
    ).astype(np.int64)
    artifact_started = time.time()
    artifact = _artifact_with_bridges(
        base, retained_src, retained_dst, retained_weight,
        path_solution.cost_steps[args.max_added_per_path], retained_successor,
        max_chunks=args.max_chunks,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    graph_path = args.out / "graph.npz"
    graph_hash = save_graph(artifact, graph_path)
    targets = critic_targets(
        trans, artifact, TargetConfig(h=5, unit="steps", kind="regret")
    )
    save_targets(targets, args.out / "targets")
    np.savez_compressed(
        args.out / "layered_paths.npz",
        cost_steps=path_solution.cost_steps,
        successor=path_solution.successor,
        bridge_count=path_solution.bridge_count,
        b_max=np.asarray(args.max_added_per_path, dtype=np.int64),
        graph_hash=np.asarray(graph_hash),
    )

    action_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(retained_lengths.astype(np.int64) * 5, dtype=np.int64),
    ))
    gathered_actions_path = args.out / "retained_actions.gather.f32"
    total_action_rows = int(action_offsets[-1])
    if total_action_rows:
        gathered_actions = np.memmap(
            gathered_actions_path, dtype=np.float32, mode="w+",
            shape=(total_action_rows, 5),
        )
        for ordinal, candidate_index in enumerate(retained_indices):
            length_steps = int(retained_lengths[ordinal]) * 5
            source_offset = int(retained_offsets[ordinal])
            dest_offset = int(action_offsets[ordinal])
            gathered_actions[dest_offset:dest_offset + length_steps] = action_store[
                source_offset:source_offset + length_steps
            ]
        gathered_actions.flush()
        saved_actions = gathered_actions
    else:
        saved_actions = np.empty((0, 5), dtype=np.float32)
    np.savez_compressed(
        args.out / "accepted_actions.npz", actions=saved_actions,
        offsets=action_offsets, source=retained_src, target=retained_dst,
        length_chunks=retained_lengths, endpoint_error=retained_errors,
        sample_index=retained_samples,
    )
    if total_action_rows:
        del gathered_actions
        gathered_actions_path.unlink(missing_ok=True)

    stage_seconds["artifact_and_target_write_seconds"] = float(time.time() - artifact_started)
    witness = None
    if not args.skip_simulator:
        q1_started = time.time()
        witness = _simulator_witness(
            retained_src, retained_dst, plan_store, data, sim_state, env_id=args.env,
            trials=args.witness_trials, seed=args.seed,
        )
        stage_seconds["q1_simulator_seconds"] = float(time.time() - q1_started)
    path_witness = None
    if args.path_baseline is not None and not args.skip_simulator:
        q3_started = time.time()
        path_witness = _servo_path_witness(
            artifact, trans, sim_state, args.path_baseline, path_solution,
            env_id=args.env, b_max=args.max_added_per_path,
        )
        stage_seconds["q3_servo_witness_seconds"] = float(time.time() - q3_started)
    replay_path_witness = None
    if args.replay_path_baseline is not None and not args.skip_simulator:
        q3_started = time.time()
        replay_path_witness = _path_witness(
            artifact, trans, sim_state, data, plan_store, args.replay_path_baseline,
            path_solution, env_id=args.env, b_max=args.max_added_per_path,
        )
        stage_seconds["q3_replay_witness_seconds"] = float(time.time() - q3_started)
    reachable_frac = float(np.mean(np.isfinite(path_solution.cost_steps[args.max_added_per_path])))
    g0_reachable_frac = float(np.mean(np.isfinite(c0)))
    reference_reachable_frac = (
        float(np.mean(np.isfinite(reference.cost))) if reference is not None else None
    )
    queried_sources = per_source_counts[:n_nodes if args.candidate_node_limit is None
                                      else min(n_nodes, args.candidate_node_limit)]
    bridge_count_values, bridge_count_frequencies = np.unique(
        path_solution.bridge_count[args.max_added_per_path], return_counts=True
    )
    used_success_edges = int(np.count_nonzero(base.nodes.is_goal[retained_dst]))
    build_total = float(time.time() - started)
    stage_seconds["total_seconds"] = build_total
    c0_target_finite_count = knn_count - excluded_target_infinite
    candidate_cache_hash = {
        "candidate_rule": CANDIDATE_RULE,
        "feature_sha256": _array_sha256(features),
        "episode_sha256": _array_sha256(episode),
        "goal_sha256": _array_sha256(base.nodes.is_goal),
    }
    result = {
        "spec": "Graph Critic v0.7 revised all-pairs layered build",
        "dataset_id": transition_meta["dataset_id"],
        "g0_hash": base.content_hash,
        "graph_hash": graph_hash,
        "model_checkpoint": str(args.checkpoint),
        "model_data_schema": data.metadata["schema"],
        "epsilon_raw": data.epsilon_raw.tolist(),
        "epsilon_z": data.epsilon_z.tolist(),
        "candidate": {"metric": "Euclidean on z-normalized continuous coordinates concatenated with binary button coordinates",
                      "backend": "jax_gpu_blocked_exact",
                      "k": args.knn, "cross_episode_only": True,
                      "candidate_rule": CANDIDATE_RULE,
                      "cache_hashes": candidate_cache_hash,
                      "knn_parity": parity_report,
                      "success_states_are_valid_destinations": True,
                      "success_destination_candidate_count": success_destination_candidate_count,
                      "count": knn_count,
                      "smoke_source_limit": args.candidate_node_limit,
                      "source_count": int(np.count_nonzero(queried_sources)),
                      "queried_source_count": int(len(queried_sources)),
                      "source_degree_fraction_at_k": float(np.mean(queried_sources == args.knn)) if len(queried_sources) else 0.0,
                      "source_degree_min_all": int(np.min(queried_sources)) if len(queried_sources) else 0,
                      "source_degree_p50": float(np.median(queried_sources[queried_sources > 0])) if np.any(queried_sources) else 0.0,
                      "source_degree_min_positive": int(np.min(queried_sources[queried_sources > 0])) if np.any(queried_sources) else 0},
        "candidate_filter": {
            "g0_target_infinite_excluded_count": excluded_target_infinite,
            "g0_target_finite_count": int(c0_target_finite_count),
            "optimistic_after_c0_cost_filter_count": int(len(optimistic_pairs)),
            "optimistic_source_initially_unreachable_count": optimistic_source_unreachable,
            "filter": "isfinite(C0[target]) and C0[source]-C0[target] >= h+delta; C0[source]=infinity allowed",
        },
        "bridges": {
            "delta_steps": args.delta, "Lambda_steps": 0.0,
            "B_max": args.max_added_per_path, "h": 5,
            "max_chunks": args.max_chunks, "N": args.plan_samples,
            "judged_candidate_count": int(validation_count),
            "model_accepted_count": accepted_count,
            "model_acceptance_rate": float(accepted_count / validation_count) if validation_count else None,
            "retained_shortest_path_edge_count": retained_count,
            "pruned_accepted_edge_count": int(accepted_count - retained_count),
            "retained_success_destination_edge_count": used_success_edges,
            "accepted_action_spool_bytes": int(action_row_count * 5 * np.dtype(np.float32).itemsize),
        },
        "cost_steps": {"finite_fraction": reachable_frac,
                       "g0_finite_fraction": g0_reachable_frac,
                       "reachable_nodes": int(np.count_nonzero(np.isfinite(path_solution.cost_steps[args.max_added_per_path]))),
                       "path_bridge_count_max": int(np.max(path_solution.bridge_count[args.max_added_per_path], initial=0)),
                       "path_bridge_count_distribution": {
                           str(int(value)): int(count)
                           for value, count in zip(bridge_count_values, bridge_count_frequencies, strict=True)
                       },
                       "cost_p50_finite": float(np.median(path_solution.cost_steps[args.max_added_per_path][np.isfinite(path_solution.cost_steps[args.max_added_per_path])])) if np.any(np.isfinite(path_solution.cost_steps[args.max_added_per_path])) else None},
        "q2_reachability": {"graph_fraction": reachable_frac,
                            "g0_fraction_same_dataset": g0_reachable_frac,
                            "gain_vs_g0": reachable_frac - g0_reachable_frac,
                            "reference_fraction": reference_reachable_frac,
                            "difference": (reachable_frac - reference_reachable_frac) if reference_reachable_frac is not None else None,
                            "reference_graph": str(args.reference_graph) if args.reference_graph else None},
        "simulator_q1": witness,
        "path_q3_servo": path_witness,
        "path_q3_replay": replay_path_witness,
        "judge_candidates_per_second": judge_throughput,
        "stage_seconds": stage_seconds,
        "elapsed_seconds": build_total,
    }
    result = _json_safe(result)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    del plan_store
    if action_store is not None:
        del action_store
    action_spool_path.unlink(missing_ok=True)
    gathered_actions_path.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in result.items() if k != "insertion"}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
