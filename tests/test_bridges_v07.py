"""Revised v0.7 kNN, batched judge, and layered-path checks."""

import numpy as np
import pytest

from graphcritic.bridges.insert import layered_reverse_dijkstra
from experiments import eval_bridges
from experiments.eval_bridges import load_or_build_candidate_pairs


def _has_jax_gpu():
    try:
        import jax
        return bool(jax.devices("gpu"))
    except (ImportError, RuntimeError):
        return False


@pytest.mark.skipif(not _has_jax_gpu(), reason="GPU kNN check requires a JAX GPU")
def test_cross_episode_knn_includes_success_nodes_as_destinations():
    features = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    episodes = np.asarray([0, 1, 2, 3], dtype=np.int64)
    goals = np.asarray([False, True, False, False])
    pairs = eval_bridges.cross_episode_knn_pairs(
        features, episodes, goals, k=1, batch_size=2,
    )
    assert (0, 1) in set(map(tuple, pairs.tolist()))


def test_candidate_cache_binds_rule_and_feature_hashes(tmp_path, monkeypatch):
    features = np.asarray([[0.0], [0.1], [0.2], [0.01]], dtype=np.float32)
    episodes = np.asarray([0, 1, 0, 1])
    goals = np.asarray([False, False, False, True])
    cache = tmp_path / "candidate-pairs.npz"
    expected_pairs = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    monkeypatch.setattr(
        eval_bridges, "cross_episode_knn_pairs",
        lambda *args, **kwargs: expected_pairs.copy(),
    )
    expected = load_or_build_candidate_pairs(
        features, episodes, goals, dataset_id="data-a", k=1, cache_path=cache,
    )
    loaded = load_or_build_candidate_pairs(
        features, episodes, goals, dataset_id="data-a", k=1, cache_path=cache,
    )
    np.testing.assert_array_equal(loaded, expected)
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    assert str(payload["candidate_rule"].item()) == "cross_episode_incl_goal_v2"
    payload["candidate_rule"] = np.asarray("cross_episode_without_goal_v1")
    np.savez_compressed(cache, **payload)
    with pytest.raises(ValueError, match="candidate_rule mismatch"):
        load_or_build_candidate_pairs(
            features, episodes, goals, dataset_id="data-a", k=1, cache_path=cache,
        )
    payload["candidate_rule"] = np.asarray("cross_episode_incl_goal_v2")
    np.savez_compressed(cache, **payload)
    changed = features.copy()
    changed[0, 0] += 0.5
    with pytest.raises(ValueError, match="feature_sha256 mismatch"):
        load_or_build_candidate_pairs(
            changed, episodes, goals, dataset_id="data-a", k=1, cache_path=cache,
        )


def test_action_plan_store_reads_only_the_requested_accepted_chunks(tmp_path):
    from experiments.eval_bridges import _ActionPlanStore

    rows = np.arange(15 * 5, dtype=np.float32).reshape(15, 5)
    spool_path = tmp_path / "actions.f32"
    rows.tofile(spool_path)
    actions = np.memmap(spool_path, dtype=np.float32, mode="r", shape=(15, 5))
    store = _ActionPlanStore(
        np.asarray([[1, 3], [1, 4], [2, 8]], dtype=np.int32),
        np.asarray([1, 2, 0], dtype=np.uint8),
        np.asarray([0.2, 0.4, 0.0], dtype=np.float32),
        np.asarray([2, 1, 0], dtype=np.uint8),
        np.asarray([0, 5, -1], dtype=np.int64), actions,
    )
    plan = store[(1, 4)]
    assert plan.length_chunks == 2
    assert plan.sample_index == 1
    np.testing.assert_array_equal(plan.chunks.reshape(-1, 5), rows[5:15])
    assert store.get((2, 8)) is None
    del actions


def _expanded_floyd_warshall(n, goals, base_src, base_dst, base_weight,
                             bridge_src, bridge_dst, bridge_weight, b_max):
    count = (b_max + 1) * n
    dist = np.full((count, count), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for budget in range(b_max + 1):
        offset = budget * n
        dist[offset + base_src, offset + base_dst] = np.minimum(
            dist[offset + base_src, offset + base_dst], base_weight
        )
    for budget in range(1, b_max + 1):
        dist[budget * n + bridge_src, (budget - 1) * n + bridge_dst] = np.minimum(
            dist[budget * n + bridge_src, (budget - 1) * n + bridge_dst], bridge_weight
        )
    for mid in range(count):
        dist = np.minimum(dist, dist[:, mid, None] + dist[None, mid, :])
    expected = np.empty((b_max + 1, n), dtype=np.float64)
    goal_nodes = np.concatenate([
        budget * n + goals for budget in range(b_max + 1)
    ])
    for budget in range(b_max + 1):
        expected[budget] = np.min(dist[budget * n:(budget + 1) * n][:, goal_nodes], axis=1)
    return expected


def test_layered_dijkstra_matches_brute_force_and_pruning_preserves_costs():
    rng = np.random.default_rng(42)
    n, b_max = 9, 2
    base_src = np.asarray([0, 1, 2, 3, 4, 5, 6, 0, 2, 4, 7], dtype=np.int64)
    base_dst = np.asarray([1, 2, 3, 4, 5, 6, 8, 4, 6, 8, 8], dtype=np.int64)
    base_weight = rng.integers(1, 15, len(base_src)).astype(np.float64)
    bridge_src = np.asarray([0, 1, 2, 3, 5, 6, 7], dtype=np.int64)
    bridge_dst = np.asarray([8, 8, 5, 8, 8, 4, 3], dtype=np.int64)
    bridge_weight = rng.integers(1, 20, len(bridge_src)).astype(np.float64)
    goals = np.asarray([8], dtype=np.int64)

    solution = layered_reverse_dijkstra(
        n, goals, base_src, base_dst, base_weight,
        bridge_src, bridge_dst, bridge_weight, b_max=b_max,
    )
    expected = _expanded_floyd_warshall(
        n, goals, base_src, base_dst, base_weight,
        bridge_src, bridge_dst, bridge_weight, b_max,
    )
    np.testing.assert_allclose(solution.cost_steps, expected)
    assert np.all(solution.bridge_count <= b_max)

    retained = solution.used_bridge_mask
    pruned = layered_reverse_dijkstra(
        n, goals, base_src, base_dst, base_weight,
        bridge_src[retained], bridge_dst[retained], bridge_weight[retained],
        b_max=b_max,
    )
    np.testing.assert_allclose(pruned.cost_steps, solution.cost_steps)


def test_unreachable_sources_can_take_a_bridge_to_a_finite_target():
    result = layered_reverse_dijkstra(
        3, np.asarray([2]), np.asarray([1]), np.asarray([2]), np.asarray([1.0]),
        np.asarray([0]), np.asarray([1]), np.asarray([2.0]), b_max=2,
    )
    assert np.isinf(result.cost_steps[0, 0])
    assert result.cost_steps[2, 0] == 3.0
    assert result.bridge_count[2, 0] == 1


def test_layered_path_extraction_keeps_budget_dependent_edges():
    from types import SimpleNamespace
    from experiments.eval_bridges import _layered_path

    artifact = SimpleNamespace(
        nodes=SimpleNamespace(n_nodes=4, is_goal=np.asarray([False, False, False, True])),
        edges=SimpleNamespace(
            src=np.asarray([0, 1, 2]), dst=np.asarray([1, 2, 3]),
            kind=np.asarray([0, 1, 0]), kappa=np.asarray([1, 5, 1]),
        ),
    )
    successor = np.full((3, 4), -1, dtype=np.int64)
    successor[2, 0] = 2 * 4 + 1
    successor[2, 1] = 1 * 4 + 2
    successor[1, 2] = 1 * 4 + 3
    solution = SimpleNamespace(successor=successor)
    path, edge_ids = _layered_path(artifact, solution, 0, 2)
    assert [(step.vertex, step.kind, step.tau) for step in path] == [
        (0, 0, 1), (1, 1, 5), (2, 0, 1), (3, None, 0),
    ]
    assert edge_ids == [0, 1, 2]


def test_layered_dijkstra_rejects_nonpositive_base_edge_weights():
    with pytest.raises(ValueError, match="base edge weights must be finite and positive"):
        layered_reverse_dijkstra(
            2, np.asarray([1]), np.asarray([0]), np.asarray([1]), np.asarray([0.0]),
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64), b_max=2,
        )


def test_layered_dijkstra_rejects_negative_bridge_weight():
    with pytest.raises(ValueError, match="bridge edge weights must be finite and non-negative"):
        layered_reverse_dijkstra(
            2, np.asarray([1]), np.asarray([0]), np.asarray([1]), np.asarray([1.0]),
            np.asarray([0]), np.asarray([1]), np.asarray([-0.1]), b_max=1,
        )


def test_layered_dijkstra_preserves_zero_cost_bridge_and_budget_count():
    result = layered_reverse_dijkstra(
        3, np.asarray([2]), np.asarray([1]), np.asarray([2]), np.asarray([1.0]),
        np.asarray([0]), np.asarray([1]), np.asarray([0.0]), b_max=2,
    )
    assert result.cost_steps[2, 0] == pytest.approx(1.0)
    assert result.bridge_count[2, 0] == 1
    assert result.used_bridge_mask.tolist() == [True]


@pytest.mark.skipif(not _has_jax_gpu(), reason="GPU kNN parity check requires a JAX GPU")
def test_gpu_knn_matches_ckdtree_on_small_tie_free_subset():
    rng = np.random.default_rng(123)
    features = rng.normal(size=(300, 7)).astype(np.float32)
    episodes = np.arange(len(features), dtype=np.int64) % 17
    report = eval_bridges.verify_gpu_knn_against_ckdtree(
        features, episodes, sample_size=160, k=8, seed=7, batch_size=32,
    )
    assert report["rows_compared_excluding_boundary_ties"] > 0
    assert report["mismatched_rows"] == 0


def test_jit_judge_matches_numpy_judge_for_256_fixed_noise_candidates():
    import jax.numpy as jnp

    class GoalVelocity:
        @staticmethod
        def apply(variables, inputs):
            del variables
            delta = inputs[:, 3:4] - inputs[:, :1]
            return jnp.broadcast_to(delta, (inputs.shape[0], 25))

    class ActionDeltaAndCopyButtons:
        @staticmethod
        def apply(variables, inputs):
            del variables
            delta = 0.5 * jnp.mean(inputs[:, 3:8], axis=1, keepdims=True)
            bits = inputs[:, 1:3] * 2.0 - 1.0
            return jnp.concatenate((delta, bits * 10.0), axis=1)

    rng = np.random.default_rng(9)
    count, samples, chunks = 256, 4, 4
    source = np.zeros((count, 3), dtype=np.float32)
    source[:, 1:] = rng.integers(0, 2, size=(count, 2)).astype(np.float32)
    target = source.copy()
    target[count // 2:, 0] = 0.3
    fixed_noise = rng.normal(0.0, 0.05, size=(chunks, count, samples, 25)).astype(np.float32)
    args = dict(
        policy=(GoalVelocity(), {}),
        transition=(ActionDeltaAndCopyButtons(), {}),
        epsilon_z=np.asarray([0.1], dtype=np.float32),
        state_std=np.asarray([1.0], dtype=np.float32),
        delta_std=np.asarray([1.0], dtype=np.float32),
        n_cont=1, n_bits=2, h=5, max_chunks=chunks,
        n_samples=samples, candidate_batch=64, noise=fixed_noise,
    )
    source_ids = np.arange(count, dtype=np.int64)
    target_ids = source_ids + count
    from graphcritic.bridges.check import make_and_evaluate as numpy_judge
    from experiments.v07_runtime import make_and_evaluate as jit_judge

    expected = numpy_judge(source, target, source_ids, target_ids, **args)
    actual = jit_judge(source, target, source_ids, target_ids, **args)
    assert len(actual) == len(expected) == count
    for got, want in zip(actual, expected, strict=True):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.length_chunks == want.length_chunks
        assert got.sample_index == want.sample_index
        np.testing.assert_allclose(got.endpoint_error, want.endpoint_error, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got.chunks, want.chunks, rtol=1e-6, atol=1e-6)
