import numpy as np

from graphcritic.bridges.insert import IterativeBridgeGraph, insert_validated_bridges
from experiments.eval_bridges import cross_episode_knn_pairs
from experiments.v07_runtime import make_and_evaluate


def test_single_edge_decrease_matches_full_recompute_without_bridge_limit():
    graph = IterativeBridgeGraph(
        n_nodes=5,
        goals=np.asarray([4]),
        base_src=np.asarray([0, 1, 2, 3]),
        base_dst=np.asarray([1, 4, 3, 4]),
        base_weight=np.asarray([5.0, 5.0, 5.0, 5.0]),
    )
    assert graph.cost[0] == 10.0
    assert graph.decrease_update(0, 4, 2.0, b_max=100)
    expected_cost, expected_b = graph.full_recompute(b_max=100)
    np.testing.assert_allclose(graph.cost, expected_cost)
    np.testing.assert_array_equal(graph.bridge_count, expected_b)
    assert graph.bridge_count[0] == 1


def test_random_validated_edge_updates_match_full_recomputation():
    rng = np.random.default_rng(2)
    n = 14
    src = np.arange(n - 1)
    dst = src + 1
    graph = IterativeBridgeGraph(
        n_nodes=n, goals=np.asarray([n - 1]), base_src=src, base_dst=dst,
        base_weight=np.full(n - 1, 8.0),
    )
    candidates = [(int(a), int(b)) for a, b in zip(
        rng.integers(0, n - 1, 30), rng.integers(1, n, 30), strict=True
    ) if a != b]
    for source, target in candidates:
        if graph.cost[target] < 5:
            continue
        graph.decrease_update(source, target, float(rng.uniform(1.0, 5.0)), b_max=100)
        cost, bridges = graph.full_recompute(b_max=100)
        np.testing.assert_allclose(graph.cost, cost)
        np.testing.assert_array_equal(graph.bridge_count, bridges)


def test_bridge_count_cap_blocks_third_added_edge_on_any_path():
    graph = IterativeBridgeGraph(
        n_nodes=4,
        goals=np.asarray([3]),
        base_src=np.asarray([0, 1, 2]),
        base_dst=np.asarray([1, 2, 3]),
        base_weight=np.asarray([10.0, 10.0, 10.0]),
    )
    assert graph.decrease_update(1, 3, 2.0, b_max=2)
    assert graph.decrease_update(0, 1, 2.0, b_max=2)
    assert graph.bridge_count[0] == 2
    assert not graph.optimistic(2, 0, min_edge_cost=1.0, delta=0.0, b_max=2)
    assert not graph.decrease_update(2, 0, 1.0, b_max=2)
    assert np.all(graph.bridge_count <= 2)


def test_bridge_can_reach_a_disconnected_source_from_a_finite_cost_target():
    graph = IterativeBridgeGraph(
        n_nodes=3,
        goals=np.asarray([2]),
        base_src=np.asarray([1]),
        base_dst=np.asarray([2]),
        base_weight=np.asarray([1.0]),
    )
    assert not np.isfinite(graph.cost[0])
    assert graph.optimistic(0, 1, min_edge_cost=1.0, delta=5.0, b_max=2)
    assert graph.decrease_update(0, 1, 1.0, b_max=2)
    assert graph.cost[0] == 2.0
    assert graph.bridge_count[0] == 1


def test_sequential_insertion_rechecks_before_calling_model():
    from types import SimpleNamespace

    graph = IterativeBridgeGraph(
        n_nodes=4, goals=np.asarray([2]),
        base_src=np.asarray([0, 1, 3]), base_dst=np.asarray([1, 2, 2]),
        base_weight=np.asarray([1.0, 20.0, 10.0]),
    )
    calls = []

    def validator(batch):
        calls.extend(map(tuple, np.asarray(batch).tolist()))
        return [SimpleNamespace(
            length_chunks=1, chunks=np.zeros((1, 5, 5), dtype=np.float32),
            endpoint_error=0.0, sample_index=0,
        )]

    accepted = insert_validated_bridges(
        graph, np.asarray([[0, 3], [1, 2]]), validator,
        min_edge_cost=5.0, delta=5.0, b_max=2,
    )
    assert calls == [(1, 2)]
    assert [(x["source"], x["target"]) for x in accepted] == [(1, 2)]
    assert graph.cost[0] == 6.0


def test_candidates_are_cross_episode_and_never_target_goals():
    features = np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32)
    episodes = np.asarray([0, 1, 0, 1])
    goals = np.asarray([False, False, False, True])
    pairs = cross_episode_knn_pairs(features, episodes, goals, k=1, initial_query=4)
    assert pairs.shape == (4, 2)
    assert not np.any(goals[pairs[:, 1]])
    assert np.all(episodes[pairs[:, 0]] != episodes[pairs[:, 1]])


def test_knn_expands_until_it_finds_k_cross_episode_neighbors():
    same_episode = np.linspace(0.0, 0.001, 200, dtype=np.float32)
    other_episode = np.linspace(1.0, 2.0, 100, dtype=np.float32)
    features = np.concatenate((same_episode, other_episode))[:, None]
    episodes = np.concatenate((np.zeros(200, dtype=np.int64), np.ones(100, dtype=np.int64)))
    pairs = cross_episode_knn_pairs(
        features, episodes, np.zeros(300, dtype=np.bool_),
        k=32, initial_query=8, block_size=32, node_limit=1,
    )
    source_zero = pairs[pairs[:, 0] == 0]
    assert source_zero.shape == (32, 2)
    assert np.all(episodes[source_zero[:, 1]] == 1)


def test_make_and_evaluate_accepts_model_validated_target():
    import jax.numpy as jnp

    class ZeroPolicy:
        @staticmethod
        def apply(variables, x):
            del variables
            return jnp.zeros((x.shape[0], 5), dtype=jnp.float32)

    class HoldTransition:
        @staticmethod
        def apply(variables, x):
            del variables
            return jnp.concatenate((jnp.zeros((x.shape[0], 1)), -jnp.ones((x.shape[0], 1))), axis=1)

    states = np.asarray([[0.0, 0.0]], dtype=np.float32)
    plans = make_and_evaluate(
        states, states, np.asarray([3]), np.asarray([7]),
        policy=(ZeroPolicy(), {}), transition=(HoldTransition(), {}),
        epsilon_z=np.asarray([1.0]), state_std=np.asarray([1.0]),
        delta_std=np.asarray([1.0]), n_cont=1, n_bits=1,
        h=1, max_chunks=1, n_samples=2, candidate_batch=1,
    )
    assert plans[0] is not None
    assert plans[0].length_chunks == 1
    assert plans[0].chunks.shape == (1, 1, 5)
