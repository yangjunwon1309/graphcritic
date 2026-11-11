"""v1.2 same-object bridge and bridge-budget regression tests."""

import numpy as np

from experiments.build_judged_graph import _parse
from experiments.train_action_quasimetric import (
    flatten_state_windows, gather_endpoint_windows,
)
from graphcritic.bridges.insert import (
    layered_reverse_dijkstra,
    used_bridge_mask_for_budget,
)
from graphcritic.bridges.object_state import same_object_candidate_subset
from graphcritic.bridges.sweep import summarize_budget_sweep


def test_same_object_filter_only_keeps_zero_change_pairs_with_or_without_cache():
    boards = np.asarray([[0, 0], [0, 0], [1, 0], [1, 1]], dtype=np.uint8)
    pairs = np.asarray([[0, 1], [0, 2], [2, 3], [3, 3]], dtype=np.int64)
    packed_delta = np.asarray([0, 1, 2, 0], dtype=np.uint64)
    selected_cached, cached_mask = same_object_candidate_subset(
        pairs, boards, pair_delta=packed_delta,
    )
    selected_computed, computed_mask = same_object_candidate_subset(pairs, boards)
    np.testing.assert_array_equal(cached_mask, [True, False, False, True])
    np.testing.assert_array_equal(computed_mask, cached_mask)
    np.testing.assert_array_equal(selected_cached, selected_computed)
    np.testing.assert_array_equal(selected_computed, pairs[[0, 3]])


def test_one_wide_layered_solve_matches_each_budget_and_prunes_selected_paths():
    n = 7
    goals = np.asarray([6], dtype=np.int64)
    base_src = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    base_dst = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    base_weight = np.ones(6, dtype=np.float64)
    bridge_src = np.asarray([0, 0, 1, 2, 4], dtype=np.int64)
    bridge_dst = np.asarray([6, 4, 5, 6, 6], dtype=np.int64)
    bridge_weight = np.asarray([8, 1, 1, 2, 3], dtype=np.float64)
    wide = layered_reverse_dijkstra(
        n, goals, base_src, base_dst, base_weight,
        bridge_src, bridge_dst, bridge_weight, b_max=4,
    )
    default_b2 = layered_reverse_dijkstra(
        n, goals, base_src, base_dst, base_weight,
        bridge_src, bridge_dst, bridge_weight, b_max=2,
    )
    legacy_b2 = used_bridge_mask_for_budget(
        default_b2, bridge_src, bridge_dst, n_nodes=n, budget=2,
    )
    np.testing.assert_array_equal(legacy_b2, default_b2.used_bridge_mask)
    for budget in range(5):
        separate = layered_reverse_dijkstra(
            n, goals, base_src, base_dst, base_weight,
            bridge_src, bridge_dst, bridge_weight, b_max=budget,
        )
        np.testing.assert_allclose(wide.cost_steps[budget], separate.cost_steps[budget])
        keep = used_bridge_mask_for_budget(
            wide, bridge_src, bridge_dst, n_nodes=n, budget=budget,
        )
        pruned = layered_reverse_dijkstra(
            n, goals, base_src, base_dst, base_weight,
            bridge_src[keep], bridge_dst[keep], bridge_weight[keep], b_max=budget,
        )
        np.testing.assert_allclose(pruned.cost_steps[budget], wide.cost_steps[budget])


def test_b_star_is_first_budget_within_half_percentage_point_of_saturation():
    costs = np.asarray([
        [0, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [0, 0, 0, 0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [0, 0, 0, 0, 0, 0, np.inf, np.inf, np.inf, np.inf],
        [0, 0, 0, 0, 0, 0, 0, 0, np.inf, np.inf],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    result = summarize_budget_sweep(costs)
    assert result["r_sat"] == 1.0
    assert result["b_star"] == 4
    assert result["budgets"][0]["reachability"] == 0.2
    assert result["budgets"][4]["cost_p50"] == 0.0


def test_builder_legacy_defaults_remain_any_and_two_bridges():
    args = _parse([
        "--judge", "quasimetric", "--g0-graph", "g0.npz",
        "--transitions", "transitions.npz", "--sim-state", "sim_state.npz",
        "--checkpoint", "checkpoint", "--candidate-cache", "candidates.npz",
        "--out", "out",
    ])
    assert args.bridge_object_change == "any"
    assert args.b_max == 2


def test_action_head_endpoint_ids_are_gathered_as_flattened_state_windows():
    windows = np.arange(6 * 15, dtype=np.float32).reshape(6, 15)
    endpoints = gather_endpoint_windows(windows, np.asarray([4, 1, 5]))
    assert endpoints.shape == (3, 15)
    np.testing.assert_array_equal(endpoints, windows[[4, 1, 5]])
    np.testing.assert_array_equal(flatten_state_windows(endpoints), windows[[4, 1, 5]])
