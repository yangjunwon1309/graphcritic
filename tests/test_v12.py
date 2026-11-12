"""v1.2 same-object bridge and bridge-budget regression tests."""

import json
from types import SimpleNamespace

import numpy as np

from experiments import build_judged_graph as builder
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
from graphcritic.types import TargetSet


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


def test_target_h_defaults_to_five_and_is_written_to_target_metadata(tmp_path, monkeypatch):
    required = [
        "--judge", "quasimetric", "--g0-graph", "g0.npz",
        "--transitions", "transitions.npz", "--sim-state", "sim_state.npz",
        "--checkpoint", "checkpoint", "--candidate-cache", "candidates.npz",
        "--out", "out",
    ]
    assert _parse(required).target_h == 5
    args = _parse([*required, "--target-h", "1"])

    trans = SimpleNamespace(n_rows=3)
    targets = TargetSet(
        y=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        valid=np.ones(3, dtype=np.bool_), h=1, unit="steps",
        graph_hash="a" * 64, dataset_id="dataset:test:hash",
        source=np.ones(3, dtype=np.uint8), kind="discounted",
        loss_mask=np.ones(3, dtype=np.bool_), discount_gamma=0.99,
    )
    calls = {}

    def fake_critic_targets(actual_trans, artifact, config):
        calls["config"] = config
        assert actual_trans is trans
        return targets

    monkeypatch.setattr(builder, "critic_targets", fake_critic_targets)
    result = builder._write_graph_targets(
        trans, object(), tmp_path,
        diagnostic_oracle_np_max=None, write_oracle_targets=False,
        target_kind="discounted", discount_gamma=0.99, target_h=args.target_h,
    )

    assert result is targets
    assert calls["config"].h == 1
    metadata = json.loads((tmp_path / "targets.json").read_text())
    assert metadata["config"]["h"] == 1


def test_action_head_endpoint_ids_are_gathered_as_flattened_state_windows():
    windows = np.arange(6 * 15, dtype=np.float32).reshape(6, 15)
    endpoints = gather_endpoint_windows(windows, np.asarray([4, 1, 5]))
    assert endpoints.shape == (3, 15)
    np.testing.assert_array_equal(endpoints, windows[[4, 1, 5]])
    np.testing.assert_array_equal(flatten_state_windows(endpoints), windows[[4, 1, 5]])
