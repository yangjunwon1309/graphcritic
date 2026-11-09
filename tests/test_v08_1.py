"""Focused tests for the v0.8.1 bridge penalty and puzzle board audit."""

import numpy as np
import pytest
from types import SimpleNamespace

from experiments.build_judged_graph import (
    _apply_bridge_lambda,
    _make_artifact,
)
from experiments.tmp_board_check import (
    _board_change_intervals,
    _minimum_press_count,
    _node_state_indices,
    analyze_bridge_arrays,
)
from graphcritic.config import BridgeConfig, FeatureConfig, GraphConfig
from graphcritic.io import save_graph, save_sim_state, save_transitions
from graphcritic.pipeline import build_artifact
from graphcritic.types import EdgeSet, GraphArtifact, NodeSet, SimStateSet, TransitionSet
from graphcritic.weights import step_cost


def _base_graph() -> GraphArtifact:
    return GraphArtifact(
        config=GraphConfig(
            graph_kind="node", feature=FeatureConfig(standardize=False)
        ),
        feature_stats={
            "mean": np.array([0.0], dtype=np.float32),
            "std": np.array([1.0], dtype=np.float32),
        },
        nodes=NodeSet(
            rep_feat=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
            rep_ref=np.array([[-1, 0], [-1, 0], [-1, 0]], dtype=np.int64),
            assign_cur=np.empty(0, dtype=np.int64),
            assign_next=np.empty(0, dtype=np.int64),
            is_goal=np.array([False, False, True]),
        ),
        edges=EdgeSet(
            src=np.array([0, 1]), dst=np.array([1, 2]),
            kappa=np.ones(2, dtype=np.int64), n_pos=np.ones(2, dtype=np.int64),
            n_neg=np.zeros(2, dtype=np.int64), kind=np.zeros(2, dtype=np.int8),
        ),
        weights=np.array([1.0, 1.0]),
        cost=np.array([2.0, 1.0, 0.0]),
        succ=np.array([1, 2, -1]),
        meta={
            "dataset_id": "lambda-test", "source": "test",
            "core_version": "0.5.0", "n_rows": 0, "c_max_steps": 3,
        },
    )


def _make_test_artifact(bridge_cost: float, bridge_distance: float, lam: float):
    return _make_artifact(
        _base_graph(), np.array([0]), np.array([2]),
        np.array([bridge_cost]), np.array([bridge_distance]),
        np.array([3.0, 1.0, 0.0]), np.array([1, 2, -1]),
        judge_name="quasimetric", model_metadata={"params_sha256": "abc"},
        candidate_rule="test", window_size=5, bridge_lambda_steps=lam,
    )


def test_zero_lambda_preserves_the_v08_graph_hash():
    artifact = _make_test_artifact(2.5, 2.5, 0.0)
    assert artifact.content_hash == "5122c919d2bca72b4e3b8003d2cb3c782340b0256c79eb4fb76bf0219fa00012"
    assert artifact.meta["bridge_lambda_steps"] == 0.0
    assert artifact.edges.dist[-1] == pytest.approx(2.5)


def test_positive_lambda_penalizes_cost_but_leaves_quasimetric_distance_raw():
    raw = np.array([2.5, 20.0], dtype=np.float32)
    penalized = _apply_bridge_lambda(raw, 10.5)
    np.testing.assert_array_equal(raw, [2.5, 20.0])
    np.testing.assert_allclose(penalized, [13.0, 30.5], rtol=0, atol=0)

    artifact = _make_test_artifact(13.0, 2.5, 10.5)
    assert artifact.meta["bridge_lambda_steps"] == 10.5
    assert artifact.meta["bridge_cost_definition"] == (
        "quasimetric_distance_steps_plus_bridge_lambda_steps"
    )
    assert artifact.edges.kappa[-1] == 13
    assert artifact.edges.dist[-1] == pytest.approx(2.5)
    assert artifact.weights[-1] == pytest.approx(13.0 * step_cost(artifact.config.gamma))


@pytest.mark.parametrize("lam", [-0.1, float("nan"), float("inf")])
def test_bridge_lambda_rejects_negative_or_non_finite_values(lam):
    with pytest.raises(ValueError, match="finite and non-negative"):
        _apply_bridge_lambda(np.array([1.0]), lam)


def test_minimum_press_count_uses_nullspace_minimum_and_reports_unsolvable_board():
    matrix = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    memo = {}
    assert _minimum_press_count(np.array([1, 1]), matrix, memo) == 1
    assert _minimum_press_count(np.array([0, 1]), matrix, memo) is None
    assert _minimum_press_count(np.array([1, 1]), matrix, memo) == 1
    assert len(memo) == 2

    matrix_with_nullspace = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert _minimum_press_count(np.array([1, 1]), matrix_with_nullspace) == 1


def _two_episode_transition_set() -> TransitionSet:
    feat = np.arange(6, dtype=np.float32)[:, None]
    next_feat = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float32)
    terminal = np.array([False, False, True, False, False, True])
    return TransitionSet(
        feat=feat,
        next_feat=next_feat,
        episode_id=np.array([10, 10, 10, 20, 20, 20]),
        t=np.array([0, 1, 2, 0, 1, 2]),
        success=np.array([False, False, False, False, False, True]),
        success_cur=np.zeros(6, dtype=np.bool_),
        terminal=terminal,
        row_idx=np.arange(6),
    )


def test_board_check_uses_within_episode_event_gaps_and_reports_impossible_bridges():
    trans = _two_episode_transition_set()
    # State-table order: rows 0..5, then terminal occurrences 6 and 7.
    boards = np.array([
        [0, 0], [0, 0], [1, 1], [0, 0], [1, 1], [1, 1], [0, 0], [0, 1],
    ], dtype=np.uint8)
    gaps = _board_change_intervals(trans, boards)
    np.testing.assert_array_equal(gaps, [1.0, 2.0])

    # The second button column has no effect, so [0,1] is outside the GF(2) image.
    matrix = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    report = analyze_bridge_arrays(
        np.array([0, 0, 0]), np.array([1, 2, 7]), np.array([0.5, 0.5, 1.0]),
        node_state_indices=np.arange(8), button_states=boards,
        episode=np.array([10, 10, 10, 20, 20, 20, 10, 20]),
        times=np.array([0, 1, 2, 0, 1, 2, 3, 3]),
        min_button_interval=2.0, toggle=matrix,
    )
    assert report["bridge_count"] == 3
    assert report["n_p_distribution"]["0"]["count"] == 1
    assert report["n_p_distribution"]["1"]["count"] == 1
    assert report["n_p_distribution"]["∞"]["count"] == 1
    assert report["impossible_bridge_count"] == 2
    assert report["impossible_by_n_p"]["1"]["count"] == 1
    assert report["impossible_by_n_p"]["∞"]["count"] == 1
    assert report["worst_20_impossible"][0]["n_p"] == "infinity"


def test_graph_representatives_map_to_current_next_and_terminal_state_occurrences():
    trans = _two_episode_transition_set()
    graph = SimpleNamespace(nodes=SimpleNamespace(rep_ref=np.array([
        [0, 0], [1, 1], [2, 1], [5, 1],
    ], dtype=np.int64)))
    np.testing.assert_array_equal(
        _node_state_indices(graph, trans), [0, 2, 6, 7]
    )


def test_temporary_board_check_cli_loads_saved_puzzle_graph_and_writes_json(tmp_path):
    import json

    from experiments.tmp_board_check import main
    from graphcritic.puzzle import toggle_matrix

    trans = _two_episode_transition_set()
    toggle = toggle_matrix(4, 4)
    boards = np.zeros((8, 16), dtype=np.int64)
    boards[2] = toggle[:, 0]
    boards[6] = 0
    boards[4] = toggle[:, 1]
    boards[7] = toggle[:, 1] ^ toggle[:, 2]
    sim_state = SimStateSet(
        qpos=np.zeros((8, 1), dtype=np.float32),
        qvel=np.zeros((8, 1), dtype=np.float32),
        button_states=boards,
    )
    transition_path = tmp_path / "transitions.npz"
    sim_state_path = tmp_path / "sim_state.npz"
    save_transitions(trans, transition_path, dataset_id="board-audit-test")
    save_sim_state(sim_state, sim_state_path)
    base = build_artifact(
        trans,
        GraphConfig(
            graph_kind="state", feature=FeatureConfig(standardize=False),
            mode_fn="none", bridge=BridgeConfig(enabled=False),
        ),
        dataset_id="board-audit-test", source="test", sim_state=sim_state,
    )
    step = step_cost(base.config.gamma)
    graph = _make_artifact(
        base, np.array([0]), np.array([2]), np.array([0.5]), np.array([0.5]),
        base.cost / step, base.succ, judge_name="quasimetric",
        model_metadata={"params_sha256": "test"}, candidate_rule="test",
        window_size=5,
    )
    graph_path = tmp_path / "quasimetric.npz"
    save_graph(graph, graph_path)
    report_path = tmp_path / "board_check.json"
    assert main([
        "--graph", str(graph_path), "--transitions", str(transition_path),
        "--sim-state", str(sim_state_path), "--out", str(report_path),
    ]) == 0
    report = json.loads(report_path.read_text())
    assert report["graphs"][0]["graph_hash"] == graph.content_hash
    assert report["graphs"][0]["bridge_count"] == 1
    assert report["graphs"][0]["impossible_bridge_count"] == 1
