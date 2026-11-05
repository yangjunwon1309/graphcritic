"""Numerical tests for graph edge costs, shortest paths, and targets."""

import numpy as np
import pytest

from graphcritic.config import GraphConfig, PessimismConfig, TargetConfig
from graphcritic.solve import solve
from graphcritic.stats import beta_lcb, edge_neglog_lcb, lcb_level
from graphcritic.sft_weights import sft_weights
from graphcritic.targets import critic_targets
from graphcritic.types import EdgeSet, NodeSet, SchemaError
from graphcritic.weights import edge_weights, step_cost
from graphcritic.pipeline import build_artifact
from envs.toy import make_chain


def _nodes(n: int, goals: list[int]) -> NodeSet:
    return NodeSet(
        rep_feat=np.arange(n, dtype=np.float32)[:, None],
        rep_ref=np.column_stack((np.full(n, -1), np.zeros(n))).astype(np.int64),
        assign_cur=np.empty(0, dtype=np.int64),
        assign_next=np.empty(0, dtype=np.int64),
        is_goal=np.isin(np.arange(n), goals),
    )


def _edges(src, dst, kind=None, n_pos=None) -> EdgeSet:
    n = len(src)
    return EdgeSet(
        src=src,
        dst=dst,
        kappa=np.ones(n, dtype=np.int64),
        n_pos=np.ones(n, dtype=np.int64) if n_pos is None else n_pos,
        n_neg=np.zeros(n, dtype=np.int64),
        kind=np.zeros(n, dtype=np.int8) if kind is None else kind,
        anchor=np.where(np.asarray(np.zeros(n, dtype=np.int8) if kind is None else kind) == 1, 0, -1),
    )


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("n_pos", [0, 1, 10, 100])
def test_beta_lcb_closed_form(alpha, n_pos):
    level = 1e-3
    ell = beta_lcb([n_pos], [0], (alpha, 1.0), level)[0]
    expected = -np.log(level) / (alpha + n_pos)
    np.testing.assert_allclose(-np.log(ell), expected, rtol=1e-10, atol=0.0)


def test_lcb_level_is_graph_wide_and_quantiles_are_monotone():
    cfg = PessimismConfig(mode="union", delta=0.03)
    assert lcb_level(cfg, 5) == pytest.approx(0.006)
    assert lcb_level(cfg, 0) == pytest.approx(0.03)
    with pytest.raises(SchemaError):
        lcb_level(cfg, -1)

    levels = beta_lcb([0, 1, 10], [0, 0, 0], (1.0, 1.0), 1e-3)
    assert np.all(np.diff(levels) > 0)
    less_success = beta_lcb([10, 10], [0, 1], (1.0, 1.0), 1e-3)
    assert less_success[1] < less_success[0]


def test_edge_weight_policies_and_large_sample_limit():
    edges = _edges([0, 1], [1, 2], n_pos=[10, 10])
    gamma = 0.97
    c = step_cost(gamma)
    time_cfg = GraphConfig(graph_kind="node", gamma=gamma, weight_policy="time")
    hop_cfg = GraphConfig(graph_kind="node", gamma=gamma, weight_policy="hop")
    lcb_cfg = GraphConfig(graph_kind="node", gamma=gamma, weight_policy="lcb")
    np.testing.assert_array_equal(edge_weights(edges, hop_cfg), [1.0, 1.0])
    np.testing.assert_array_equal(edge_weights(edges, time_cfg), [c, c])
    assert np.all(edge_weights(edges, lcb_cfg) > edge_weights(edges, time_cfg))

    many = _edges([0], [1], n_pos=[1_000_000_000])
    ratio = edge_weights(many, GraphConfig(graph_kind="node", gamma=gamma, weight_policy="lcb"))[0] / c
    assert ratio - 1.0 < 1e-6


def test_reverse_dijkstra_chain_duplicates_isolates_and_no_goal():
    nodes = _nodes(5, [3])
    edges = _edges([0, 1, 2], [1, 2, 3])
    weights = np.array([2.0, 3.0, 4.0])
    cost, succ = solve(nodes, edges, weights)
    np.testing.assert_allclose(cost[:4], [9.0, 7.0, 4.0, 0.0])
    np.testing.assert_array_equal(succ[:4], [1, 2, 3, -1])
    assert np.isinf(cost[4]) and succ[4] == -1

    duplicate_edges = _edges([0, 0, 1], [1, 1, 2], kind=[0, 1, 0])
    duplicate_nodes = _nodes(4, [2])
    duplicate_cost, duplicate_succ = solve(duplicate_nodes, duplicate_edges, [5.0, 2.0, 1.0])
    assert duplicate_cost[0] == pytest.approx(3.0)
    assert duplicate_succ[0] == 1
    assert np.isinf(duplicate_cost[3]) and duplicate_succ[3] == -1

    no_goal_nodes = _nodes(2, [])
    with pytest.warns(RuntimeWarning, match="no goal"):
        no_goal_cost, no_goal_succ = solve(no_goal_nodes, _edges([0], [1]), [1.0])
    assert np.all(np.isinf(no_goal_cost))
    np.testing.assert_array_equal(no_goal_succ, [-1, -1])


def _chain_artifact():
    trans, info = make_chain(L=8, n_episodes=3, p_forward=1.0, seed=2)
    cfg = GraphConfig(graph_kind="node", radius=0.25, compress=False, weight_policy="time")
    artifact = build_artifact(trans, cfg, dataset_id="chain-test", source="test")
    return trans, info, cfg, artifact


def _brute_targets(trans, artifact, h, unit):
    c = step_cost(artifact.config.gamma)
    cap = float(artifact.meta["c_max_steps"])
    y = np.zeros(trans.n_rows, dtype=np.float64)
    valid = np.zeros(trans.n_rows, dtype=np.bool_)
    for i in range(trans.n_rows):
        end = i + h
        if end > trans.n_rows or np.any(trans.terminal[i : end - 1]):
            continue
        valid[i] = True
        if trans.success_cur[i]:
            steps = 0.0
        else:
            hits = np.flatnonzero(trans.success[i:end])
            if hits.size:
                steps = float(hits[0] + 1)
            else:
                node = artifact.nodes.assign_next[end - 1]
                steps = h + min(float(artifact.cost[node]) / c, cap)
        y[i] = steps * c if unit == "nats" else steps
    return y, valid


@pytest.mark.parametrize("h", [1, 3, 5])
@pytest.mark.parametrize("unit", ["steps", "nats"])
def test_targets_match_row_loop_and_qc_validity(h, unit):
    trans, _, _, artifact = _chain_artifact()
    targets = critic_targets(trans, artifact, TargetConfig(h=h, unit=unit, kind="cost"))
    expected_y, expected_valid = _brute_targets(trans, artifact, h, unit)
    np.testing.assert_array_equal(targets.valid, expected_valid)
    np.testing.assert_allclose(targets.y, expected_y, rtol=1e-6, atol=1e-6)
    targets.validate(trans)
    assert np.all(targets.y[~targets.valid] == 0.0)
    assert np.all(np.isfinite(targets.y))

    qc_valid = np.zeros(trans.n_rows, dtype=np.bool_)
    for i in range(trans.n_rows):
        if i + h <= trans.n_rows:
            qc_valid[i] = not np.any(trans.terminal[i : i + h - 1])
    np.testing.assert_array_equal(targets.valid, qc_valid)


def test_sft_weights_zero_invalid_bounded_and_one_on_shortest_chain():
    trans, _, _, artifact = _chain_artifact()
    targets = critic_targets(trans, artifact, TargetConfig(h=3, unit="steps", kind="cost"))
    weights = sft_weights(trans, artifact, targets, beta=2.0, w_max=7.0)
    assert np.all(weights[~targets.valid] == 0.0)
    assert np.max(weights) <= 7.0
    np.testing.assert_allclose(weights[targets.valid], 1.0, atol=1e-6)

    with pytest.raises(SchemaError, match="different graph"):
        other = build_artifact(
            trans,
            GraphConfig(graph_kind="node", radius=0.2, compress=False, weight_policy="time"),
            dataset_id="chain-test",
            source="test",
        )
        sft_weights(trans, other, targets, beta=1.0, w_max=2.0)
