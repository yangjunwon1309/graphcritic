"""Deterministic tests for state features, radius nodes, and edge aggregation."""

import numpy as np

from graphcritic.config import FeatureConfig, GraphConfig
from graphcritic.edges import build_edges
from graphcritic.featurize import Standardizer, state_table
from graphcritic.nodes import build_nodes
from graphcritic.types import TransitionSet


def _pack_episode(states, *, goal=None):
    states = np.asarray(states, dtype=np.float32).reshape(-1, 1)
    n = states.shape[0] - 1
    success_cur = (states[:-1, 0] == goal) if goal is not None else np.zeros(n, dtype=np.bool_)
    success = (states[1:, 0] == goal) if goal is not None else np.zeros(n, dtype=np.bool_)
    terminal = np.zeros(n, dtype=np.bool_)
    terminal[-1] = True
    return TransitionSet(
        feat=states[:-1],
        next_feat=states[1:],
        episode_id=np.zeros(n, dtype=np.int64),
        t=np.arange(n, dtype=np.int64),
        success=success,
        success_cur=success_cur,
        terminal=terminal,
        row_idx=np.arange(n, dtype=np.int64),
    )


def test_state_table_and_standardizer_match_terminal_state_schema():
    trans = _pack_episode([0.0, 1.0, 2.0])
    x, ref, success = state_table(trans)
    assert x.shape == (trans.n_rows + 1, 1)
    np.testing.assert_array_equal(ref, [[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(success, [False, False, False])

    fitted = Standardizer.fit(trans, FeatureConfig(standardize=True))
    np.testing.assert_allclose(fitted.mean, np.mean(x, axis=0))
    np.testing.assert_allclose(fitted.std, np.std(x, axis=0))
    np.testing.assert_allclose(fitted.transform(x).mean(axis=0), [0.0], atol=1e-7)
    identity = Standardizer.fit(trans, FeatureConfig(standardize=False))
    np.testing.assert_array_equal(identity.mean, [0.0])
    np.testing.assert_array_equal(identity.std, [1.0])


def test_greedy_radius_clusters_are_bounded_deterministic_and_keep_raw_reps():
    trans = _pack_episode([0.0, 0.09, 0.18, 0.27])
    cfg = GraphConfig(radius=0.1, feature=FeatureConfig(standardize=True), compress=True)
    standardizer = Standardizer.fit(trans, cfg.feature)
    x, _, _ = state_table(trans)
    z = standardizer.transform(x)

    nodes = build_nodes(trans, cfg, standardizer)
    repeated = build_nodes(trans, cfg, standardizer)
    for field in ("rep_feat", "rep_ref", "assign_cur", "assign_next", "is_goal"):
        np.testing.assert_array_equal(getattr(nodes, field), getattr(repeated, field))

    state_assignment = np.concatenate((nodes.assign_cur, nodes.assign_next[trans.terminal]))
    for node in range(nodes.n_nodes):
        members = np.flatnonzero(state_assignment == node)
        leader = members[0]
        assert np.all(np.abs(z[members] - z[leader]) <= cfg.radius + 1e-7)
    for node_id, (row, role) in enumerate(nodes.rep_ref):
        expected = trans.feat[row] if role == 0 else trans.next_feat[row]
        np.testing.assert_array_equal(nodes.rep_feat[node_id], expected)
    nodes.validate(trans)


def test_success_and_non_success_occurrences_at_same_coordinate_stay_separate():
    trans = TransitionSet(
        feat=[[0.0], [1.0]],
        next_feat=[[1.0], [2.0]],
        episode_id=[0, 1],
        t=[0, 0],
        success=[False, False],
        success_cur=[False, True],
        terminal=[True, True],
        row_idx=[0, 1],
    )
    cfg = GraphConfig(radius=2.0, compress=False)
    nodes = build_nodes(trans, cfg, Standardizer.fit(trans, cfg.feature))
    non_goal_next = nodes.assign_next[0]
    successful_current = nodes.assign_cur[1]
    assert non_goal_next != successful_current
    assert not nodes.is_goal[non_goal_next]
    assert nodes.is_goal[successful_current]
    nodes.validate(trans)


def test_compressed_and_uncompressed_edge_occurrences():
    # State assignments are [0, 0, 0, 1, 1, 2], so transition changes
    # have source dwell durations 3, 2, and 1.
    trans = _pack_episode([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0], goal=3.0)
    cfg = GraphConfig(radius=0.1, compress=True, weight_policy="time")
    stdz = Standardizer.fit(trans, cfg.feature)
    nodes = build_nodes(trans, cfg, stdz)
    edges = build_edges(trans, nodes, cfg)
    edges.validate(nodes.n_nodes)
    np.testing.assert_array_equal(edges.src, [0, 1, 2])
    np.testing.assert_array_equal(edges.dst, [1, 2, 3])
    np.testing.assert_array_equal(edges.kappa, [3, 2, 1])
    np.testing.assert_array_equal(edges.n_pos, [1, 1, 1])
    np.testing.assert_array_equal(edges.n_neg, [0, 0, 0])
    np.testing.assert_array_equal(edges.kind, [0, 0, 0])

    uncompressed_cfg = GraphConfig(radius=0.1, compress=False, weight_policy="time")
    uncompressed = build_edges(trans, nodes, uncompressed_cfg)
    np.testing.assert_array_equal(uncompressed.src, edges.src)
    np.testing.assert_array_equal(uncompressed.dst, edges.dst)
    np.testing.assert_array_equal(uncompressed.kappa, [1, 1, 1])
