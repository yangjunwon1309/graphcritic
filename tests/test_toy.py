"""End-to-end checks on small deterministic and stitched datasets."""

import numpy as np

from envs.toy import make_chain, make_grid, make_stitch
from graphcritic.config import GraphConfig
from graphcritic.pipeline import build_artifact
from graphcritic.weights import step_cost


def _artifact(trans, *, radius=0.25, compress=False, policy="time"):
    config = GraphConfig(
        graph_kind="node",
        gamma=0.97,
        radius=radius,
        compress=compress,
        weight_policy=policy,
    )
    return build_artifact(trans, config, dataset_id="toy-test", source="test")


def test_chain_cost_matches_exact_distance_for_visited_states():
    trans, info = make_chain(L=12, n_episodes=20, p_forward=0.7, seed=14)
    artifact = _artifact(trans)
    c = step_cost(artifact.config.gamma)
    for node, feature in enumerate(artifact.nodes.rep_feat[:, 0]):
        coord = (int(feature),)
        if coord in info["d_star"]:
            np.testing.assert_allclose(
                artifact.cost[node], c * info["d_star"][coord], rtol=0.0, atol=1e-9
            )


def test_grid_cost_matches_bfs_distance_and_respects_wall_detour():
    trans, info = make_grid(W=7, H=6, n_episodes=100, eps=0.0, seed=4)
    artifact = _artifact(trans)
    c = step_cost(artifact.config.gamma)
    visited = set(map(tuple, trans.feat.astype(np.int64)))
    visited.update(map(tuple, trans.next_feat[trans.terminal].astype(np.int64)))
    for node, feature in enumerate(artifact.nodes.rep_feat.astype(np.int64)):
        coord = tuple(map(int, feature))
        if coord in visited and np.isfinite(info["d_star"].get(coord, np.inf)):
            np.testing.assert_allclose(artifact.cost[node] / c, info["d_star"][coord], atol=1e-9)

    # The default wall forces a route through its single opening at y=0.
    direct_manhattan = (7 - 1 - 0) + (6 - 1 - 3)
    assert info["d_star"][(0, 3)] > direct_manhattan


def test_stitching_connects_fragments_that_are_separate_episodes():
    trans, info = make_stitch(L=18)
    artifact = _artifact(trans)
    c = step_cost(artifact.config.gamma)
    start_rows = np.flatnonzero(trans.feat[:, 0] == 0)
    assert start_rows.size > 0
    start_node = artifact.nodes.assign_cur[start_rows[0]]
    np.testing.assert_allclose(artifact.cost[start_node], c * info["d_star"][(0,)], atol=1e-9)

    # The first fragment alone has no success-labeled transition, while the
    # union graph has a finite path across the shared midpoint.
    first_episode = trans.episode_id == trans.episode_id[0]
    assert not np.any(trans.success[first_episode])
    assert np.isfinite(artifact.cost[start_node])
