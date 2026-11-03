"""Acceptance checks from docs/spec.md section 4.4."""

import numpy as np

from envs.toy import make_chain, make_grid
from graphcritic.config import GraphConfig, PessimismConfig
from graphcritic.pipeline import build_artifact
from graphcritic.stats import beta_lcb
from graphcritic.types import EdgeSet, NodeSet
from graphcritic.solve import solve
from graphcritic.weights import step_cost


def test_boost_hop_cost_equivalence():
    rng = np.random.default_rng(19)
    n_nodes = 40
    src = []
    dst = []
    for u in range(n_nodes - 1):
        # Every node has a path to the terminal goal; extra edges form a DAG.
        candidates = np.arange(u + 1, n_nodes)
        chosen = candidates[rng.random(candidates.size) < 0.12]
        if u + 1 not in chosen:
            chosen = np.r_[u + 1, chosen]
        src.extend([u] * len(chosen))
        dst.extend(chosen.tolist())
    edges = EdgeSet(
        src=src,
        dst=dst,
        kappa=np.ones(len(src), dtype=np.int64),
        n_pos=np.ones(len(src), dtype=np.int64),
        n_neg=np.zeros(len(src), dtype=np.int64),
        kind=np.zeros(len(src), dtype=np.int8),
    )
    nodes = NodeSet(
        rep_feat=np.arange(n_nodes, dtype=np.float32)[:, None],
        rep_ref=np.column_stack((np.full(n_nodes, -1), np.zeros(n_nodes))).astype(np.int64),
        assign_cur=np.empty(0, dtype=np.int64),
        assign_next=np.empty(0, dtype=np.int64),
        is_goal=np.arange(n_nodes) == n_nodes - 1,
    )
    cost, _ = solve(nodes, edges, np.ones(edges.n_edges))
    # Dynamic-programming BFS on the same directed graph is BOOST's hop cost.
    expected = np.full(n_nodes, np.inf)
    expected[-1] = 0.0
    src_array, dst_array = np.asarray(src), np.asarray(dst)
    for u in range(n_nodes - 2, -1, -1):
        expected[u] = 1.0 + np.min(expected[dst_array[src_array == u]])
    np.testing.assert_array_equal(cost, expected)


def test_toy_chain_and_grid_costs():
    chain, chain_info = make_chain(L=14, n_episodes=12, p_forward=1.0, seed=5)
    cfg = GraphConfig(gamma=0.99, radius=0.25, compress=False, weight_policy="time")
    chain_graph = build_artifact(chain, cfg, dataset_id="chain", source="test")
    c = step_cost(cfg.gamma)
    for node, feature in enumerate(chain_graph.nodes.rep_feat[:, 0]):
        np.testing.assert_allclose(
            chain_graph.cost[node], c * chain_info["d_star"][(int(feature),)], atol=1e-9
        )

    grid, grid_info = make_grid(W=8, H=7, n_episodes=150, eps=0.0, seed=6)
    grid_graph = build_artifact(grid, cfg, dataset_id="grid", source="test")
    visited = set(map(tuple, grid.feat.astype(np.int64)))
    visited.update(map(tuple, grid.next_feat[grid.terminal].astype(np.int64)))
    for node, feature in enumerate(grid_graph.nodes.rep_feat.astype(np.int64)):
        coord = tuple(map(int, feature))
        if coord in visited:
            np.testing.assert_allclose(
                grid_graph.cost[node] / c, grid_info["d_star"][coord], atol=1e-9
            )


def test_lcb_closed_form_matches_beta_quantile():
    level = 1e-3
    for prior_alpha in (0.5, 1.0, 2.0):
        for n_pos in (0, 1, 10, 100):
            ell = beta_lcb([n_pos], [0], (prior_alpha, 1.0), level)[0]
            expected = -np.log(level) / (prior_alpha + n_pos)
            np.testing.assert_allclose(-np.log(ell), expected, rtol=1e-10, atol=0.0)


def test_artifact_hash_is_deterministic():
    trans, _ = make_chain(L=9, n_episodes=5, p_forward=1.0, seed=13)
    cfg = GraphConfig(radius=0.25, compress=False, weight_policy="time")
    first = build_artifact(trans, cfg, dataset_id="same-data", source="test")
    second = build_artifact(trans, cfg, dataset_id="same-data", source="test")
    assert first.content_hash == second.content_hash


def test_numpy_126_and_2_matrix_smoke():
    # Run this same test in both `core` (NumPy 1.26) and `core-numpy2`.
    major = int(np.__version__.split(".", maxsplit=1)[0])
    assert major in (1, 2)
    values = np.asarray([1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(np.cumsum(values), [1, 3, 6])
