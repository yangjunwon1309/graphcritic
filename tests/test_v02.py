"""Acceptance checks for the occurrence-level v0.2 graph schema."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from graphcritic.config import BridgeConfig, BridgeCostConfig, GraphConfig, TargetConfig
from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.io import load_graph, load_sim_state, load_targets, load_transitions, save_graph, save_sim_state, save_targets, save_transitions
from graphcritic.pipeline import build_artifact
from graphcritic.segments import contract_segments
from graphcritic.targets import critic_targets
from graphcritic.types import SchemaError, SimStateSet, TransitionSet
from graphcritic.weights import edge_weights, step_cost


def _trajectories(*, second_mode: int = 0) -> TransitionSet:
    a = np.asarray([[-4.0, 0.0], [-3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0],
                    [0.3, 0.0], [1.3, 0.0], [2.3, 0.0], [3.3, 0.0],
                    [4.3, 0.0], [5.3, 0.0]], dtype=np.float32)
    b = np.asarray([[float(i), 0.0] for i in range(9)], dtype=np.float32)
    episodes = (a, b)
    features, next_features, episode_ids, times, success, terminal, modes, actions = [], [], [], [], [], [], [], []
    offset = 0
    for ep, states in enumerate(episodes):
        current, nxt = states[:-1], states[1:]
        n = len(current)
        features.append(current)
        next_features.append(nxt)
        episode_ids.append(np.full(n, ep, dtype=np.int64))
        times.append(np.arange(n, dtype=np.int64))
        success.append(np.all(nxt == np.asarray([8.0, 0.0], dtype=np.float32), axis=1))
        done = np.zeros(n, dtype=np.bool_)
        done[-1] = True
        terminal.append(done)
        mode = second_mode if ep == 1 else 0
        modes.append(np.full(n, mode, dtype=np.int64))
        actions.append(nxt - current)
        offset += n
    mode = np.concatenate(modes)
    mode_next = mode.copy()
    terminal_all = np.concatenate(terminal)
    episode_all = np.concatenate(episode_ids)
    for i in np.flatnonzero(~terminal_all):
        mode_next[i] = mode[i + 1]
    return TransitionSet(
        feat=np.concatenate(features), next_feat=np.concatenate(next_features),
        episode_id=episode_all, t=np.concatenate(times), success=np.concatenate(success),
        terminal=terminal_all, row_idx=np.arange(offset), mode=mode, mode_next=mode_next,
        action=np.concatenate(actions),
    )


def _config(*, mode_block: bool = False, policy: str = "prior") -> GraphConfig:
    return GraphConfig(
        graph_kind="state",
        mode_fn="none",
        bridge=BridgeConfig(
            k=4, radius=0.5, radius_unit="step", landing_steps=5,
            min_time_gap=20, source_stride=1, same_mode=True,
            cost=BridgeCostConfig(
                policy=policy, lambda0_steps=1.0, lambda1_steps=1.0,
                n_bins=1, action_tol=0.5, land_tol=1.0,
                min_evidence=0, lambda_cap_steps=200.0,
            ),
        ),
    )


def test_occurrence_graph_temporal_cost_and_near_bridge_cost_are_explicit():
    trans = _trajectories()
    cfg = _config()
    artifact = build_artifact(trans, cfg, dataset_id="near-stitch", source="test")
    c = step_cost(cfg.gamma)
    temporal = artifact.edges.kind == 0
    np.testing.assert_allclose(artifact.weights[temporal], c)
    assert artifact.nodes.n_nodes == trans.n_rows + int(np.count_nonzero(trans.terminal))
    assert np.all(artifact.edges.kappa[temporal] == 1)

    source = 4  # A's state 0.3 from B's v=0 state.
    expected_anchor = int(np.count_nonzero(trans.episode_id == 0))  # B's first current-state row.
    bridge = np.flatnonzero(
        (artifact.edges.kind == 1)
        & (artifact.edges.src == source)
        & (artifact.edges.anchor == expected_anchor)
    )
    assert bridge.size == 1
    edge = int(bridge[0])
    radius = cfg.bridge.radius * float(artifact.meta["d_step"])
    expected_lambda = c * (
        cfg.bridge.cost.lambda0_steps
        + cfg.bridge.cost.lambda1_steps * (float(artifact.edges.dist[edge]) / radius) ** 2
    )
    np.testing.assert_allclose(artifact.weights[edge], c * cfg.bridge.landing_steps + expected_lambda)
    assert artifact.cost[0] < np.inf


def test_mode_mismatch_blocks_a_geometrically_near_stitch():
    trans = _trajectories(second_mode=1)
    artifact = build_artifact(trans, _config(), dataset_id="mode-gate", source="test")
    a_rows = trans.episode_id == 0
    assert not np.any(np.isfinite(artifact.cost[artifact.nodes.assign_cur[a_rows]]))


def test_data_calibrated_penalty_is_bridge_only_and_success_path_is_no_worse_than_replay():
    trans = _trajectories()
    cfg = _config(policy="data_calibrated")
    artifact = build_artifact(trans, cfg, dataset_id="calibration", source="test")
    c = step_cost(cfg.gamma)
    np.testing.assert_allclose(artifact.weights[artifact.edges.kind == 0], c)
    assert np.all(artifact.edges.n_pos[artifact.edges.kind == 1] >= 0)
    for row in np.flatnonzero(trans.episode_id == 1):
        if trans.success_cur[row]:
            continue
        goal_row = int(np.flatnonzero(trans.success)[0])
        remaining = goal_row - int(row) + 1
        assert artifact.cost[artifact.nodes.assign_cur[row]] / c <= remaining + 1e-9


def test_segment_contraction_preserves_all_boundary_costs():
    trans = _trajectories()
    cfg = _config()
    artifact = build_artifact(trans, cfg, dataset_id="contract", source="test")
    _, _, _, modes = state_table(trans)
    weights = edge_weights(artifact.edges, cfg, d_step=artifact.meta["d_step"], meta=artifact.meta)
    segments = contract_segments(
        trans, artifact.nodes, artifact.edges, modes,
        max_len=cfg.segment_max_len, edge_weights=weights,
    )
    goals = np.flatnonzero(artifact.nodes.is_goal[segments.state_ids])
    reverse = csr_matrix(
        (segments.weights, (segments.dst, segments.src)),
        shape=(segments.state_ids.size, segments.state_ids.size),
    )
    compressed_cost = dijkstra(reverse, directed=True, indices=goals, min_only=True)
    np.testing.assert_allclose(
        compressed_cost,
        artifact.cost[segments.state_ids],
        atol=1e-9,
        rtol=0,
    )


def test_target_provenance_and_schema_three_sidecars_round_trip(tmp_path):
    trans = _trajectories()
    cfg = _config()
    artifact = build_artifact(trans, cfg, dataset_id="roundtrip", source="test")
    target = critic_targets(trans, artifact, TargetConfig(h=3, unit="steps", kind="cost"))
    assert set(np.unique(target.source)).issubset({0, 1, 2, 3})
    graph_path = tmp_path / "graph.npz"
    save_graph(artifact, graph_path)
    loaded = load_graph(graph_path, trans=trans)
    assert loaded.content_hash == artifact.content_hash
    target_path = tmp_path / "targets.npz"
    save_targets(target, target_path)
    assert (tmp_path / "targets_source.npy").is_file()
    np.testing.assert_array_equal(load_targets(target_path, graph=loaded).source, target.source)

    trans_path = tmp_path / "transitions.npz"
    save_transitions(trans, trans_path, dataset_id="roundtrip")
    loaded_trans, _ = load_transitions(trans_path)
    np.testing.assert_array_equal(loaded_trans.mode_next, trans.mode_next)
    np.testing.assert_array_equal(loaded_trans.action, trans.action)

    sim = SimStateSet(qpos=np.ones((trans.n_rows + 2, 3)), qvel=np.zeros((trans.n_rows + 2, 2)))
    sim_path = tmp_path / "sim_state.npz"
    save_sim_state(sim, sim_path)
    np.testing.assert_array_equal(load_sim_state(sim_path, n_states=sim.n_states).qpos, sim.qpos)
