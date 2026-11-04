"""v0.3 direct travel-time bridges, path extraction, and tracker checks."""

import json

import numpy as np
import pytest

from graphcritic.config import BridgeConfig, BridgeCostConfig, FeatureConfig, GraphConfig
from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.io import load_graph, save_graph
from graphcritic.path import GraphTrackingPolicy, PathStep, extract_path
from graphcritic.pipeline import build_artifact
from graphcritic.tau_model import FrameGapModel
from graphcritic.types import SimStateSet, TransitionSet
from graphcritic.weights import step_cost
from experiments.train_tau import _calibrate_quantile
from envs.ogbench_layout import (
    SceneObservationLayout,
    robomimic_position_features,
    scene_position_features,
    verify_scene_position_layout,
)


def _transition(episodes, *, actions=None):
    feat, next_feat, episode_id, times, success, success_cur, terminal = [], [], [], [], [], [], []
    all_actions = []
    for episode, (current, nxt, is_success) in enumerate(episodes):
        current = np.asarray(current, dtype=np.float32)
        nxt = np.asarray(nxt, dtype=np.float32)
        count = current.shape[0]
        feat.append(current)
        next_feat.append(nxt)
        episode_id.append(np.full(count, episode, dtype=np.int64))
        times.append(np.arange(count, dtype=np.int64))
        labels = np.zeros(count, dtype=np.bool_)
        labels[-1] = is_success
        success.append(labels)
        cur = np.zeros(count, dtype=np.bool_)
        if count > 1:
            cur[1:] = labels[:-1]
        success_cur.append(cur)
        done = np.zeros(count, dtype=np.bool_)
        done[-1] = True
        terminal.append(done)
        if actions is None:
            all_actions.append(nxt.copy())
        else:
            all_actions.append(np.asarray(actions[episode], dtype=np.float32))
    return TransitionSet(
        feat=np.concatenate(feat), next_feat=np.concatenate(next_feat),
        episode_id=np.concatenate(episode_id), t=np.concatenate(times),
        success=np.concatenate(success), success_cur=np.concatenate(success_cur),
        terminal=np.concatenate(terminal), row_idx=np.arange(sum(len(x) for x in feat)),
        action=np.concatenate(all_actions),
    )


def _time_config(*, radius=2.0, rho=1.0, tau_max=48, lambda0=5.0):
    return GraphConfig(
        graph_kind="state", feature=FeatureConfig(standardize=False), mode_fn="none",
        bridge=BridgeConfig(
            k=4, radius=radius, radius_unit="step", min_time_gap=0,
            same_mode=True,
            cost=BridgeCostConfig(
                policy="time", tau_model="geometric", rho=rho,
                tau_max_steps=tau_max, lambda0_steps=lambda0,
            ),
        ),
    )


def _near_goal_data():
    return _transition([
        ([[1.0], [2.0]], [[2.0], [3.0]], False),
        ([[-1.0], [0.0]], [[0.0], [1.0]], True),
    ])


def test_geometric_bridge_lands_directly_on_goal_and_extract_path_cost_matches():
    trans = _near_goal_data()
    cfg = _time_config()
    graph = build_artifact(trans, cfg, dataset_id="v03-goal", source="test")
    c = step_cost(cfg.gamma)
    source = 0
    goal_vertex = int(np.flatnonzero(graph.nodes.is_goal)[0])
    bridge = np.flatnonzero(
        (graph.edges.kind == 1)
        & (graph.edges.src == source)
        & (graph.edges.dst == goal_vertex)
    )
    assert bridge.size == 1
    edge = int(bridge[0])
    assert graph.edges.anchor[edge] == goal_vertex
    assert graph.edges.kappa[edge] == 1
    np.testing.assert_allclose(graph.weights[edge], c * (1 + cfg.bridge.cost.lambda0_steps))
    path = extract_path(graph, source)
    assert path[0].vertex == source and path[0].kind == 1
    assert path[-1].vertex == goal_vertex and path[-1].kind is None
    path_cost_steps = sum(step.tau for step in path) + cfg.bridge.cost.lambda0_steps * sum(
        step.kind == 1 for step in path
    )
    np.testing.assert_allclose(graph.cost[source] / c, path_cost_steps)


def test_time_graph_preserves_monte_carlo_upper_bound_on_successful_rows():
    trans = _near_goal_data()
    cfg = _time_config()
    graph = build_artifact(trans, cfg, dataset_id="v03-mc", source="test")
    c = step_cost(cfg.gamma)
    success_rows = np.flatnonzero(trans.success)
    for event_row in success_rows:
        episode = trans.episode_id[event_row]
        start = int(np.flatnonzero(trans.episode_id == episode)[0])
        for row in range(start, int(event_row) + 1):
            if trans.episode_id[row] != episode:
                continue
            remaining = int(event_row) - row + 1
            cost = graph.cost[graph.nodes.assign_cur[row]] / c
            assert cost <= remaining + 1e-9


def test_geometric_bridges_over_tau_max_are_removed():
    trans = _transition([
        ([[0.0], [1.0]], [[1.0], [2.0]], False),
        ([[2.4], [3.4]], [[3.4], [4.4]], True),
    ])
    cfg = _time_config(radius=3.0, rho=5.0, tau_max=1)
    graph = build_artifact(trans, cfg, dataset_id="v03-tau-cap", source="test")
    assert graph.meta["bridge_tau_dropped_far"] > 0
    assert graph.edges.kind[graph.edges.kind == 1].size == 0


def test_time_graph_does_not_call_simulator_during_build(monkeypatch):
    trans = _near_goal_data()
    sim_state = SimStateSet(
        qpos=np.ones((trans.n_rows + 2, 2), dtype=np.float32),
        qvel=np.zeros((trans.n_rows + 2, 1), dtype=np.float32),
    )

    class ForbiddenVerifier:
        def __getattr__(self, name):
            raise AssertionError(f"simulator access during build: {name}")

    graph = build_artifact(
        trans, _time_config(), dataset_id="v03-offline", source="test",
        sim_state=sim_state, verifier=ForbiddenVerifier(),
    )
    assert graph.meta["bridge_cost_policy"] == "time"


def test_graph_tracker_replans_under_noise_while_open_loop_fragment_does_not():
    # The action is an absolute target. Small actuation noise is corrected by
    # selecting the nearest point on the successful trajectory each step.
    trans = _transition([
        ([[0.0]], [[0.0]], False),
        ([[0.1], [1.1], [2.1]], [[1.1], [2.1], [3.1]], True),
    ], actions=[[[0.0]], [[1.1], [2.1], [3.1]]])
    cfg = _time_config(radius=0.5, rho=1.0, lambda0=0.0)
    graph = build_artifact(trans, cfg, dataset_id="v03-tracker", source="test")
    featurizer = StateFeaturizer.fit(trans, cfg.feature)
    policy = GraphTrackingPolicy(graph, trans, featurizer, "none", window=10)
    terminal_vertices = np.flatnonzero(graph.nodes.rep_ref[:, 1] == 1)
    np.testing.assert_array_equal(
        policy._action_row[terminal_vertices], graph.nodes.rep_ref[terminal_vertices, 0]
    )
    policy.reset(0)
    assert policy.distance_to_vertex(np.asarray([0.0], dtype=np.float32), {}, 0) == 0.0
    rng = np.random.default_rng(7)
    position = 0.0
    success = False
    for _ in range(30):
        action = policy.act(np.asarray([position], dtype=np.float32), {})
        position = float(action[0]) + float(rng.normal(0.0, 0.01))
        success = position >= 2.9
        if success:
            break
    assert success
    assert policy.status()["bridge_use_count"] >= 1

    # Open-loop replay from the failed episode repeats its own non-progressing action.
    open_loop_position = 0.0
    for action in trans.action[:1]:
        open_loop_position = 0.5 * open_loop_position + 0.5 * float(action[0])
    assert open_loop_position < 2.9


def test_graph_tracker_falls_through_consecutive_bridges_when_window_has_no_action():
    trans = _transition([
        ([[0.0], [1.0]], [[1.0], [2.0]], False),
        ([[2.0], [3.0]], [[3.0], [4.0]], True),
    ])
    cfg = _time_config(radius=3.0, rho=1.0)
    graph = build_artifact(trans, cfg, dataset_id="v03-bridge-chain", source="test")
    featurizer = StateFeaturizer.fit(trans, cfg.feature)
    policy = GraphTrackingPolicy(graph, trans, featurizer, "none", window=0)
    goal = int(np.flatnonzero(graph.nodes.is_goal)[0])
    policy.reset(0)
    policy.path = [
        PathStep(0, 1, 1),
        PathStep(0, 1, 1),
        PathStep(1, 0, 1),
        PathStep(goal, None, 0),
    ]
    action = policy.act(np.asarray([0.0], dtype=np.float32), {})
    assert action.shape == trans.action.shape[1:]
    assert policy.status()["bridge_use_count"] == 2
    policy.pointer = len(policy.path)
    terminal_action = policy.act(np.asarray([3.0], dtype=np.float32), {})
    assert terminal_action.shape == trans.action.shape[1:]


def test_frame_gap_model_numpy_and_jax_match_and_checkpoint_round_trips(tmp_path):
    pytest.importorskip("jax")
    dim, hidden, horizon = 40, 512, 48
    rng = np.random.default_rng(2)
    params = {}
    for layer, fan_in, fan_out in ((0, 3 * dim, hidden), (1, hidden, hidden), (2, hidden, hidden), (3, hidden, horizon + 1)):
        params[f"w{layer}"] = rng.normal(size=(fan_in, fan_out)).astype(np.float32) * 0.1
        params[f"b{layer}"] = np.zeros(fan_out, dtype=np.float32)
        if layer < 3:
            params[f"ln_scale{layer}"] = np.ones(fan_out, dtype=np.float32)
            params[f"ln_bias{layer}"] = np.zeros(fan_out, dtype=np.float32)
    model = FrameGapModel(
        params, np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32), horizon,
        {"model": "frame_gap_mlp_v1", "horizon": horizon, "calibrated_quantiles": {}},
    )
    left = rng.normal(size=(5, dim)).astype(np.float32)
    right = rng.normal(size=(5, dim)).astype(np.float32)
    p_numpy = model.predict_proba_pairs(left, right, backend="numpy")
    p_jax = model.predict_proba_pairs(left, right, backend="jax")
    np.testing.assert_allclose(p_numpy, p_jax, rtol=0, atol=1e-5)
    path = tmp_path / "tau_model.npz"
    model.save(path)
    loaded = FrameGapModel.load(path)
    assert loaded.checkpoint_sha256 is not None
    np.testing.assert_allclose(loaded.predict_proba_pairs(left, right, backend="numpy"), p_numpy)


def test_scene_named_layout_and_qpos_derived_positions_agree():
    rng = np.random.default_rng(11)
    n = 1000
    qpos = np.zeros((n, 25), dtype=np.float64)
    qpos[:, 14:17] = rng.normal(size=(n, 3))
    quat_yaw = rng.uniform(-np.pi, np.pi, size=n)
    qpos[:, 17] = np.cos(quat_yaw / 2.0)
    qpos[:, 20] = np.sin(quat_yaw / 2.0)
    qpos[:, 23:25] = rng.uniform(-0.2, 0.2, size=(n, 2))
    obs = np.zeros((n, 40), dtype=np.float32)
    obs[:, 19:22] = ((qpos[:, 14:17] - [0.425, 0.0, 0.0]) * 10).astype(np.float32)
    obs[:, 22:26] = qpos[:, 17:21].astype(np.float32)
    obs[:, 26] = np.cos(quat_yaw)
    obs[:, 27] = np.sin(quat_yaw)
    obs[:, 36] = (qpos[:, 23] * 18).astype(np.float32)
    obs[:, 38] = (qpos[:, 24] * 15).astype(np.float32)
    layout = SceneObservationLayout.from_dimensions(40, 25)
    assert layout.dimensions()["button_0_state_one_hot"] == [28, 30]
    report = verify_scene_position_layout(obs, qpos)
    assert report["passed"]
    phi, groups, _ = scene_position_features(obs, qpos)
    assert phi.shape == (n, 13)
    assert groups == (("eef", 0, 5), ("gripper", 5, 6), ("cube", 6, 11), ("drawer", 11, 12), ("window", 12, 13))


def test_robomimic_position_features_are_quaternion_sign_invariant():
    obs = np.zeros((2, 12), dtype=np.float32)
    obs[:, :3] = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    obs[0, 3:7] = [0.7, -0.2, 0.4, 0.1]
    obs[1, 3:7] = -obs[0, 3:7]
    obs[:, 7:9] = [0.02, -0.03]
    obs[:, 9:12] = [0.4, 0.5, 0.6]
    features, groups = robomimic_position_features(obs)
    np.testing.assert_allclose(features[0], features[1])
    assert features.shape == (2, 12)
    assert groups == (("eef", 0, 7), ("gripper", 7, 9), ("object", 9, 12))


def test_learned_bridge_checkpoint_hash_is_part_of_graph_artifact(tmp_path):
    hidden, horizon = 8, 4
    params = {}
    for layer, fan_in, fan_out in ((0, 3, hidden), (1, hidden, hidden), (2, hidden, hidden), (3, hidden, horizon + 1)):
        params[f"w{layer}"] = np.zeros((fan_in, fan_out), dtype=np.float32)
        params[f"b{layer}"] = np.zeros(fan_out, dtype=np.float32)
        if layer < 3:
            params[f"ln_scale{layer}"] = np.ones(fan_out, dtype=np.float32)
            params[f"ln_bias{layer}"] = np.zeros(fan_out, dtype=np.float32)
    params["b3"][0] = 10.0
    checkpoint = tmp_path / "tau_model.npz"
    FrameGapModel(
        params, np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32), horizon,
        {"model": "frame_gap_mlp_v1", "horizon": horizon},
    ).save(checkpoint)
    trans = _near_goal_data()
    cfg = GraphConfig(
        graph_kind="state", feature=FeatureConfig(standardize=False), mode_fn="none",
        bridge=BridgeConfig(
            k=4, radius=2.0, radius_unit="step", min_time_gap=0,
            cost=BridgeCostConfig(
                policy="time", tau_model="learned", tau_ckpt=str(checkpoint),
                tau_max_steps=4,
            ),
        ),
    )
    graph = build_artifact(trans, cfg, dataset_id="v03-learned", source="test")
    assert graph.meta["tau_model_sha256"]
    graph_path = tmp_path / "graph.npz"
    save_graph(graph, graph_path)
    loaded = load_graph(graph_path, trans=trans)
    assert loaded.meta["tau_model_sha256"] == graph.meta["tau_model_sha256"]
    assert loaded.content_hash == graph.content_hash


def test_tau_conformal_calibration_targets_upper_time_coverage():
    # Labels are zero-based: label 0 means Δ=1.  For an upper quantile τ,
    # coverage is F(Δ-1) < q', not F(Δ) >= q'.
    probabilities = np.asarray([
        [0.10, 0.20, 0.20, 0.50],
        [0.20, 0.10, 0.30, 0.40],
        [0.05, 0.15, 0.20, 0.60],
        [0.40, 0.10, 0.10, 0.40],
    ], dtype=np.float64)
    labels = np.asarray([0, 1, 2, 1], dtype=np.int64)
    q_prime, calibration_coverage, target_coverage = _calibrate_quantile(
        probabilities, labels, 0.5
    )
    cdf = np.cumsum(probabilities[:, :-1], axis=1)
    scores = np.asarray([0.0, cdf[1, 0], cdf[2, 1], cdf[3, 0]])
    tau = np.argmax(cdf >= q_prime, axis=1) + 1
    assert np.mean((scores < q_prime) & ((labels + 1) <= tau)) == calibration_coverage
    assert calibration_coverage >= target_coverage
    assert calibration_coverage == 0.75
