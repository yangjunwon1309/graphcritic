"""Physical-coordinate, bridge-bound, and servo regressions for v0.4."""

import numpy as np

from graphcritic.config import BridgeConfig, BridgeCostConfig, FeatureConfig, GraphConfig
from graphcritic.pipeline import build_artifact
from graphcritic.passive_bridges import physical_tau_lower_bounds
from graphcritic.path import PathStep
from graphcritic.servo import ServoTrackingPolicy
from graphcritic.state_schema import (
    OGBenchSceneStateSchema,
    RobomimicStateSchema,
    StateCoordinates,
    estimate_passive_tolerance,
    passive_difference,
)
from graphcritic.solve import solve
from graphcritic.state_graph import _state_nodes
from graphcritic.tau_model import FrameGapModel
from graphcritic.types import EdgeSet, GraphArtifact, SimStateSet, TransitionSet
from experiments.train_tau import _episode_bootstrap_interval


def _scene_rows(n=2):
    obs = np.zeros((n, 40), dtype=np.float32)
    obs[:, 15] = 1.0
    obs[:, 17] = 3.0
    qpos = np.zeros((n, 25), dtype=np.float32)
    qpos[:, 14:17] = [0.4, 0.1, 0.02]
    qpos[:, 17] = 1.0
    qpos[:, 23] = 0.01
    qpos[:, 24] = 0.02
    obs[:, 12:15] = 0.0
    obs[:, 19:22] = (qpos[:, 14:17] - np.asarray([0.425, 0.0, 0.0])) * 10.0
    obs[:, 22] = 1.0
    obs[:, 37] = qpos[:, 23] * 18.0
    obs[:, 39] = qpos[:, 24] * 15.0
    buttons = np.zeros((n, 2), dtype=np.int64)
    return obs, qpos, buttons


def test_scene_schema_uses_physical_units_and_separates_control_from_objects():
    obs, qpos, buttons = _scene_rows()
    schema = OGBenchSceneStateSchema()
    coords = schema.coordinates(obs, {"qpos": qpos, "button_states": buttons})
    assert coords.r.shape == (2, 19)
    assert coords.q_a.shape == (2, 5)
    assert coords.o.shape == (2, 6)
    assert coords.e.shape == (2, 4)
    assert coords.q_p.shape == (2, 6)
    np.testing.assert_allclose(coords.q_a[0, :3], [0.425, 0.0, 0.0])
    np.testing.assert_allclose(coords.o[0, :3], [0.4, 0.1, 0.02])
    assert schema.kinematic_steps(coords.r[0], coords.r[1]) == 0


def test_scene_grasp_switches_cube_passive_coordinate_to_relative_frame():
    obs, qpos, buttons = _scene_rows()
    schema = OGBenchSceneStateSchema()
    obs[:, 17] = 0.0
    obs[:, 18] = 1.0
    # Put the gripper directly on the cube.
    obs[:, 12:15] = (qpos[:, 14:17] - np.asarray([0.425, 0.0, 0.0])) * 10.0
    coords = schema.coordinates(obs, {"qpos": qpos, "button_states": buttons})
    assert coords.mode[0] == coords.mode[1] > 0
    np.testing.assert_allclose(coords.q_p[:, :4], coords.e)


def test_scene_near_closed_gripper_without_contact_is_not_a_grasp_mode():
    obs, qpos, buttons = _scene_rows()
    obs[:, 17] = 0.0
    obs[:, 18] = 0.0
    obs[:, 12:15] = (qpos[:, 14:17] - np.asarray([0.425, 0.0, 0.0])) * 10.0
    mode = OGBenchSceneStateSchema().mode(obs, {"qpos": qpos, "button_states": buttons})
    np.testing.assert_array_equal(mode, np.zeros(2, dtype=np.int64))


def test_scene_passive_periodicity_handles_every_cube_yaw_dimension():
    schema = OGBenchSceneStateSchema()
    source = np.zeros(10, dtype=np.float64)
    target = source.copy()
    target[7] = np.deg2rad(89.0)
    delta = passive_difference(schema, source, target)
    assert np.isclose(delta[7], np.deg2rad(-1.0))


def test_robomimic_schema_exposes_relative_eef_pose():
    obs = np.zeros((2, 12), dtype=np.float32)
    obs[:, :3] = [[0.5, 0.0, 0.2], [0.6, 0.0, 0.2]]
    obs[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    obs[:, 7:9] = 0.0
    obs[:, 9:12] = [0.4, 0.0, 0.2]
    schema = RobomimicStateSchema()
    coords = schema.coordinates(obs)
    np.testing.assert_allclose(coords.e[0, :3], [0.1, 0.0, 0.0])
    assert coords.q_a.shape == (2, 7)
    assert schema.kinematic_steps(coords.r[0], coords.r[1]) == 2


def test_robomimic_relative_translation_uses_object_frame_not_eef_frame():
    obs = np.zeros((1, 12), dtype=np.float32)
    obs[:, :3] = [0.5, 0.0, 0.2]
    obs[:, 3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    obs[:, 9:12] = [0.4, 0.0, 0.2]
    relative = RobomimicStateSchema().relative(obs)
    np.testing.assert_allclose(relative[0, :3], [0.1, 0.0, 0.0], atol=1e-6)


def test_passive_tolerance_uses_free_space_q99_and_physical_floor():
    obs, qpos, buttons = _scene_rows(4)
    schema = OGBenchSceneStateSchema()
    coords = schema.coordinates(obs, {"qpos": qpos, "button_states": buttons})
    coords.q_p[1, 0] += 0.001
    coords.q_p[2, 0] += 0.002
    report = estimate_passive_tolerance(
        schema, coords.q_p, np.asarray([0, 0, 0, 0]), np.zeros(4, dtype=np.int64),
        coords.q_a, contact_mask=np.zeros(4, dtype=np.bool_),
    )
    assert report["eligible_steps"] == 3
    assert report["epsilon"][0] >= 0.005
    assert np.isclose(report["epsilon"][3], np.deg2rad(2.0))


def test_manipulation_tau_is_clamped_by_kinematic_settle_and_passive_speed():
    tau, kin_violation, passive_violation = physical_tau_lower_bounds(
        np.asarray([1, 8, 2]), np.asarray([4, 2, 1]), np.asarray([3, 9, 1]), settle=2,
    )
    np.testing.assert_array_equal(tau, [6, 9, 3])
    np.testing.assert_array_equal(kin_violation, [True, False, False])
    np.testing.assert_array_equal(passive_violation, [True, True, False])


def test_tau_coverage_interval_bootstraps_whole_episodes():
    interval = _episode_bootstrap_interval(
        np.ones(12, dtype=np.bool_),
        np.repeat(np.arange(3, dtype=np.int64), 4),
        seed=7,
    )
    assert interval["n_pairs"] == 12
    assert interval["n_episodes"] == 3
    assert interval["replicates"] == 1000
    assert interval["low"] == interval["high"] == 1.0
    assert interval["width"] == 0.0


def test_tau_pair_layout_includes_delta_only_for_objects_and_relative_pose(tmp_path):
    hidden, horizon = 4, 2
    params = {
        "w0": np.zeros((8, hidden), dtype=np.float32), "b0": np.zeros(hidden, dtype=np.float32),
        "w1": np.zeros((hidden, hidden), dtype=np.float32), "b1": np.zeros(hidden, dtype=np.float32),
        "w2": np.zeros((hidden, hidden), dtype=np.float32), "b2": np.zeros(hidden, dtype=np.float32),
        "w3": np.zeros((hidden, horizon + 1), dtype=np.float32), "b3": np.zeros(horizon + 1, dtype=np.float32),
    }
    for layer in range(3):
        params[f"ln_scale{layer}"] = np.ones(hidden, dtype=np.float32)
        params[f"ln_bias{layer}"] = np.zeros(hidden, dtype=np.float32)
    metadata = {"model": "frame_gap_mlp_v1", "horizon": horizon,
                "observation_dim": 3, "pair_delta_indices": [1, 2],
                "calibrated_quantiles": {"0.8": 0.8}}
    model = FrameGapModel(params, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), horizon, metadata)
    pairs = model.prepare_pairs([[1, 2, 3]], [[2, 4, 7]])
    np.testing.assert_allclose(pairs, [[1, 2, 3, 2, 4, 7, 2, 4]])
    model.save(tmp_path / "tau_model.npz")
    loaded = FrameGapModel.load(tmp_path / "tau_model.npz")
    np.testing.assert_allclose(loaded.prepare_pairs([[1, 2, 3]], [[2, 4, 7]]), pairs)


def test_passive_r_graph_builds_same_world_reposition_edges_with_physical_costs():
    current, qpos, buttons = _scene_rows(6)
    current[:, 12:15] = np.asarray([[-4.25, 0, 0], [-3.75, 0, 0], [-4.0, 0, 0], [-3.5, 0, 0], [-3.75, 0, 0], [-3.5, 0, 0]], dtype=np.float32)
    next_obs = np.asarray([current[1], current[4], current[3], current[5]], dtype=np.float32)
    trans = TransitionSet(
        feat=current[:4], next_feat=next_obs,
        episode_id=[0, 0, 1, 1], t=[0, 1, 0, 1],
        success=[False, False, False, True], success_cur=[False] * 4,
        terminal=[False, True, False, True], row_idx=[0, 1, 2, 3],
        action=np.zeros((4, 5), dtype=np.float32),
    )
    sim_state = SimStateSet(
        qpos=qpos, qvel=np.zeros_like(qpos), button_states=buttons,
    )
    cfg = GraphConfig(
        feature=FeatureConfig(name="identity"), mode_fn="ogbench_scene_v04",
        bridge=BridgeConfig(
            k=2, strategy="passive_r", state_schema="ogbench_scene",
            passive_jitter_multiplier=1.0, kinematic_settle=2,
            cost=BridgeCostConfig(policy="time", tau_model="geometric", lambda0_steps=0.0),
        ),
    )
    artifact = build_artifact(
        trans, cfg, dataset_id="scene-r-test", source="test", sim_state=sim_state,
    )
    bridge = artifact.edges.kind == 1
    assert np.count_nonzero(bridge) > 0
    assert np.all(artifact.edges.bridge_type[bridge] == 0)
    assert np.all(artifact.edges.kappa[bridge] >= 2)
    np.testing.assert_allclose(
        artifact.weights[bridge], -np.log(cfg.gamma) * artifact.edges.kappa[bridge]
    )
    assert artifact.meta["bridge_type_counts"]["M"] == 0


def _servo_fixture():
    trans = TransitionSet(
        feat=[[0.0], [1.0], [2.0]],
        next_feat=[[1.0], [1.5], [3.0]],
        episode_id=[0, 0, 1],
        t=[0, 1, 0],
        success=[False, False, True],
        success_cur=[False, False, False],
        terminal=[False, True, True],
        row_idx=[0, 1, 2],
        action=np.zeros((3, 5), dtype=np.float32),
    )
    states = np.asarray([0.0, 1.0, 2.0, 1.5, 3.0], dtype=np.float32)[:, None]
    nodes = _state_nodes(trans, states, np.asarray([False, False, False, False, True]))
    edges = EdgeSet(
        src=[0, 1, 2, 1], dst=[1, 3, 4, 2], kappa=[1, 1, 1, 2],
        n_pos=[1, 1, 1, 0], n_neg=[0, 0, 0, 0], kind=[0, 0, 0, 1],
        dist=[0.0, 0.0, 0.0, 0.0], anchor=[-1, -1, -1, 2],
        bridge_type=[-1, -1, -1, 0],
    )
    cfg = GraphConfig(graph_kind="state", feature=FeatureConfig(name="identity"))
    weights = np.asarray([1.0, 1.0, 1.0, 2.0])
    cost, succ = solve(nodes, edges, weights)
    artifact = GraphArtifact(
        config=cfg, feature_stats={"mean": [0.0], "std": [1.0]}, nodes=nodes, edges=edges,
        weights=weights, cost=cost, succ=succ,
        meta={"dataset_id": "servo-test", "source": "test", "core_version": "0.4", "n_rows": 3,
              "n_states": 5, "c_max_steps": 4, "d_step": 1.0},
    )
    qa = np.zeros((5, 5), dtype=np.float32)
    qa[1, 0] = 0.025
    qa[2, 0] = 0.05
    qa[4, 0] = 0.10
    qp = np.zeros((5, 6), dtype=np.float32)
    coords = StateCoordinates(
        r=np.zeros((5, 19), dtype=np.float32), o=np.zeros((5, 6), dtype=np.float32),
        e=np.zeros((5, 4), dtype=np.float32), q_p=qp,
        mode=np.zeros(5, dtype=np.int64), q_a=qa,
    )
    policy = ServoTrackingPolicy(
        artifact, trans, OGBenchSceneStateSchema(), coords,
        epsilon_a=np.full(5, 0.01), epsilon_p=np.full(6, 0.005),
    )
    return policy


def test_servo_tracks_r_bridge_to_landing_then_resumes_temporal_path():
    policy = _servo_fixture()
    policy.reset(0)
    action = policy.act({"q_a": np.zeros(5), "q_p": np.zeros(6), "mode": np.asarray(0), "free": True})
    assert action[0] == 0.5
    assert policy.status()["stage"] == "temporal"
    assert policy.pointer == 1
    action = policy.act({"q_a": np.asarray([0.025, 0, 0, 0, 0]), "q_p": np.zeros(6), "mode": np.asarray(0), "free": True})
    assert action[0] == 0.5
    assert policy.status()["stage"] == "bridge"
    policy.act({"q_a": np.asarray([0.05, 0, 0, 0, 0]), "q_p": np.zeros(6), "mode": np.asarray(0), "free": True})
    record = policy.status()["bridge_landing"][0]
    assert record["landed"]
    assert record["bridge_type"] == 0
    assert policy.pointer == 2


def test_servo_lookahead_does_not_skip_to_same_pose_after_passive_world_changes():
    policy = _servo_fixture()
    policy.reset(0)
    policy.path = [
        PathStep(0, 0, 1), PathStep(1, 0, 1),
        PathStep(2, 0, 1), PathStep(4, None, 0),
    ]
    policy.q_p[2, 0] = 0.1  # same active pose, but a different object world
    action = policy.act({"q_a": np.zeros(5), "q_p": np.zeros(6), "mode": np.asarray(0), "free": True})
    assert policy.status()["selected_vertex"] == 0
    assert policy.status()["target_vertex"] == 1
    assert action[0] == 0.5

