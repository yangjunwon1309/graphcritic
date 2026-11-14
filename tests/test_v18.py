from pathlib import Path

import numpy as np

from graphcritic.bridges.object_state import (
    cube_object_threshold_match,
    cube_symmetry_rotations,
    object_distance,
    object_threshold_match,
)
from graphcritic.envs.ogbench_cube import make_cube_transition_set, split_observation
from graphcritic.types import TransitionSet


def _matrix_to_wxyz(matrix):
    """Stable matrix-to-quaternion conversion for the 24 exact test rotations."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = np.empty(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    return q / np.linalg.norm(q)


def test_cube_group_and_symmetry_quotient_rotation_distance():
    group = cube_symmetry_rotations()
    assert group.shape == (24, 3, 3)
    np.testing.assert_allclose(
        group @ np.swapaxes(group, -1, -2), np.broadcast_to(np.eye(3), (24, 3, 3)), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(group), np.ones(24), atol=1e-12)

    first = np.zeros((2, 7), dtype=np.float64)
    first[:, 3] = 1.0
    first[:, :3] = [[0.4, 0.1, 0.02], [0.5, -0.1, 0.06]]
    for rotation in group:
        second = first.copy()
        second[:, 3:] = _matrix_to_wxyz(rotation)
        d_pos, d_rot = object_distance(first, second)
        assert d_pos == 0.0
        assert d_rot < 1e-7


def test_quaternion_sign_flip_does_not_change_object_distance():
    first = np.array([[0.4, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0],
                      [0.5, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]])
    second = first.copy()
    second[:, 3:] *= -1
    assert object_distance(first, second) == (0.0, 0.0)
    assert object_threshold_match(first, second, delta_pos=0.005, delta_rot=np.deg2rad(15))


def test_observation_pose_scaling_is_restored_to_metric_coordinates():
    obs = np.zeros((1, 37), dtype=np.float32)
    expected = np.array([[0.44, -0.08, 0.02], [0.49, 0.12, 0.06]])
    obs[0, 19:22] = (expected[0] - [0.425, 0, 0]) * 10
    obs[0, 22] = 1.0
    obs[0, 28:31] = (expected[1] - [0.425, 0, 0]) * 10
    obs[0, 31] = 1.0
    robot, objects = split_observation(obs)
    assert robot.shape == (1, 19)
    np.testing.assert_allclose(objects[0, :, :3], expected, atol=1e-7)
    np.testing.assert_allclose(objects[0, :, 3:], [[1, 0, 0, 0], [1, 0, 0, 0]])


def test_puzzle_exact_match_remains_delta_zero_special_case():
    left = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    same = left.copy()
    changed = left.copy()
    changed[0, 2] = 1
    assert object_threshold_match(left, same, delta_pos=0, delta_rot=0, representation="puzzle")
    assert not object_threshold_match(left, changed, delta_pos=0, delta_rot=0, representation="puzzle")
    try:
        object_threshold_match(left, same, delta_pos=0.001, delta_rot=0, representation="puzzle")
    except ValueError:
        pass
    else:
        raise AssertionError("Puzzle matching must reject nonzero continuous tolerances")


def test_cube_transition_rows_match_compact_and_regular_loader_order():
    observations = np.zeros((12, 37), dtype=np.float32)
    observations[:, 22] = 1.0
    observations[:, 31] = 1.0
    actions = np.arange(12 * 5, dtype=np.float32).reshape(12, 5)
    terminals = np.zeros(12, dtype=bool)
    terminals[[5, 11]] = True
    goals = np.array([[0.425, 0.0, 0.02], [0.425, 0.1, 0.02]])
    transitions = make_cube_transition_set(observations, actions, terminals, goals=goals)

    assert transitions.n_rows == 10
    np.testing.assert_array_equal(transitions.episode_id, np.repeat([0, 1], 5))
    np.testing.assert_array_equal(transitions.t, np.tile(np.arange(5), 2))
    np.testing.assert_array_equal(transitions.terminal, [False] * 4 + [True] + [False] * 4 + [True])
    np.testing.assert_array_equal(transitions.row_idx, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
    np.testing.assert_array_equal(transitions.feat[:, 0], observations[transitions.row_idx, 0])
    np.testing.assert_array_equal(transitions.next_feat[:, 0], observations[transitions.row_idx + 1, 0])


def test_h5_target_resolves_terminal_endpoint_from_its_own_episode():
    from graphcritic.bridges.cube_graph import _next_state_indices, _target_set

    episode_id = np.repeat(np.arange(2), 5)
    terminal = np.zeros(10, dtype=bool)
    terminal[[4, 9]] = True
    observations = np.zeros((10, 37), dtype=np.float32)
    transitions = TransitionSet(
        feat=observations,
        next_feat=observations.copy(),
        episode_id=episode_id,
        t=np.tile(np.arange(5), 2),
        success=np.zeros(10, dtype=bool),
        success_cur=np.zeros(10, dtype=bool),
        terminal=terminal,
        row_idx=np.arange(10),
        mode=np.zeros(10, dtype=np.int64),
        mode_next=np.zeros(10, dtype=np.int64),
        action=np.zeros((10, 5), dtype=np.float32),
    )
    terminal_state_ids = _next_state_indices(terminal)[terminal]
    state_cost = np.full(12, np.inf)
    state_cost[terminal_state_ids] = [100.0, 200.0]
    context = {
        "transitions": transitions,
        "terminal_state_index": _next_state_indices(terminal),
    }
    target = _target_set(
        context, 1, state_cost, "a" * 64, "cube-double-task", kind="gamma_pow", h=5, gamma=0.99,
    )
    np.testing.assert_array_equal(np.flatnonzero(target.valid), [0, 5])
    np.testing.assert_allclose(target.y[[0, 5]], [0.99**105, 0.99**205], rtol=1e-6)


def test_five_step_constraint_maps_boundary_endpoint_to_episode_terminal_state():
    from types import SimpleNamespace

    from experiments.train_quasimetric import _targets_at_offset

    data = SimpleNamespace(
        episode_end_rows=np.full(5, 4, dtype=np.int64),
        terminal_state_index=np.asarray([-1, -1, -1, -1, 5], dtype=np.int64),
    )
    rows, endpoints = _targets_at_offset(np.asarray([0, 1, 2]), data, 5)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(endpoints, [5])


def _write_v18_gate_run(path, *, step=30_000, violation=0.04, collapse=False, dead=0.0,
                        frame_dim=19, input_dim=95):
    import json

    checkpoint = path / f"step_{step:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({
        "step": step, "input_part": "robot", "frame_dim": frame_dim, "input_dim": input_dim,
    }))
    (path / "train_metrics.jsonl").write_text(json.dumps({
        "step": step,
        "collapse_alarm": collapse,
        "collapse_alarm_reason": "synthetic" if collapse else None,
        "diagnostics": {"dead_unit_frac_by_layer": {"encoder_0": dead}},
    }) + "\n")
    (path / "training.jsonl").write_text(json.dumps({
        "step": step,
        "heldout_chunk_constraint": {"overall_violation_rate_d_gt_limit": violation},
        "train_chunk_constraint": {"overall_violation_rate_d_gt_limit": 0.03},
    }) + "\n")


def test_v18_gate_checker_accepts_requested_step_and_report_only_f3(tmp_path):
    import json
    import subprocess
    import sys

    run = tmp_path / "run"
    run.mkdir()
    _write_v18_gate_run(run, violation=0.06)
    script = Path(__file__).resolve().parents[1] / "experiments/check_v18_model.py"
    stop = subprocess.run(
        [sys.executable, str(script), "--run", str(run), "--step", "30000"],
        capture_output=True, text=True,
    )
    assert stop.returncode == 2
    stop_report = json.loads(stop.stdout)
    assert stop_report["step"] == 30_000
    assert stop_report["F3_violation_reported"] is False
    report = subprocess.run(
        [sys.executable, str(script), "--run", str(run), "--step", "30000",
         "--violation-gate", "report"],
        capture_output=True, text=True,
    )
    assert report.returncode == 0
    report_json = json.loads(report.stdout)
    assert report_json["gate_passed"] is True
    assert report_json["F3_passed"] is False
    assert report_json["F3_violation_reported"] is True
    assert report_json["train_chunk_violation_d_gt_6_25"] == 0.03


def test_v18_report_mode_keeps_collapse_dead_unit_and_dimension_hard_gates(tmp_path):
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "experiments/check_v18_model.py"
    cases = [
        {"collapse": True},
        {"dead": 0.001},
        {"input_dim": 19},
        {"frame_dim": 20},
    ]
    for index, values in enumerate(cases):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        _write_v18_gate_run(run, violation=0.06, **values)
        result = subprocess.run(
            [sys.executable, str(script), "--run", str(run), "--step", "30000",
             "--violation-gate", "report"],
            capture_output=True, text=True,
        )
        assert result.returncode == 2


def _cube_pose_pair():
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[:, 3] = 1.0
    poses[:, :3] = [[0.42, 0.0, 0.02], [0.52, 0.0, 0.02]]
    return poses


def test_v18b_component_match_rejects_height_gap_and_contact_mismatch_but_keeps_cube_symmetry():
    first = _cube_pose_pair()
    raised = first.copy()
    raised[1, 2] += 0.015
    assert not cube_object_threshold_match(
        first, raised, delta_xy=0.04, delta_z=0.01, contact=False, contact_prime=False
    )
    assert not cube_object_threshold_match(
        first, first, delta_xy=0.04, delta_z=0.01, contact=False, contact_prime=True
    )
    symmetric = first.copy()
    symmetric[:, 3:] = _matrix_to_wxyz(cube_symmetry_rotations()[7])
    assert cube_object_threshold_match(
        first, symmetric, delta_xy=0.04, delta_z=0.01,
        delta_rot=np.deg2rad(15), contact=False, contact_prime=False,
    )


def _budget_context(contact_flags):
    from types import SimpleNamespace

    states = np.zeros((4, 37), dtype=np.float32)
    states[:, 18] = np.asarray(contact_flags, dtype=np.float32)
    objects = _cube_pose_pair()[None, :, :].repeat(4, axis=0)
    return {
        "states": states,
        "objects": objects,
        "graph_node_state_ids": np.arange(4, dtype=np.int64),
        "state_episode": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "state_time": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "state_success_by_task": {1: np.asarray([False, False, False, True])},
        "transitions": SimpleNamespace(
            episode_id=np.asarray([0, 1], dtype=np.int64),
            terminal=np.asarray([True, True], dtype=bool),
            n_rows=2,
        ),
    }


def test_v18b_free_robot_distance_cap_only_applies_when_both_endpoints_are_free():
    from graphcritic.bridges.cube_graph import task_budget_sweep

    pairs = np.asarray([[0, 2]], dtype=np.int64)
    free_ctx = _budget_context([0, 0, 0, 0])
    raised_ctx = _budget_context([0, 0, 1, 0])
    free = task_budget_sweep(
        free_ctx, 1, pairs, np.asarray([30.0]), max_budget=1,
        bridge_filter="none", k_free=40,
    )
    with_contact = task_budget_sweep(
        raised_ctx, 1, pairs, np.asarray([30.0]), max_budget=1,
        bridge_filter="none", k_free=40,
    )
    assert free["n_accepted_bridges"] == 1
    assert with_contact["n_accepted_bridges"] == 0


def test_v18b_none_filter_allows_a_two_bridge_chain():
    from graphcritic.bridges.cube_graph import budget_cost_sweep

    costs = budget_cost_sweep(
        4, np.asarray([3]),
        np.asarray([2]), np.asarray([3]), np.asarray([1.0]),
        np.asarray([0, 1]), np.asarray([1, 2]), np.asarray([2.0, 2.0]),
        max_budget=2,
    )
    assert np.isinf(costs[0, 0])
    assert np.isinf(costs[1, 0])
    assert np.isfinite(costs[2, 0])
    assert costs[2, 0] == 5.0


def test_v18b_puzzle_graph_hash_regression_fixture_is_unchanged():
    # Keep the generic/puzzle artifact construction covered while v1.8b
    # changes only the cube-specific bridge builder.
    from graphcritic.config import FeatureConfig, GraphConfig
    from graphcritic.types import EdgeSet, GraphArtifact, NodeSet
    from experiments.build_judged_graph import _make_artifact

    base = GraphArtifact(
        config=GraphConfig(graph_kind="node", feature=FeatureConfig(standardize=False)),
        feature_stats={"mean": np.asarray([0.0], np.float32), "std": np.asarray([1.0], np.float32)},
        nodes=NodeSet(
            rep_feat=np.asarray([[0.0], [1.0], [2.0]], np.float32),
            rep_ref=np.asarray([[-1, 0], [-1, 0], [-1, 0]], np.int64),
            assign_cur=np.empty(0, np.int64), assign_next=np.empty(0, np.int64),
            is_goal=np.asarray([False, False, True]),
        ),
        edges=EdgeSet(
            src=np.asarray([0, 1]), dst=np.asarray([1, 2]),
            kappa=np.ones(2, np.int64), n_pos=np.ones(2, np.int64),
            n_neg=np.zeros(2, np.int64), kind=np.zeros(2, np.int8),
        ),
        weights=np.asarray([1.0, 1.0]), cost=np.asarray([2.0, 1.0, 0.0]),
        succ=np.asarray([1, 2, -1]),
        meta={"dataset_id": "lambda-test", "source": "test", "core_version": "0.5.0",
              "n_rows": 0, "c_max_steps": 3},
    )
    artifact = _make_artifact(
        base, np.asarray([0]), np.asarray([2]), np.asarray([2.5]), np.asarray([2.5]),
        np.asarray([3.0, 1.0, 0.0]), np.asarray([1, 2, -1]),
        judge_name="quasimetric", model_metadata={"params_sha256": "abc"},
        candidate_rule="test", window_size=5, bridge_lambda_steps=0.0,
    )
    assert artifact.content_hash == "5122c919d2bca72b4e3b8003d2cb3c782340b0256c79eb4fb76bf0219fa00012"
