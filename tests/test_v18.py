import numpy as np

from graphcritic.bridges.object_state import (
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
