"""Simulator-free Lights Out algebra tests for the puzzle data and generator."""

import numpy as np

from graphcritic.puzzle import apply_presses, minimum_press_solution, solve_gf2, toggle_matrix
from experiments.gen_puzzle_task_data import _make_press_events, _standard_rows
from graphcritic.state_schema import OGBenchPuzzleStateSchema


def test_gf2_solutions_reach_targets_for_supported_bench_layouts():
    rng = np.random.default_rng(17)
    for n_rows, n_cols in ((3, 3), (4, 4), (4, 5)):
        matrix = toggle_matrix(n_rows, n_cols)
        for _ in range(4):
            start = rng.integers(0, 2, size=n_rows * n_cols, dtype=np.uint8)
            known_solution = rng.integers(0, 2, size=n_rows * n_cols, dtype=np.uint8)
            goal = apply_presses(start, known_solution, matrix)
            difference = start ^ goal
            solution = solve_gf2(matrix, difference, rng=rng)
            minimum = minimum_press_solution(matrix, difference)
            np.testing.assert_array_equal(apply_presses(start, solution, matrix), goal)
            np.testing.assert_array_equal(apply_presses(start, minimum, matrix), goal)
            assert minimum.sum() <= solution.sum()


def test_button_transition_is_an_involution_for_every_press_location():
    for n_rows, n_cols in ((3, 3), (4, 4), (4, 5)):
        matrix = toggle_matrix(n_rows, n_cols)
        board = np.zeros(n_rows * n_cols, dtype=np.uint8)
        for button in range(board.size):
            presses = np.zeros(board.size, dtype=np.uint8)
            presses[button] = 1
            after_one = apply_presses(board, presses, matrix)
            np.testing.assert_array_equal(apply_presses(after_one, presses, matrix), board)


def test_task_directed_press_events_preserve_goal_with_out_and_back_detours():
    rng = np.random.default_rng(0)
    for n_rows, n_cols in ((3, 3), (4, 4), (4, 5)):
        matrix = toggle_matrix(n_rows, n_cols)
        start = np.zeros(n_rows * n_cols, dtype=np.uint8)
        goal = np.ones(n_rows * n_cols, dtype=np.uint8)
        solution = minimum_press_solution(matrix, start ^ goal)
        events = _make_press_events(solution, rng, p_detour=1.0)
        current = start.copy()
        for button in events:
            press = np.zeros(current.size, dtype=np.uint8)
            press[button] = 1
            current = apply_presses(current, press, matrix)
        np.testing.assert_array_equal(current, goal)
        assert len(events) == int(solution.sum()) * 3


def test_puzzle_state_schema_encodes_sidecar_board_and_nearest_button_pose():
    obs = np.zeros((2, 55), dtype=np.float32)
    obs[:, 14] = 2.0  # 0.2 m EE height in the task observation scaling.
    obs[1, 18] = 1.0  # contact mode
    qpos = np.zeros((2, 23), dtype=np.float32)
    qpos[:, 14:] = 0.01
    buttons = np.zeros((2, 9), dtype=np.int64)
    buttons[1, 0] = 1
    coords = OGBenchPuzzleStateSchema().coordinates(
        obs, {"qpos": qpos, "button_states": buttons}
    )
    np.testing.assert_array_equal(coords.q_p[:, 0], [0.0, 1.0])
    np.testing.assert_array_equal(coords.mode, [0, 1])
    assert coords.e.shape == (2, 3)
    np.testing.assert_allclose(coords.e[:, 2], 0.2 - 0.1354, atol=1e-7)


def test_standard_mix_budget_counts_qc_transitions_not_terminal_only_rows(tmp_path):
    path = tmp_path / "standard.npz"
    terminals = np.zeros(8, dtype=np.bool_)
    terminals[[3, 7]] = True
    np.savez_compressed(
        path,
        observations=np.arange(24, dtype=np.float32).reshape(8, 3),
        actions=np.ones((8, 5), dtype=np.float32),
        terminals=terminals,
        qpos=np.arange(8 * 23, dtype=np.float32).reshape(8, 23),
        qvel=np.zeros((8, 22), dtype=np.float32),
        button_states=np.zeros((8, 16), dtype=np.int64),
    )
    mixed = _standard_rows(path, transitions=4, seed=0)
    assert mixed["observations"].shape[0] - mixed["terminals"].sum() == 4
    np.testing.assert_array_equal(np.flatnonzero(mixed["terminals"]), [3, 5])
    np.testing.assert_array_equal(mixed["actions"][mixed["terminals"]], 0.0)
