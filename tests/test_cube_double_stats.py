import numpy as np

from experiments.cube_double_stats import (
    _goal_success,
    analyze_cube_double,
    decode_cube_xyz,
)


def _obs_from_positions(positions):
    positions = np.asarray(positions, dtype=np.float64)
    obs = np.zeros((len(positions), 37), dtype=np.float32)
    obs[:, 19:22] = (positions[:, 0] - np.asarray([0.425, 0.0, 0.0])) * 10.0
    obs[:, 28:31] = (positions[:, 1] - np.asarray([0.425, 0.0, 0.0])) * 10.0
    return obs


def _dataset(episodes):
    obs = []
    next_obs = []
    terminals = []
    for states in episodes:
        states = _obs_from_positions(states)
        obs.extend(states[:-1])
        next_obs.extend(states[1:])
        terminals.extend([False] * (len(states) - 2) + [True])
    return {
        "observations": np.asarray(obs, dtype=np.float32),
        "next_observations": np.asarray(next_obs, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=bool),
    }


def test_cube_observation_xyz_decode_and_assignment_invariant_goal_success():
    states = np.asarray([[[0.425, 0.1, 0.02], [0.425, 0.0, 0.02]]])
    encoded = _obs_from_positions(states.reshape(-1, 2, 3))
    np.testing.assert_allclose(decode_cube_xyz(encoded), states, atol=1e-7)
    goals = np.asarray([[0.425, 0.0, 0.02], [0.425, 0.1, 0.02]])
    assert _goal_success(states, goals, 0.04)[0]


def test_cube_double_stats_counts_cross_episode_matches_goals_and_graph_reachability():
    # Each episode contains an initial stable layout and a stable goal layout;
    # one episode stores the cube order swapped to match the randomized eval order.
    a = np.asarray([[0.30, 0.00, 0.02], [0.40, 0.00, 0.02]])
    b = np.asarray([[0.30, 0.10, 0.02], [0.40, 0.10, 0.02]])
    drift = np.linspace(0.0, 0.025, 8)
    move = np.stack([a[0] + np.asarray([x, 0, 0]) for x in drift])
    move2 = np.stack([a[1] + np.asarray([x, 0, 0]) for x in drift])
    moving = np.stack([move, move2], axis=1)
    episode1 = np.concatenate((np.repeat(a[None], 20, axis=0), moving, np.repeat(b[None], 20, axis=0)))
    episode2 = np.concatenate((np.repeat(a[::-1][None], 20, axis=0), moving[:, ::-1], np.repeat(b[::-1][None], 20, axis=0)))
    goals = {1: b.copy()}
    report = analyze_cube_double(_dataset([episode1, episode2]), goals)

    assert report["dataset"]["episode_count"] == 2
    assert report["dataset"]["transition_count"] == 2 * (len(episode1) - 1)
    assert report["movement"]["stationary_layouts"]["layout_count"] == 4
    assert report["task_goal_statistics"]["1"]["successful_stationary_layout_count"] == 2
    assert report["task_goal_statistics"]["1"]["episodes_with_successful_frame"] == 2
    assert report["cross_episode_object_state_matches"]["1"]["both_cubes_matching_pair_count"] >= 2
    assert report["stationary_layout_graph_connectivity"]["1"]["task_goal_reachability"]["1"]["goal_reachable_fraction"] == 1.0
