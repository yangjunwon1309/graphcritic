import numpy as np

from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.direct_q import (
    action_qrl_loss_components,
    min_plus_direct_q,
    puzzle_observation_to_frame_features,
    puzzle_observation_window,
    select_direct_q_action,
)
from graphcritic.types import SimStateSet, TransitionSet


def _puzzle_fixture():
    rng = np.random.default_rng(12)
    n_episode, episode_len, buttons = 2, 12, 9
    n = n_episode * episode_len
    qpos = rng.normal(size=(n + n_episode, 14 + buttons)).astype(np.float32)
    qvel = rng.normal(size=(n + n_episode, 14 + buttons)).astype(np.float32)
    board = rng.integers(0, 2, size=(n + n_episode, buttons), dtype=np.int64)
    obs = np.zeros((n, 19 + 4 * buttons), dtype=np.float32)
    obs[:, :14] = qpos[:n, :14]
    obs[:, 15] = 1.0
    obs[:, 17] = 3.0
    obs[:, 19 : 19 + 2 * buttons] = np.stack((1 - board[:n], board[:n]), axis=-1).reshape(n, -1)
    obs[:, 19 + 2 * buttons : 19 + 3 * buttons] = qpos[:n, 14:]
    obs[:, 19 + 3 * buttons :] = qvel[:n, 14:]
    terminals = np.zeros(n, dtype=bool)
    terminals[episode_len - 1 :: episode_len] = True
    episode_id = np.repeat(np.arange(n_episode), episode_len)
    times = np.tile(np.arange(episode_len), n_episode)
    next_obs = obs.copy()
    for start in range(0, n, episode_len):
        next_obs[start : start + episode_len - 1] = obs[start + 1 : start + episode_len]
    trans = TransitionSet(
        feat=obs, next_feat=next_obs, episode_id=episode_id, t=times,
        success=np.zeros(n, dtype=bool), terminal=terminals, row_idx=np.arange(n),
        action=rng.uniform(-1, 1, size=(n, 5)).astype(np.float32),
    )
    sim = SimStateSet(qpos=qpos, qvel=qvel, button_states=board)
    return trans, sim


def test_action_qrl_constraint_signs_and_epsilon_slack():
    import jax.numpy as jnp

    total, maximize, gap1, gap2 = action_qrl_loss_components(
        jnp.asarray([5.25]), jnp.asarray([0.25]), jnp.asarray([10.0]),
        2.0, 3.0, horizon=5, epsilon_c1=1.25, epsilon_c2=0.25,
    )
    _, worse_maximize, worse_gap1, worse_gap2 = action_qrl_loss_components(
        jnp.asarray([7.0]), jnp.asarray([0.5]), jnp.asarray([0.0]),
        2.0, 3.0, horizon=5, epsilon_c1=1.25, epsilon_c2=0.25,
    )
    assert float(gap1) == pytest.approx(-1.5)
    assert float(gap2) == pytest.approx(0.0)
    assert float(worse_gap1) > float(gap1)
    assert float(worse_gap2) > float(gap2)
    assert float(maximize) < float(worse_maximize)
    assert float(total) == pytest.approx(float(maximize) + 2.0 * float(gap1) + 3.0 * float(gap2))


def test_environment_observation_features_match_transition_sidecar_features():
    trans, sim = _puzzle_fixture()
    data = prepare_puzzle_model_data(trans, sim, seed=0)
    observed = puzzle_observation_to_frame_features(
        trans.feat[:trans.n_rows], data.metadata["state_mean"], data.metadata["state_std"],
        qpos=sim.qpos[:trans.n_rows], button_states=sim.button_states[:trans.n_rows],
        n_buttons=9,
    )
    expected = np.concatenate((data.state_cont[:trans.n_rows], data.state_bits[:trans.n_rows]), axis=1)
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(observed[:, -9:], sim.button_states[: trans.n_rows])


def test_live_observation_window_left_pads_episode_start_like_graph_windows():
    trans, sim = _puzzle_fixture()
    data = prepare_puzzle_model_data(trans, sim, seed=0)
    observations = trans.feat[:2]
    qpos = sim.qpos[:2]
    buttons = sim.button_states[:2]
    window, board = puzzle_observation_window(
        observations, data.metadata["state_mean"], data.metadata["state_std"],
        qpos_history=qpos, button_history=buttons, n_buttons=9, window=5,
    )
    frames = puzzle_observation_to_frame_features(
        observations, data.metadata["state_mean"], data.metadata["state_std"],
        qpos=qpos, button_states=buttons, n_buttons=9,
    )
    expected = np.concatenate((np.repeat(frames[:1], 4, axis=0), frames[1:2]), axis=0)
    np.testing.assert_allclose(window.reshape(5, -1), expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(board, buttons[1])


def test_min_plus_direct_q_matches_exact_h_step_cost():
    distances = np.full((4, 1, 1), 5.0, dtype=np.float32)
    q, conservative = min_plus_direct_q(distances, np.asarray([37.0]), k_limit=20)
    np.testing.assert_allclose(conservative, [[5.0]])
    np.testing.assert_allclose(q, [42.0])


def test_infinite_routes_fall_back_to_first_proposal():
    distances = np.full((4, 3, 2), 25.0, dtype=np.float32)
    q, _ = min_plus_direct_q(distances, np.asarray([10.0, 20.0]), k_limit=20)
    assert np.all(np.isinf(q))
    selected, fallback = select_direct_q_action(q)
    assert selected == 0
    assert fallback
    selected, fallback = select_direct_q_action(np.asarray([np.inf, 7.0, 7.0]))
    assert selected == 1
    assert not fallback


import pytest
