"""Focused tests for the v0.8.2 object-state quasimetric path."""

import numpy as np
import pytest

from graphcritic.bridges.judge_quasimetric import intersect_quasimetric_judges
from graphcritic.bridges.object_state import (
    chunk_constraint_costs,
    prepare_puzzle_object_data,
)
from graphcritic.state_schema import OGBenchPuzzleStateSchema
from graphcritic.types import SimStateSet, TransitionSet


def _object_dataset():
    steps_per_episode = 20
    n = 2 * steps_per_episode
    episodes = np.repeat(np.asarray([11, 29], dtype=np.int64), steps_per_episode)
    times = np.tile(np.arange(steps_per_episode, dtype=np.int64), 2)
    terminal = times == steps_per_episode - 1
    obs = np.zeros((n + 2, 83), dtype=np.float32)
    qpos = np.zeros((n + 2, 30), dtype=np.float32)
    qvel = np.zeros((n + 2, 30), dtype=np.float32)
    boards = np.zeros((n + 2, 16), dtype=np.int64)
    # Each episode has two isolated board changes. State-table terminal rows
    # duplicate the next state of each episode's final transition.
    for offset in (0, steps_per_episode):
        boards[offset + 5:offset + steps_per_episode] = 1
        boards[offset + 12:offset + steps_per_episode] = 0
    boards[n] = boards[steps_per_episode]
    boards[n + 1] = boards[-1]
    features = obs[:n]
    next_features = np.empty_like(features)
    next_features[:-1] = features[1:]
    next_features[steps_per_episode - 1] = obs[n]
    next_features[n - 1] = obs[n + 1]
    trans = TransitionSet(
        feat=features,
        next_feat=next_features,
        episode_id=episodes,
        t=times,
        success=np.zeros(n, dtype=np.bool_),
        terminal=terminal,
        row_idx=np.arange(n, dtype=np.int64),
        action=np.zeros((n, 5), dtype=np.float32),
    )
    sim = SimStateSet(qpos=qpos, qvel=qvel, button_states=boards)
    return trans, sim


def test_puzzle_object_schema_has_onehot_depth_and_velocity_but_passive_remains_packed():
    obs = np.zeros((2, 83), dtype=np.float32)
    qpos = np.zeros((2, 30), dtype=np.float32)
    qvel = np.zeros((2, 30), dtype=np.float32)
    qpos[1, 14] = 0.25
    qvel[1, 14] = -0.5
    buttons = np.zeros((2, 16), dtype=np.int64)
    buttons[1, 0] = 1
    coords = OGBenchPuzzleStateSchema().coordinates(
        obs, {"qpos": qpos, "qvel": qvel, "button_states": buttons}
    )
    assert coords.o.shape == (2, 64)
    np.testing.assert_array_equal(coords.o[0, :4], [1, 0, 1, 0])
    np.testing.assert_array_equal(coords.o[1, :4], [0, 1, 1, 0])
    np.testing.assert_allclose(coords.o[1, 32], 0.25)
    np.testing.assert_allclose(coords.o[1, 48], -0.5)
    np.testing.assert_array_equal(coords.q_p[:, 0], [0, 1])


def test_object_chunk_movement_and_onset_follow_button_change_windows():
    trans, sim = _object_dataset()
    data = prepare_puzzle_object_data(trans, sim, seed=0)
    # Button changes at frame 5 and 12 induce overlapping moved windows at
    # starts 0..4 and 7..11. Onset labels are evaluated on the non-overlapping
    # five-frame chunk grid, so the corresponding starts are 0 and 10.
    np.testing.assert_array_equal(np.flatnonzero(data.moved[:20]), np.r_[0:5, 7:12])
    np.testing.assert_array_equal(np.flatnonzero(data.onset[:20]), [0, 10])
    assert data.onset_report["minimum_frames"] == 10.0
    assert data.onset_report["p1_frames"] == pytest.approx(10.0)
    assert data.frame_features.shape[1] == 64
    assert data.windows.shape[1] == 320


def test_object_qrl_chunk_costs_match_event_and_motion_definitions():
    trans, sim = _object_dataset()
    data = prepare_puzzle_object_data(trans, sim, seed=0)
    event_costs = chunk_constraint_costs(data, "events")
    motion_costs = chunk_constraint_costs(data, "motion")
    np.testing.assert_array_equal(event_costs[data.onset], 1.0)
    np.testing.assert_array_equal(event_costs[~data.onset], 0.0)
    np.testing.assert_array_equal(motion_costs[data.moved], 5.0)
    np.testing.assert_array_equal(motion_costs[~data.moved], 0.0)


def test_object_event_and_motion_judges_intersect_and_report_joint_rejections():
    report = intersect_quasimetric_judges(
        np.asarray([4.0, 4.0, 21.0, 4.0, 21.0]),
        m_psi=np.asarray([3.0, 22.0, 3.0, 22.0, 22.0]),
        e_phi=np.asarray([1.0, 1.0, 1.0, 2.0, 2.0]),
        max_events=1,
    )
    np.testing.assert_array_equal(report["accept"], [True, False, False, False, False])
    assert report["categories"]["m_psi_only"] == 1
    assert report["categories"]["d_theta_only"] == 1
    assert report["categories"]["m_psi_and_e_phi_or_oracle"] == 1
    assert report["categories"]["all_three"] == 1
    assert report["individual_rejections"] == {"d_theta": 2, "m_psi": 3, "e_phi_or_oracle": 2}


def test_oracle_replaces_event_judge_and_uses_integer_press_limit():
    report = intersect_quasimetric_judges(
        np.asarray([4.0, 4.0, 4.0]),
        e_phi=None,
        oracle_press_count=np.asarray([0, 2, -1]),
        oracle_max_press_count=1,
    )
    np.testing.assert_array_equal(report["accept"], [True, False, False])
    assert report["individual_rejections"]["e_phi_or_oracle"] == 2


def test_oracle_press_count_uses_exact_minimum_gf2_solution():
    from experiments.build_judged_graph import _oracle_press_counts
    from graphcritic.puzzle import toggle_matrix

    toggle = toggle_matrix(4, 4)
    boards = np.zeros((3, 16), dtype=np.uint8)
    boards[1] = toggle[:, 0]
    boards[2] = toggle[:, 0] ^ toggle[:, 1]
    counts = _oracle_press_counts(np.asarray([[0, 1], [0, 2], [1, 2]]), boards)
    np.testing.assert_array_equal(counts, [1, 2, 1])


def test_d3_recall_uses_d1_oracle_feasible_theta_candidates():
    from experiments.build_judged_graph import _oracle_candidate_recall

    report = _oracle_candidate_recall(
        np.asarray([0, 1, 2, -1, 1]),
        np.asarray([2.0, 20.0, 3.0, 1.0, 21.0]),
        np.asarray([True, True, True, True, False]),
        max_press_count=1,
    )
    assert report["eligible_d1_candidate_count"] == 2
    assert report["retained_by_d3_count"] == 2
    assert report["recall"] == 1.0


@pytest.mark.skipif(not __import__("importlib").util.find_spec("jax"), reason="QRL loss requires JAX")
def test_qrl_loss_accepts_per_chunk_object_cost_vector():
    import jax.numpy as jnp
    from experiments.v08_runtime import qrl_loss_components

    _, _, violation, _ = qrl_loss_components(
        jnp.asarray([10.0, 20.0]), jnp.asarray([0.5, 1.5]), 1.0,
        epsilon=0.25, constraint_horizon=jnp.asarray([0.0, 1.0]),
    )
    np.testing.assert_allclose(violation, 0.25, atol=1e-6)
