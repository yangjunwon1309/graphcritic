"""Focused v0.5 regressions for witness control and physical-time estimates."""

import numpy as np
import pytest

from graphcritic.config import BridgeConfig, BridgeCostConfig
from graphcritic.physical_time import (
    estimate_static_articulation_tolerance,
    free_space_mask,
    realized_travel_steps,
)
from graphcritic.passive_bridges import _indices_in_mode, _r_tau_cap
from graphcritic.servo import contact_replay_correction
from graphcritic.state_schema import StateCoordinates, get_state_schema
from graphcritic.types import SchemaError


def _scene_coords(q_a, mode=None):
    q_a = np.asarray(q_a, dtype=np.float32)
    n = q_a.shape[0]
    return StateCoordinates(
        r=np.zeros((n, 19), dtype=np.float32),
        o=np.zeros((n, 6), dtype=np.float32),
        e=np.zeros((n, 4), dtype=np.float32),
        q_p=np.zeros((n, 6), dtype=np.float32),
        mode=np.zeros(n, dtype=np.int64) if mode is None else np.asarray(mode, dtype=np.int64),
        q_a=q_a,
    )


def test_contact_pose_correction_is_exact_replay_when_on_path():
    recorded = np.asarray([0.1, -0.2, 0.3, -0.4, 0.7], dtype=np.float32)
    action = contact_replay_correction(
        recorded,
        np.zeros(5, dtype=np.float64),
        np.asarray([0.05, 0.05, 0.05, 0.3, 1.0]),
        0.5,
    )
    np.testing.assert_array_equal(action, recorded)


def test_scene_free_mask_checks_cubes_buttons_handles_and_contact():
    schema = get_state_schema("ogbench_scene")
    qpos = np.zeros((4, 25), dtype=np.float32)
    qpos[:, 14:17] = [0.4, 0.0, 0.02]
    obs = np.zeros((4, 40), dtype=np.float32)
    q_a = np.asarray([
        [0.4, 0.0, 0.02, 0.0, 1.0],  # cube
        [0.8, 0.0, 0.2, 0.0, 1.0],  # free
        [0.58, -0.05, 0.125, 0.0, 1.0],  # button
        [0.8, 0.0, 0.2, 0.0, 1.0],  # contact
    ], dtype=np.float32)
    obs[3, 18] = 1.0
    free = free_space_mask(obs, qpos, _scene_coords(q_a), distance=0.05)
    np.testing.assert_array_equal(free, [False, True, False, False])


def test_tau_real_uses_max_coordinate_rate_and_settle():
    tau = realized_travel_steps(
        np.zeros(5),
        np.asarray([0.044, 0.024, 0.0, 0.2, 0.25]),
        np.asarray([0.02, 0.01, 0.03, 0.1, 0.1]),
        settle=2,
        periodic=(3,),
    )
    assert int(tau) == 5


def test_mode_filter_uses_full_state_labels_for_subset_indices():
    modes = np.asarray([0, 1, 1, 0, 2, 1])
    free_state_ids = np.asarray([1, 2, 4, 5])
    np.testing.assert_array_equal(_indices_in_mode(free_state_ids, modes, 1), [1, 2, 5])
    np.testing.assert_array_equal(_indices_in_mode(free_state_ids, modes, 2), [4])


def test_strict_m_bridge_extension_requires_a_learned_model():
    with pytest.raises(SchemaError, match="requires a learned tau checkpoint"):
        BridgeConfig(strategy="passive_r_strict", include_m_bridges=True)
    config = BridgeConfig(
        strategy="passive_r_strict", include_m_bridges=True,
        r_tau_max_steps=10,
        cost=BridgeCostConfig(
            tau_model="learned", tau_ckpt="model.npz", tau_max_steps=48,
        ),
    )
    assert config.include_m_bridges
    assert _r_tau_cap(config, config.cost) == 10


def test_tau_real_with_observed_speed_below_action_limits_exceeds_tau_kin():
    schema = get_state_schema("ogbench_scene")
    qa = np.asarray([0, 0, 0, 0, 1], dtype=np.float64)
    qb = np.asarray([0.11, 0.08, 0.02, 0.31, 0.7], dtype=np.float64)
    r_a = np.zeros(19, dtype=np.float64)
    r_b = np.zeros(19, dtype=np.float64)
    r_a[14:19] = qa
    r_b[14:19] = qb
    tau_kin = int(schema.kinematic_steps(r_a, r_b))
    tau_real = int(realized_travel_steps(
        qa, qb, np.asarray([0.04, 0.04, 0.04, 0.25, 0.8]), settle=2,
        periodic=(3,),
    ))
    assert tau_real >= tau_kin


def test_static_articulation_tolerance_uses_one_cm_floor():
    n = 120
    q_p = np.zeros((n, 6), dtype=np.float64)
    q_p[:, 4] = np.linspace(0.0, 5e-5, n)
    q_p[:, 5] = np.linspace(0.0, 8e-5, n)
    report = estimate_static_articulation_tolerance(
        q_p, np.zeros(n, dtype=np.int64), np.arange(n), np.ones(n, dtype=np.bool_),
    )
    np.testing.assert_allclose(report["epsilon"], [0.01, 0.01], atol=1e-8)
    assert all(not item["used_one_cm_fallback"] for item in report["dimensions"])
