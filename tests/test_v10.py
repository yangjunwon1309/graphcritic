"""Focused v1.0 matched-object, discounted-target, and direct-Q checks."""

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from envs.toy import make_chain
from experiments.v08_runtime import (
    object_consistency_batch_sizes,
    object_consistency_loss_components,
    orient_pair_batch,
    initialize_train_state,
    make_object_consistency_train_step,
    make_quasimetric_model,
)
from graphcritic.bridges.object_state import load_or_build_object_pair_index, object_match
from graphcritic.config import GraphConfig, TargetConfig
from graphcritic.direct_q import (
    action_qrl_loss_components,
    min_plus_direct_q,
    puzzle_observation_to_frame_features,
    puzzle_observation_window,
    select_direct_q_action,
)
from graphcritic.io import load_targets, save_targets
from graphcritic.pipeline import build_artifact
from graphcritic.targets import critic_targets
from graphcritic.weights import step_cost
from experiments.tmp_board_check import (
    _candidate_only_mode,
    _c0_optimistic_mask,
    _distance_group_report,
)
from experiments.train_action_quasimetric import embed_state_windows, flatten_state_windows
from experiments.write_direct_q_config import _validate_dplus_anchor


def test_object_match_exact_binary_states_and_componentwise_tolerance():
    left = np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    right = np.asarray([[0, 1, 0], [1, 1, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(object_match(left, right), [True, False])
    np.testing.assert_array_equal(
        object_match(np.asarray([[0.0, 1.0], [0.0, 1.0]]),
                     np.asarray([[0.01, 1.015], [0.03, 1.0]]), epsilon=[0.02, 0.02]),
        [True, False],
    )


def test_object_pair_cache_partitions_candidate_rows_and_reuses_cache(tmp_path):
    candidate_path = tmp_path / "candidates.npz"
    pairs = np.asarray([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    states = np.asarray([[0, 0], [0, 0], [0, 1], [1, 1]], dtype=np.uint8)
    np.savez_compressed(
        candidate_path,
        pairs=pairs,
        node_ids=np.arange(4, dtype=np.int32),
        dataset_id=np.asarray("unit-test"),
        window=np.asarray(5),
        k=np.asarray(32),
    )
    cached_pairs, matched, unmatched, index_path, hit = load_or_build_object_pair_index(
        candidate_path, states, dataset_id="unit-test",
    )
    np.testing.assert_array_equal(cached_pairs, pairs)
    np.testing.assert_array_equal(matched, [0])
    np.testing.assert_array_equal(unmatched, [1, 2, 3])
    assert index_path.is_file() and not hit
    _, matched_again, unmatched_again, _, hit_again = load_or_build_object_pair_index(
        candidate_path, states, dataset_id="unit-test",
    )
    np.testing.assert_array_equal(matched_again, matched)
    np.testing.assert_array_equal(unmatched_again, unmatched)
    assert hit_again


def test_v1_p2_p3_loss_signs_pair_orientation_and_batch_ratio():
    random_n, unmatched_n = object_consistency_batch_sizes(4096, 0.5)
    assert (random_n, unmatched_n) == (2048, 2048)
    flipped = orient_pair_batch(
        jnp.asarray([[1, 2], [3, 4], [5, 6]]),
        jnp.asarray([True, False, True]),
    )
    np.testing.assert_array_equal(np.asarray(flipped), [[2, 1], [3, 4], [6, 5]])

    _, maximize_small, satisfied_chunk, satisfied_match = object_consistency_loss_components(
        jnp.asarray([10.0, 10.0]), jnp.asarray([10.0, 10.0]),
        jnp.asarray([5.0]), jnp.asarray([2.0]), jnp.asarray([2.0]),
        0.01, 0.01, constraint_horizon=5, epsilon_chunk=1.25,
        epsilon_match=1.25,
    )
    _, maximize_large, violated_chunk, violated_match = object_consistency_loss_components(
        jnp.asarray([200.0, 200.0]), jnp.asarray([200.0, 200.0]),
        jnp.asarray([7.0]), jnp.asarray([5.0]), jnp.asarray([2.0]),
        0.01, 0.01, constraint_horizon=5, epsilon_chunk=1.25,
        epsilon_match=1.25,
    )
    assert float(satisfied_chunk) < 0.0
    assert float(satisfied_match) < 0.0
    assert float(violated_chunk) > float(satisfied_chunk)
    assert float(violated_match) > float(satisfied_match)
    assert float(maximize_large) < float(maximize_small)


def test_v1_object_consistency_update_runs_with_two_duals():
    model = make_quasimetric_model(
        encoder_widths=(8, 4), projector_widths=(8, 8),
        n_components=2, component_dim=4,
    )
    update, primal, dual = make_object_consistency_train_step(
        model, batch_size=4, unmatched_max_frac=0.5,
    )
    import jax
    import jax.numpy as jnp

    params, primal_state, raw_chunk, dual_chunk, key = initialize_train_state(
        model, seed=0, input_dim=3, primal_optimizer=primal, dual_optimizer=dual,
    )
    raw_match = jnp.asarray(raw_chunk)
    dual_match = dual.init(raw_match)
    states = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    output = update(
        params, primal_state, raw_chunk, dual_chunk, raw_match, dual_match, key,
        states, jnp.arange(8, dtype=jnp.int32), jnp.arange(6, dtype=jnp.int32),
        jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        jnp.asarray([1, 2, 3, 4], dtype=jnp.int32),
        jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        jnp.asarray([1.0, 2.0]), jnp.asarray([2.0, 1.0]),
        jnp.asarray([[0, 2], [1, 3]], dtype=jnp.int32),
    )
    metrics = output[-1]
    assert np.isfinite(float(metrics["loss"]))
    assert np.isfinite(float(metrics["lambda_chunk"]))
    assert np.isfinite(float(metrics["lambda_match"]))


def _no_goal_chain_artifact():
    trans, _ = make_chain(L=8, n_episodes=3, p_forward=1.0, seed=8)
    trans = replace(
        trans,
        success=np.zeros_like(trans.success),
        success_cur=np.zeros_like(trans.success_cur),
    )
    cfg = GraphConfig(
        graph_kind="node", radius=0.25, compress=False, weight_policy="time",
    )
    with pytest.warns(RuntimeWarning, match="no goal"):
        artifact = build_artifact(trans, cfg, dataset_id="no-goal", source="test")
    return trans, artifact


def test_discounted_targets_match_discounted_steps_and_cap_unreachable_at_100(tmp_path):
    gamma = 0.99
    trans, artifact = make_chain(L=8, n_episodes=3, p_forward=1.0, seed=8)
    cfg = GraphConfig(
        graph_kind="node", radius=0.25, compress=False, weight_policy="time",
    )
    reachable = build_artifact(trans, cfg, dataset_id="chain-discount", source="test")
    targets = critic_targets(
        trans, reachable,
        TargetConfig(h=5, unit="steps", kind="discounted", discount_gamma=gamma),
    )
    row = int(np.flatnonzero(targets.valid & (targets.source == 1))[0])
    endpoint = reachable.nodes.assign_next[row + 4]
    endpoint_cost_steps = reachable.cost[endpoint] / step_cost(reachable.config.gamma)
    expected = (1.0 - gamma ** (5.0 + endpoint_cost_steps)) / (1.0 - gamma)
    assert targets.y[row] == pytest.approx(expected, rel=1e-6)
    path = tmp_path / "discounted_targets.npz"
    save_targets(targets, path)
    restored = load_targets(path)
    assert restored.kind == "discounted"
    assert restored.discount_gamma == pytest.approx(gamma)
    np.testing.assert_array_equal(restored.y, targets.y)

    unreachable_trans, unreachable_graph = _no_goal_chain_artifact()
    capped = critic_targets(
        unreachable_trans, unreachable_graph,
        TargetConfig(h=5, unit="steps", kind="discounted", discount_gamma=gamma),
    )
    cap_rows = capped.valid & (capped.source == 2)
    assert np.any(cap_rows)
    np.testing.assert_allclose(capped.y[cap_rows], 100.0, atol=1e-5, rtol=0)


def test_v09_direct_q_min_plus_fallback_and_qrl_constraints_remain_stable():
    total, maximize, gap1, gap2 = action_qrl_loss_components(
        jnp.asarray([5.25]), jnp.asarray([0.25]), jnp.asarray([10.0]),
        2.0, 3.0, horizon=5, epsilon_c1=1.25, epsilon_c2=0.25,
    )
    assert float(gap1) == pytest.approx(-1.5)
    assert float(gap2) == pytest.approx(0.0)
    assert float(total) == pytest.approx(float(maximize) + 2 * float(gap1) + 3 * float(gap2))

    q, distances = min_plus_direct_q(
        np.full((4, 1, 1), 5.0, dtype=np.float32), np.asarray([37.0]),
    )
    np.testing.assert_allclose(distances, [[5.0]])
    np.testing.assert_allclose(q, [42.0])
    selected, fallback = select_direct_q_action(np.asarray([np.inf, np.inf]))
    assert selected == 0 and fallback
    selected, fallback = select_direct_q_action(np.asarray([np.inf, 7.0]))
    assert selected == 1 and not fallback


def test_v09_live_observation_window_preserves_sidecar_feature_layout():
    n, buttons = 2, 9
    observations = np.zeros((n, 19 + 4 * buttons), dtype=np.float32)
    observations[:, 15] = 1.0
    observations[:, 17] = 3.0
    qpos = np.zeros((n, 14 + buttons), dtype=np.float32)
    qvel = np.zeros_like(qpos)
    board = np.zeros((n, buttons), dtype=np.uint8)
    board[1, 4] = 1
    observations[:, 19 : 19 + 2 * buttons] = np.stack(
        (1 - board, board), axis=-1,
    ).reshape(n, -1)
    observations[:, 19 + 2 * buttons : 19 + 3 * buttons] = qpos[:, 14:]
    observations[:, 19 + 3 * buttons :] = qvel[:, 14:]
    mean = np.zeros(19, dtype=np.float64)
    std = np.ones(19, dtype=np.float64)
    frames = puzzle_observation_to_frame_features(
        observations, mean, std, qpos=qpos, button_states=board, n_buttons=buttons,
    )
    window, final_board = puzzle_observation_window(
        observations, mean, std, qpos_history=qpos, button_history=board,
        n_buttons=buttons, window=5,
    )
    expected = np.concatenate((np.repeat(frames[:1], 4, axis=0), frames[1:]), axis=0)
    np.testing.assert_allclose(window.reshape(5, -1), expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(final_board, board[-1])


def test_discounted_targets_reject_non_step_units():
    with pytest.raises(ValueError, match="discounted targets"):
        TargetConfig(h=5, unit="nats", kind="discounted", discount_gamma=0.99)


def test_v10_distance_candidate_audit_does_not_require_event_checkpoint():
    args = SimpleNamespace(
        distance_checkpoint="dplus", object_event_checkpoint=None,
        reference_checkpoint="dtheta",
    )
    assert _candidate_only_mode(args) == "distance"

    counts = np.asarray([0, 1, 2, 3, 4, -1])
    dplus = np.asarray([2, 4, 6, 8, 10, 12], dtype=np.float32)
    dtheta = np.asarray([3, 5, 7, 9, 11, 13], dtype=np.float32)
    summary = _distance_group_report(counts, dplus, dtheta)
    assert [summary[key]["count"] for key in ("0", "1", "2", "3", "≥4", "∞")] == [1] * 6
    assert summary["2"]["d_plus_p50"] == pytest.approx(6)
    assert summary["2"]["d_theta_p50"] == pytest.approx(7)
    np.testing.assert_array_equal(
        _c0_optimistic_mask(np.asarray([[0, 1], [1, 2], [2, 3]]),
                            np.asarray([20, 10, np.inf, 1]), minimum_gain=10),
        [True, False, True],
    )


def test_v10_action_state_embedder_flattens_windows_in_v08_order():
    import jax
    import jax.numpy as jnp
    from experiments.v08_runtime import make_quasimetric_embedder

    windows = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    expected_flat = windows.reshape(3, 20)
    np.testing.assert_array_equal(np.asarray(flatten_state_windows(windows)), expected_flat)
    embedder = make_quasimetric_embedder(
        make_quasimetric_model(
            encoder_widths=(8, 4), projector_widths=(8, 8),
            n_components=2, component_dim=4,
        )
    )
    params = embedder.init(jax.random.PRNGKey(31), jnp.asarray(expected_flat))["params"]
    embedded_window = embed_state_windows(embedder, params, jnp.asarray(windows))
    embedded_flat = embedder.apply({"params": params}, jnp.asarray(expected_flat))
    np.testing.assert_allclose(
        np.asarray(embedded_window), np.asarray(embedded_flat), atol=0, rtol=0
    )


def test_v10_dplus_anchor_uses_provenance_hash_and_object_constraint():
    _validate_dplus_anchor(
        {
            "target_provenance": {"kind": "learned"},
            "bridge_model_metadata": {"params_sha256": "checkpoint-sha"},
        },
        {"spec_version": "v0.8", "match_constraint": "object"},
        "checkpoint-sha",
    )
    with pytest.raises(ValueError, match="object-match"):
        _validate_dplus_anchor(
            {
                "target_provenance": {"kind": "learned"},
                "bridge_model_metadata": {"params_sha256": "checkpoint-sha"},
            },
            {"spec_version": "v1.0", "match_constraint": "none"},
            "checkpoint-sha",
        )
