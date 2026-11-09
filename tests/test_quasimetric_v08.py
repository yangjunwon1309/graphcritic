"""v0.8 judge and IQE structural tests."""

import numpy as np
import pytest


def _has_jax():
    try:
        import jax  # noqa: F401
        import flax  # noqa: F401
        return True
    except ImportError:
        return False


def _has_jax_gpu():
    if not _has_jax():
        return False
    try:
        import jax
        return bool(jax.devices("gpu"))
    except (RuntimeError, ValueError):
        return False


@pytest.mark.skipif(not _has_jax(), reason="IQE property test requires JAX/Flax")
def test_iqe_model_has_zero_self_distance_and_triangle_inequality():
    import jax
    import jax.numpy as jnp

    from experiments.v08_runtime import make_quasimetric_model

    model = make_quasimetric_model(
        encoder_widths=(32, 16), projector_widths=(32, 24),
        n_components=3, component_dim=8,
    )
    rng = np.random.default_rng(42)
    points = rng.normal(size=(40, 7)).astype(np.float32)
    params = model.init(
        jax.random.PRNGKey(9), jnp.asarray(points[:4]), jnp.asarray(points[4:8])
    )["params"]
    predict = lambda x, y: np.asarray(
        model.apply({"params": params}, jnp.asarray(x), jnp.asarray(y))
    )
    self_distance = predict(points, points)
    np.testing.assert_allclose(self_distance, 0.0, atol=1e-6)

    triplets = rng.integers(0, len(points), size=(500, 3))
    u, v, w = points[triplets[:, 0]], points[triplets[:, 1]], points[triplets[:, 2]]
    d_uv, d_vw, d_uw = predict(u, v), predict(v, w), predict(u, w)
    assert np.all(d_uw <= d_uv + d_vw + 2e-5)
    assert np.all(d_uv >= -1e-7)
    assert np.any(np.abs(d_uv - predict(v, u)) > 1e-6)


@pytest.mark.skipif(not _has_jax(), reason="model configuration test requires JAX/Flax")
def test_gelu_layernorm_model_exposes_collapse_diagnostics():
    import jax
    import jax.numpy as jnp

    from experiments.v08_runtime import (
        activation_diagnostics,
        make_activation_diagnostic_fn,
        make_quasimetric_model,
    )

    model = make_quasimetric_model(
        encoder_widths=(16, 8), projector_widths=(16, 8),
        n_components=2, component_dim=4,
    )
    states = np.random.default_rng(4).normal(size=(64, 5)).astype(np.float32)
    params = model.init(
        jax.random.PRNGKey(5), jnp.asarray(states[:4]), jnp.asarray(states[4:8])
    )["params"]
    assert "encoder_0_norm" in params
    assert "projector_0_norm" in params
    report = activation_diagnostics(
        make_activation_diagnostic_fn(model), params, states, activation="gelu"
    )
    assert report["n_diagnostic_states"] == len(states)
    assert set(report["dead_unit_frac_by_layer"]) == {"encoder_0", "projector_0"}
    assert all(0.0 <= value <= 1.0 for value in report["dead_unit_frac_by_layer"].values())
    assert report["projector_output_mean_variance"] >= 0.0


def test_collapse_alarm_uses_strict_dead_fraction_and_variance_limits():
    from experiments.v08_runtime import collapse_alarm

    healthy = {
        "dead_unit_frac_by_layer": {"projector_1": 0.95},
        "projector_output_mean_variance": 1e-6,
    }
    assert collapse_alarm(healthy, last_projector_layer="projector_1") == (False, None)
    dead = {
        "dead_unit_frac_by_layer": {"projector_1": 0.951},
        "projector_output_mean_variance": 1.0,
    }
    alarm, reason = collapse_alarm(dead, last_projector_layer="projector_1")
    assert alarm and "dead_unit_frac" in reason
    constant = {
        "dead_unit_frac_by_layer": {"projector_1": 0.1},
        "projector_output_mean_variance": 0.0,
    }
    alarm, reason = collapse_alarm(constant, last_projector_layer="projector_1")
    assert alarm and "projector_output_mean_variance" in reason


def test_projected_dual_sgd_moves_lambda_with_gap_and_respects_zero_boundary():
    if not _has_jax():
        pytest.skip("dual update test requires JAX")
    import jax
    import jax.numpy as jnp

    from experiments.v08_runtime import projected_dual_sgd

    increased = projected_dual_sgd(jnp.asarray(0.3), 0.1, 1.0, jnp)
    decreased = projected_dual_sgd(jnp.asarray(0.3), -0.1, 1.0, jnp)
    projected = projected_dual_sgd(jnp.asarray(0.1), -1.0, 1.0, jnp)
    np.testing.assert_allclose(increased, 0.4, atol=1e-6)
    np.testing.assert_allclose(decreased, 0.2, atol=1e-6)
    np.testing.assert_allclose(projected, 0.0, atol=0.0)


def test_warm_start_copies_dense_and_alpha_but_preserves_identity_layernorm():
    from experiments.train_quasimetric import _warm_start_parameters

    target = {
        "encoder_0": {"kernel": np.zeros((2, 3)), "bias": np.zeros(3)},
        "encoder_0_norm": {"scale": np.ones(3), "bias": np.zeros(3)},
        "projector_out": {"kernel": np.zeros((3, 4)), "bias": np.zeros(4)},
        "raw_alpha": np.asarray(0.0),
    }
    source = {
        "encoder_0": {"kernel": np.full((2, 3), 2.0), "bias": np.full(3, 3.0)},
        "projector_out": {"kernel": np.full((3, 4), 4.0), "bias": np.full(4, 5.0)},
        "raw_alpha": np.asarray(-2.0),
    }
    result = _warm_start_parameters(target, source)
    assert np.all(result["encoder_0"]["kernel"] == 2.0)
    assert np.all(result["projector_out"]["bias"] == 5.0)
    assert np.all(result["encoder_0_norm"]["scale"] == 1.0)
    assert np.all(result["encoder_0_norm"]["bias"] == 0.0)
    np.testing.assert_array_equal(result["raw_alpha"], -2.0)


@pytest.mark.skipif(not _has_jax(), reason="optimizer test requires JAX/Flax/Optax")
def test_train_step_reports_preclip_norm_with_projected_sgd_dual():
    import jax
    import jax.numpy as jnp

    from experiments.v08_runtime import initialize_train_state, make_quasimetric_model, make_train_step

    model = make_quasimetric_model(
        encoder_widths=(8, 4), projector_widths=(8, 8),
        n_components=2, component_dim=4,
    )
    update, primal_optimizer, dual_optimizer = make_train_step(
        model, batch_size=4, grad_clip=0.1, dual="sgd", lambda_learning_rate=1.0
    )
    params, primal_state, raw_lambda, dual_state, key = initialize_train_state(
        model, seed=3, input_dim=5,
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer,
    )
    states = jnp.asarray(np.random.default_rng(8).normal(size=(12, 5)), dtype=jnp.float32)
    next_indices = jnp.asarray(np.roll(np.arange(12, dtype=np.int32), -1))
    train_rows = jnp.arange(12, dtype=jnp.int32)
    result = update(
        params, primal_state, raw_lambda, dual_state, key,
        states, next_indices, train_rows,
    )
    metrics = result[-1]
    assert float(metrics["gradient_norm_preclip"]) > 0.0
    assert np.isfinite(float(metrics["lambda"]))


def test_quasimetric_judge_batches_and_uses_distance_as_cost():
    from graphcritic.bridges.judge_quasimetric import QuasimetricJudge

    calls = []

    def predictor(source, destination):
        calls.append(len(source))
        return np.abs(source[:, 0] - destination[:, 0]) * 2.0

    judge = QuasimetricJudge("dataset-x", predictor, max_distance_steps=2.0, batch_size=2)
    source = np.asarray([[0.0], [0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    destination = np.asarray([[0.0], [1.0], [3.0], [4.0], [0.0]], dtype=np.float32)
    result = judge.judge(source, destination)

    np.testing.assert_array_equal(result.accept, [True, True, True, True, False])
    np.testing.assert_allclose(result.cost_steps[result.accept], [0.0, 2.0, 2.0, 0.0])
    assert np.isnan(result.cost_steps[-1])
    np.testing.assert_allclose(result.extras["distance_steps"], [0.0, 2.0, 2.0, 0.0, 12.0])
    assert calls == [2, 2, 1]


def test_quasimetric_judge_rejects_wrong_shape_and_negative_distance():
    from graphcritic.bridges.judge_quasimetric import QuasimetricJudge

    wrong_shape = QuasimetricJudge("x", lambda s, d: np.zeros((len(s), 1)))
    with pytest.raises(ValueError, match="one scalar per pair"):
        wrong_shape.judge(np.zeros((2, 3)), np.ones((2, 3)))
    negative = QuasimetricJudge("x", lambda s, d: -np.ones(len(s)))
    with pytest.raises(ValueError, match="invalid distance"):
        negative.judge(np.zeros((2, 3)), np.ones((2, 3)))


def test_judge_result_rejects_dataset_misalignment_extras():
    from graphcritic.bridges.judge import JudgeResult

    with pytest.raises(ValueError, match="extra 'aux'"):
        JudgeResult(
            np.asarray([True, False]), np.asarray([1.0, np.nan]),
            {"aux": np.zeros((1,))},
        ).validate(2)


def test_flow_judge_adapter_returns_chunk_cost_and_action_extras():
    from graphcritic.bridges.check import BridgePlan
    from graphcritic.bridges.judge_flow import FlowJudge

    def evaluate(source, destination):
        del destination
        return [
            BridgePlan(0, 1, np.ones((2, 5, 5), dtype=np.float32), 2, 0.25, 3),
            None,
        ][:len(source)]

    result = FlowJudge("dataset-flow", evaluate).judge(
        np.zeros((2, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32)
    )
    np.testing.assert_array_equal(result.accept, [True, False])
    np.testing.assert_allclose(result.cost_steps[0], 10.0)
    assert np.isnan(result.cost_steps[1])
    assert result.extras["chunk_count"].tolist() == [2, 0]
    np.testing.assert_array_equal(result.extras["action_chunks"][0, :2], 1.0)


def test_candidate_comparison_reports_four_decision_cells_by_board_change(tmp_path):
    from experiments.compare_judges import compare_candidate_decisions

    pairs = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    changed = np.asarray([False, False, True, True])
    flow_path, quasi_path = tmp_path / "flow.npz", tmp_path / "quasi.npz"
    np.savez_compressed(
        flow_path, pairs=pairs, board_changed=changed,
        dataset_id=np.asarray("dataset"), judge=np.asarray("flow"),
        accept=np.asarray([True, True, False, False]),
    )
    np.savez_compressed(
        quasi_path, pairs=pairs, board_changed=changed,
        dataset_id=np.asarray("dataset"), judge=np.asarray("quasimetric"),
        accept=np.asarray([True, False, True, False]),
    )
    result = compare_candidate_decisions(flow_path, quasi_path)
    assert result["board_unchanged"] == {
        "both_accept": 1, "flow_only": 1, "quasimetric_only": 0,
        "both_reject": 0, "n": 2, "agreement_fraction": 0.5,
    }
    assert result["board_changed"]["quasimetric_only"] == 1


@pytest.mark.skipif(not _has_jax_gpu(), reason="grid learning smoke test requires GPU")
def test_quasimetric_learns_small_open_grid_distances():
    from experiments.quasimetric_grid import train_grid_quasimetric

    report = train_grid_quasimetric(side=5, steps=6000, seed=0)
    assert report["distance_spearman"] > 0.95
    assert report["distance_mae_steps"] < 1.2
