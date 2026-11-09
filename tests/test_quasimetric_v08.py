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
