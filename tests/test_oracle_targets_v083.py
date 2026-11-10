from types import SimpleNamespace

import numpy as np
import pytest

from experiments import build_judged_graph as builder


def test_oracle_target_flag_requires_oracle_filter():
    with pytest.raises(SystemExit, match="requires --diagnostic-oracle-np-max"):
        builder._validate_oracle_target_flags(None, True)


def test_oracle_diagnostic_without_explicit_write_does_not_build_targets(
    tmp_path, monkeypatch
):
    def unexpected(*args, **kwargs):
        raise AssertionError("oracle diagnostic without opt-in must not build or save targets")

    monkeypatch.setattr(builder, "critic_targets", unexpected)
    monkeypatch.setattr(builder, "save_targets", unexpected)
    targets = builder._write_graph_targets(
        SimpleNamespace(n_rows=3), object(), tmp_path,
        diagnostic_oracle_np_max=1, write_oracle_targets=False,
    )
    assert targets is None


def test_oracle_target_write_uses_regret_config_and_records_provenance(
    tmp_path, monkeypatch
):
    trans = SimpleNamespace(n_rows=4)
    artifact = object()
    targets = SimpleNamespace(
        y=np.zeros(trans.n_rows, dtype=np.float32),
        valid=np.ones(trans.n_rows, dtype=np.bool_),
        loss_mask=np.ones(trans.n_rows, dtype=np.bool_),
    )
    calls = {}

    def fake_critic_targets(actual_trans, actual_artifact, config):
        calls["target_config"] = (actual_trans, actual_artifact, config)
        return targets

    def fake_save_targets(actual_targets, path, *, target_provenance):
        calls["save"] = (actual_targets, path, target_provenance)

    monkeypatch.setattr(builder, "critic_targets", fake_critic_targets)
    monkeypatch.setattr(builder, "save_targets", fake_save_targets)
    result = builder._write_graph_targets(
        trans, artifact, tmp_path,
        diagnostic_oracle_np_max=1, write_oracle_targets=True,
    )

    assert result is targets
    assert calls["target_config"][0:2] == (trans, artifact)
    config = calls["target_config"][2]
    assert (config.h, config.unit, config.kind) == (5, "steps", "regret")
    assert calls["save"] == (
        targets,
        tmp_path / "targets",
        {"kind": "oracle_upper_bound", "oracle_np_max": 1},
    )


def test_target_summary_uses_valid_loss_mask_rows():
    targets = SimpleNamespace(
        y=np.asarray([0.0, 2.0, 99.0], dtype=np.float32),
        valid=np.asarray([True, True, True]),
        loss_mask=np.asarray([True, True, False]),
        kind="regret",
    )
    summary = builder._target_summary(targets)
    assert summary["h5_regret_p90"] == pytest.approx(1.8)
    assert summary["h5_zero_regret_fraction"] == pytest.approx(0.5)
    assert summary["h5_regret_loss_mask_fraction"] == pytest.approx(2 / 3)


def _minimal_builder_args(*extra):
    return builder._parse([
        "--judge", "quasimetric",
        "--g0-graph", "g0.npz",
        "--transitions", "transitions.npz",
        "--sim-state", "sim_state.npz",
        "--checkpoint", "checkpoint",
        "--candidate-cache", "candidates.npz",
        "--out", "out",
        *extra,
    ])


def test_target_kind_flag_defaults_to_regret_and_accepts_cost():
    assert _minimal_builder_args().target_kind == "regret"
    assert _minimal_builder_args("--target-kind", "cost").target_kind == "cost"


def test_cost_target_writer_propagates_kind_and_transition_row_count(
    tmp_path, monkeypatch
):
    trans = SimpleNamespace(n_rows=4)
    artifact = object()
    targets = SimpleNamespace(
        y=np.arange(trans.n_rows, dtype=np.float32),
        valid=np.ones(trans.n_rows, dtype=np.bool_),
        loss_mask=np.ones(trans.n_rows, dtype=np.bool_),
        source=np.asarray([0, 1, 2, 2], dtype=np.uint8),
        kind="cost",
    )
    calls = {}

    def fake_critic_targets(actual_trans, actual_artifact, config):
        calls["config"] = config
        assert actual_trans is trans
        assert actual_artifact is artifact
        return targets

    def fake_save_targets(actual_targets, path, *, target_provenance):
        calls["save"] = (actual_targets, path, target_provenance)

    monkeypatch.setattr(builder, "critic_targets", fake_critic_targets)
    monkeypatch.setattr(builder, "save_targets", fake_save_targets)
    result = builder._write_graph_targets(
        trans, artifact, tmp_path,
        diagnostic_oracle_np_max=None,
        write_oracle_targets=False,
        target_kind="cost",
    )

    assert result is targets
    assert (calls["config"].h, calls["config"].unit, calls["config"].kind) == (
        5, "steps", "cost"
    )
    assert len(result.y) == trans.n_rows
    assert calls["save"] == (
        targets, tmp_path / "targets", {"kind": "learned"}
    )
    summary = builder._target_summary(result, target_kind="cost")
    assert summary["h5_cost_mean"] == pytest.approx(1.5)
    assert summary["h5_cost_p50"] == pytest.approx(1.5)
    assert summary["h5_cost_p90"] == pytest.approx(2.7)
    assert summary["h5_cost_max"] == pytest.approx(3.0)
    assert summary["h5_cost_capped_loss_mask_fraction"] == pytest.approx(0.5)
