"""v0.6 regret-target invariants and target artifact schema tests."""

import json

import numpy as np
import pytest

from envs.toy import make_chain
from graphcritic.config import BridgeConfig, GraphConfig, TargetConfig
from graphcritic.diagnostics import report
from graphcritic.io import load_targets, save_targets
from graphcritic.pipeline import build_artifact
from graphcritic.targets import critic_targets
from graphcritic.types import SchemaError, _content_hash


def _artifact(trans, graph_kind="state"):
    cfg = GraphConfig(
        graph_kind=graph_kind, radius=0.25, compress=False, weight_policy="time",
        bridge=BridgeConfig(enabled=False),
    )
    return build_artifact(trans, cfg, dataset_id="toy:regret", source="unit-test")


def test_regret_nonnegative_and_shortest_chain_chunks_have_zero_regret():
    trans, _ = make_chain(L=12, n_episodes=5, p_forward=1.0, seed=1)
    artifact = _artifact(trans)
    targets = critic_targets(trans, artifact, TargetConfig(h=3))
    assert targets.kind == "regret"
    assert targets.loss_mask.all() == targets.valid.all()
    assert np.all(targets.y[targets.loss_mask] >= 0)
    np.testing.assert_allclose(targets.y[targets.loss_mask], 0.0, atol=1e-5)
    assert report(trans, artifact, targets)["targets"]["property_1_violation_count"] == 0


def test_regret_preserves_same_state_argmin_and_cost_target_is_explicit():
    trans, _ = make_chain(L=10, n_episodes=8, p_forward=0.7, seed=5)
    artifact = _artifact(trans, graph_kind="node")
    cost = critic_targets(trans, artifact, TargetConfig(h=2, kind="cost"))
    regret = critic_targets(trans, artifact, TargetConfig(h=2, kind="regret"))
    rows = np.flatnonzero(cost.valid & regret.loss_mask & (cost.source == 1) & (regret.source == 1))
    node = artifact.nodes.assign_cur[rows]
    groups = [rows[node == current] for current in np.unique(node)]
    groups = [group for group in groups if group.size >= 2]
    assert groups
    for group in groups:
        # All rows in a group share C_G(s_i); subtracting that constant
        # preserves the candidate ordering exactly.
        np.testing.assert_array_equal(
            np.argsort(cost.y[group], kind="stable"),
            np.argsort(regret.y[group], kind="stable"),
        )


def test_both_capped_rows_remain_qc_valid_but_are_excluded_from_loss():
    trans, _ = make_chain(L=7, n_episodes=2, p_forward=1.0, seed=4)
    trans.success[:] = False
    trans.success_cur[:] = False
    artifact = _artifact(trans)
    targets = critic_targets(trans, artifact, TargetConfig(h=2))
    assert np.any(targets.valid & (targets.source == 4))
    assert np.all(~targets.loss_mask[targets.source == 4])
    assert np.all(targets.valid[targets.source == 4])
    assert np.all(targets.y[targets.source == 4] == 0)


def test_target_schema_v06_round_trip_persists_kind_and_loss_mask(tmp_path):
    trans, _ = make_chain(L=8, n_episodes=3, p_forward=1.0, seed=0)
    artifact = _artifact(trans)
    targets = critic_targets(trans, artifact, TargetConfig(h=2))
    save_targets(targets, tmp_path / "targets")
    metadata = json.loads((tmp_path / "targets.json").read_text())
    assert metadata["schema_version"] == "0.6"
    loaded = load_targets(tmp_path / "targets", graph=artifact)
    assert loaded.kind == "regret"
    np.testing.assert_array_equal(loaded.loss_mask, targets.loss_mask)
    np.testing.assert_array_equal(loaded.source, targets.source)


def test_qc_guard_rejects_legacy_target_artifact_without_loss_mask(tmp_path):
    arrays = {
        "y": np.asarray([1.0, 0.0], dtype=np.float32),
        "valid": np.asarray([True, False], dtype=np.bool_),
        "source": np.asarray([1, 3], dtype=np.uint8),
    }
    config = {
        "h": 1,
        "unit": "steps",
        "graph_hash": "a" * 64,
        "dataset_id": "toy:legacy",
    }
    np.savez(tmp_path / "legacy.npz", **arrays)
    metadata = {
        "schema_version": "0.3",
        "config": config,
        "meta": {},
        "content_hash": _content_hash("0.3", config, arrays),
        "arrays": [
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        ],
    }
    (tmp_path / "legacy.json").write_text(json.dumps(metadata))

    with pytest.raises(SchemaError, match="v0.6 target artifact with an explicit loss_mask"):
        load_targets(tmp_path / "legacy", require_loss_mask=True)

    legacy = load_targets(tmp_path / "legacy")
    assert legacy.kind == "cost"
    np.testing.assert_array_equal(legacy.loss_mask, legacy.valid)


def test_target_config_defaults_to_regret_and_rejects_unknown_kind():
    assert TargetConfig(h=5).kind == "regret"
    with pytest.raises(ValueError, match="kind"):
        TargetConfig.from_dict({"h": 5, "kind": "advantage"})
