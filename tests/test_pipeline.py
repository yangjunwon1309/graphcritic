"""Artifact assembly, diagnostics, and CLI acceptance tests."""

import json

import numpy as np
import pytest

from envs.toy import make_chain
from experiments.build_graph import main
from graphcritic.config import GraphConfig
from graphcritic.diagnostics import report
from graphcritic.edges import build_edges
from graphcritic.featurize import Standardizer
from graphcritic.nodes import build_nodes
from graphcritic.pipeline import build_artifact, rebuild_from_external
from graphcritic.targets import critic_targets
from graphcritic.config import TargetConfig


def test_pipeline_artifact_hash_and_external_hop_rebuild_are_deterministic():
    trans, info = make_chain(L=10, n_episodes=4, p_forward=1.0, seed=7)
    cfg = GraphConfig(graph_kind="node", radius=0.25, compress=False, weight_policy="time")
    one = build_artifact(trans, cfg, dataset_id="chain", source="test")
    two = build_artifact(trans, cfg, dataset_id="chain", source="test")
    assert one.content_hash == two.content_hash
    np.testing.assert_array_equal(one.cost, two.cost)

    hop_cfg = GraphConfig(graph_kind="node", radius=0.25, compress=False, weight_policy="hop")
    stdz = Standardizer.fit(trans, hop_cfg.feature)
    nodes = build_nodes(trans, hop_cfg, stdz)
    edges = build_edges(trans, nodes, hop_cfg)
    external = rebuild_from_external(
        trans, nodes, edges, hop_cfg, dataset_id="chain", source="synthetic external graph"
    )
    for node, feature in enumerate(external.nodes.rep_feat[:, 0]):
        np.testing.assert_allclose(external.cost[node], info["d_star"][(int(feature),)], atol=1e-12)


def test_report_uses_next_success_episodes_and_target_unit_statistics():
    trans, _ = make_chain(L=6, n_episodes=2, p_forward=1.0, seed=1)
    cfg = GraphConfig(graph_kind="node", radius=0.25, compress=False, weight_policy="time")
    artifact = build_artifact(trans, cfg, dataset_id="chain", source="test")
    targets = critic_targets(trans, artifact, TargetConfig(h=2, unit="nats"))
    summary = report(trans, artifact, targets)
    assert summary["n_success_episodes"] == 2
    assert summary["monotonicity"] == pytest.approx(1.0)
    assert summary["rank_corr"] == pytest.approx(1.0)
    assert summary["targets"]["valid_frac"] == pytest.approx(np.mean(targets.valid))
    np.testing.assert_allclose(
        summary["targets"]["y_mean"], targets.y[targets.valid].mean(), rtol=1e-7
    )
    json.dumps(summary, allow_nan=False)


def test_build_graph_toy_cli_writes_pairs_targets_diagnostics_and_stable_hash(tmp_path, capsys):
    cfg_path = tmp_path / "graph.json"
    GraphConfig(graph_kind="node", radius=0.25, compress=False, weight_policy="time").to_yaml(cfg_path)
    hashes = []
    for output in (tmp_path / "run-a", tmp_path / "run-b"):
        assert main([
            "--toy", "chain",
            "--config", str(cfg_path),
            "--h", "1",
            "--out", str(output),
            "--seed", "3",
        ]) == 0
        payload = json.loads(capsys.readouterr().out)
        hashes.append(payload["graph_hash"])
        assert (output / "graph.npz").is_file()
        assert (output / "graph.json").is_file()
        assert (output / "targets.npz").is_file()
        assert (output / "targets.json").is_file()
        diagnostics = json.loads((output / "diagnostics.json").read_text())
        assert diagnostics["monotonicity"] == pytest.approx(1.0)
        assert diagnostics["rank_corr"] == pytest.approx(1.0)
    assert hashes[0] == hashes[1]
