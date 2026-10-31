"""Schema, validation, and safe persistence tests for graphcritic v0.1."""

import json

import numpy as np
import pytest

from graphcritic.config import FeatureConfig, GraphConfig, PessimismConfig, TargetConfig
from graphcritic.io import (
    load_graph,
    load_targets,
    load_transitions,
    save_graph,
    save_targets,
    save_transitions,
)
from graphcritic.types import EdgeSet, GraphArtifact, NodeSet, SchemaError, TargetSet, TransitionSet


def make_transitions() -> TransitionSet:
    return TransitionSet(
        feat=[[0.0], [1.0], [2.0]],
        next_feat=[[1.0], [2.0], [3.0]],
        episode_id=[7, 7, 7],
        t=[0, 1, 2],
        success=[False, True, False],
        terminal=[False, False, True],
        row_idx=[10, 11, 12],
    )


def make_nodes() -> NodeSet:
    return NodeSet(
        rep_feat=[[0.0], [1.0], [2.0], [3.0]],
        rep_ref=[[0, 0], [0, 1], [1, 1], [2, 1]],
        assign_cur=[0, 1, 2],
        assign_next=[1, 2, 3],
        is_goal=[False, False, True, False],
    )


def make_edges() -> EdgeSet:
    return EdgeSet(
        src=[0, 1, 2],
        dst=[1, 2, 3],
        kappa=[1, 1, 1],
        n_pos=[1, 1, 1],
        n_neg=[0, 0, 0],
        kind=[0, 0, 0],
    )


def make_graph(
    trans: TransitionSet | None = None,
    *,
    weights=(1.0, 1.0, 1.0),
    cost=(2.0, 1.0, 0.0, np.inf),
    succ=(1, 2, -1, -1),
    feature_mean=(1.5,),
) -> GraphArtifact:
    if trans is None:
        trans = make_transitions()
    return GraphArtifact(
        config=GraphConfig(
            gamma=0.99,
            radius=0.5,
            feature=FeatureConfig(name="identity", standardize=True),
            pessimism=PessimismConfig(mode="union", delta=0.05),
        ),
        feature_stats={"mean": feature_mean, "std": [1.1180339]},
        nodes=make_nodes(),
        edges=make_edges(),
        weights=weights,
        cost=cost,
        succ=succ,
        meta={
            "dataset_id": "toy-chain-test",
            "source": "test",
            "core_version": "0.1.0",
            "n_rows": trans.n_rows,
            "c_max_steps": 4,
        },
    )


def make_targets(graph: GraphArtifact | None = None) -> TargetSet:
    graph = graph or make_graph()
    return TargetSet(
        y=[3.0, 1.0, 0.0],
        valid=[True, True, False],
        h=2,
        unit="steps",
        graph_hash=graph.content_hash,
        dataset_id=graph.meta["dataset_id"],
    )


def test_schema_coerces_to_declared_dtypes_and_checks_shapes():
    trans = make_transitions()
    assert trans.feat.dtype == np.float32
    assert trans.episode_id.dtype == np.int64
    assert trans.success.dtype == np.bool_
    assert trans.row_idx.shape == (3,)
    assert make_nodes().rep_feat.dtype == np.float32
    assert make_edges().kind.dtype == np.int8

    with pytest.raises(SchemaError, match="shape|feature dimension"):
        TransitionSet(
            feat=[[0.0], [1.0]],
            next_feat=[[1.0]],
            episode_id=[0, 0],
            t=[0, 1],
            success=[False, False],
            terminal=[False, True],
            row_idx=[0, 1],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"episode_id": [0, 1, 0]}, "contiguous"),
        ({"t": [0, 2, 2]}, "increase by one"),
        ({"terminal": [True, False, True]}, "final row"),
    ],
)
def test_transition_episode_invariants(changes, message):
    values = {
        "feat": [[0.0], [1.0], [2.0]],
        "next_feat": [[1.0], [2.0], [3.0]],
        "episode_id": [0, 0, 1],
        "t": [0, 1, 0],
        "success": [False, False, False],
        "terminal": [False, True, True],
        "row_idx": [0, 1, 2],
    }
    values.update(changes)
    with pytest.raises(SchemaError, match=message):
        TransitionSet(**values)


def test_next_feature_disagreement_warns_but_is_allowed():
    with pytest.warns(UserWarning, match="next_feat differs"):
        TransitionSet(
            feat=[[0.0], [1.5]],
            next_feat=[[1.0], [2.0]],
            episode_id=[0, 0],
            t=[0, 1],
            success=[False, False],
            terminal=[False, True],
            row_idx=[0, 1],
        )


def test_node_assignments_and_goal_membership_are_checked_against_rows():
    trans = make_transitions()
    nodes = make_nodes()
    nodes.assign_next[0] = 2
    nodes.rep_ref[1, 0] = -1
    with pytest.raises(SchemaError, match="assign_next"):
        nodes.validate(trans)

    nodes = make_nodes()
    nodes.is_goal[1] = True
    with pytest.raises(SchemaError, match="goal-node membership"):
        nodes.validate(trans)

    with pytest.raises(SchemaError, match="out-of-range"):
        NodeSet(
            rep_feat=[[0.0]], rep_ref=[[0, 0]], assign_cur=[0], assign_next=[1], is_goal=[False]
        )

    nodes = make_nodes()
    nodes.rep_ref[1] = [2, 0]
    nodes.rep_feat[1] = trans.feat[2]
    with pytest.raises(SchemaError, match="rep_ref"):
        nodes.validate(trans)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"src": [0], "dst": [0], "kappa": [1], "n_pos": [0], "n_neg": [0], "kind": [0]}, "self-edges"),
        ({"src": [0, 0], "dst": [1, 1], "kappa": [1, 1], "n_pos": [0, 0], "n_neg": [0, 0], "kind": [0, 0]}, "unique"),
        ({"src": [0], "dst": [1], "kappa": [0], "n_pos": [0], "n_neg": [0], "kind": [0]}, "kappa"),
        ({"src": [0], "dst": [1], "kappa": [1], "n_pos": [-1], "n_neg": [0], "kind": [0]}, "non-negative"),
        ({"src": [0], "dst": [1], "kappa": [1], "n_pos": [0], "n_neg": [0], "kind": [2]}, "kind"),
        ({"src": [0, 1], "dst": [1], "kappa": [1], "n_pos": [0], "n_neg": [0], "kind": [0]}, "shape"),
    ],
)
def test_edge_invariants(values, message):
    with pytest.raises(SchemaError, match=message):
        EdgeSet(**values)


def test_graph_invariants_and_hash_staleness():
    graph = make_graph()
    graph.validate(make_transitions())
    assert graph.content_hash == graph.compute_hash()

    with pytest.raises(SchemaError, match="strictly positive"):
        GraphArtifact(
            config=GraphConfig(radius=0.5), feature_stats={"mean": [0.0], "std": [1.0]},
            nodes=make_nodes(), edges=make_edges(), weights=[1.0, 0.0, 1.0],
            cost=[2.0, 1.0, 0.0, np.inf], succ=[1, 2, -1, -1],
            meta={"dataset_id": "x", "source": "test", "core_version": "0.1", "n_rows": 3, "c_max_steps": 4},
        )

    with pytest.raises(SchemaError, match="non-negative"):
        make_graph(cost=[2.0, -1.0, 0.0, np.inf])
    with pytest.raises(SchemaError, match="goal-node costs"):
        make_graph(cost=[2.0, 1.0, 0.5, np.inf])
    with pytest.raises(SchemaError, match="finite and strictly positive"):
        make_graph(weights=[1.0, np.inf, 1.0])
    with pytest.raises(SchemaError, match="successor"):
        make_graph(succ=[-1, 2, -1, -1])
    assert make_graph(feature_mean=[1.6]).content_hash != graph.content_hash

    graph.weights[0] = 2.0
    with pytest.raises(SchemaError, match="stale"):
        graph.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 1.0},
        {"radius": 0.0},
        {"consistency": True},
        {"weight_policy": "unknown"},
    ],
)
def test_invalid_graph_configuration(kwargs):
    with pytest.raises(SchemaError):
        GraphConfig(**kwargs)


def test_fixed_pessimism_requires_lambda():
    with pytest.raises(SchemaError, match="lam"):
        PessimismConfig(mode="fixed", delta=0.05)
    with pytest.raises(SchemaError, match="delta"):
        PessimismConfig(delta=0.0)


def test_artifact_requires_feature_stats_and_complete_metadata():
    graph = make_graph()
    with pytest.raises(SchemaError, match="feature_stats"):
        GraphArtifact(
            config=graph.config, feature_stats={"mean": [0.0]}, nodes=graph.nodes,
            edges=graph.edges, weights=graph.weights, cost=graph.cost, succ=graph.succ,
            meta=graph.meta,
        )
    incomplete = dict(graph.meta)
    incomplete.pop("dataset_id")
    with pytest.raises(SchemaError, match="meta"):
        GraphArtifact(
            config=graph.config, feature_stats=graph.feature_stats, nodes=graph.nodes,
            edges=graph.edges, weights=graph.weights, cost=graph.cost, succ=graph.succ,
            meta=incomplete,
        )


def test_target_validity_and_invalid_rows_are_zero_filled():
    graph = make_graph()
    make_targets(graph).validate(make_transitions())
    with pytest.raises(SchemaError, match="filled with zero"):
        TargetSet(y=[1.0], valid=[False], h=1, unit="steps", graph_hash="g", dataset_id="d")


def test_graph_config_dict_and_yaml_round_trip(tmp_path):
    config = GraphConfig(
        gamma=0.97,
        radius=0.2,
        feature=FeatureConfig(name="state", standardize=True),
        pessimism=PessimismConfig(mode="fixed", delta=0.1, lam=4.0, prior_data=(2, 1)),
        c_max_steps=200,
    )
    assert GraphConfig.from_dict(config.to_dict()) == config
    yaml_path = tmp_path / "graph.yaml"
    serialized = config.to_yaml(yaml_path)
    assert GraphConfig.from_yaml(serialized) == config
    assert GraphConfig.from_yaml(yaml_path) == config
    assert TargetConfig.from_dict({"h": 5, "unit": "steps"}) == TargetConfig(5)


def test_transition_round_trip_hash_and_pickle_free_io(tmp_path):
    trans = make_transitions()
    first_hash = save_transitions(trans, tmp_path / "transitions", dataset_id="toy", source="test")
    second_hash = save_transitions(trans, tmp_path / "copy.npz", dataset_id="toy", source="test")
    assert first_hash == second_hash
    loaded, metadata = load_transitions(tmp_path / "transitions.npz", expected_dataset_id="toy")
    assert metadata["n_rows"] == trans.n_rows
    for field in ("feat", "next_feat", "episode_id", "t", "success", "terminal", "row_idx"):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(trans, field))
    with np.load(tmp_path / "transitions.npz", allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)

    changed = make_transitions()
    changed.feat[0, 0] = 0.25
    assert save_transitions(changed, tmp_path / "changed", dataset_id="toy") != first_hash


def test_graph_and_target_round_trip_and_hash_validation(tmp_path):
    trans = make_transitions()
    graph = make_graph(trans)
    graph_hash = save_graph(graph, tmp_path / "graph")
    assert graph_hash == graph.content_hash
    loaded_graph = load_graph(tmp_path / "graph.json", trans=trans)
    assert loaded_graph.content_hash == graph.content_hash
    np.testing.assert_array_equal(loaded_graph.nodes.assign_cur, graph.nodes.assign_cur)
    np.testing.assert_array_equal(loaded_graph.edges.kind, graph.edges.kind)
    np.testing.assert_array_equal(loaded_graph.cost, graph.cost)
    assert loaded_graph.config == graph.config
    for field in ("rep_feat", "rep_ref", "assign_cur", "assign_next", "is_goal"):
        np.testing.assert_array_equal(getattr(loaded_graph.nodes, field), getattr(graph.nodes, field))
    for field in ("src", "dst", "kappa", "n_pos", "n_neg", "kind"):
        np.testing.assert_array_equal(getattr(loaded_graph.edges, field), getattr(graph.edges, field))
    np.testing.assert_array_equal(loaded_graph.weights, graph.weights)
    np.testing.assert_array_equal(loaded_graph.succ, graph.succ)
    for name in ("mean", "std"):
        np.testing.assert_array_equal(loaded_graph.feature_stats[name], graph.feature_stats[name])

    targets = make_targets(graph)
    target_hash = save_targets(targets, tmp_path / "targets")
    loaded_targets = load_targets(tmp_path / "targets", graph=loaded_graph)
    assert save_targets(loaded_targets, tmp_path / "target-copy") == target_hash
    np.testing.assert_array_equal(loaded_targets.y, targets.y)
    np.testing.assert_array_equal(loaded_targets.valid, targets.valid)

    metadata_path = tmp_path / "graph.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["content_hash"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SchemaError, match="content_hash"):
        load_graph(tmp_path / "graph")


def test_schema_version_mismatch_is_rejected(tmp_path):
    save_transitions(make_transitions(), tmp_path / "data", dataset_id="toy")
    json_path = tmp_path / "data.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "99"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SchemaError, match="schema_version"):
        load_transitions(tmp_path / "data")
