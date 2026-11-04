import numpy as np

from graphcritic.config import BridgeConfig, BridgeCostConfig, GraphConfig
from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.modes import resolve_state_modes
from graphcritic.state_graph import build_state_graph
from graphcritic.types import SimStateSet
from graphcritic.verification import verify_bridge_edges
from graphcritic.weights import edge_weights, step_cost
from tests.test_v02 import _trajectories


class _ReplayVerifier:
    def __init__(self, nodes, edge_dst, *, fail=False):
        self.nodes = nodes
        self.edge_dst = edge_dst
        self.fail = fail

    def rollout(self, state, actions):
        source = int(state["qpos"][0])
        anchor = int(actions[0, 0])
        destination = self.edge_dst[(source, anchor)]
        observation = (
            np.full(self.nodes.rep_feat.shape[1], 100.0, dtype=np.float32)
            if self.fail else self.nodes.rep_feat[destination]
        )
        return {"observation": observation}


def _fixture():
    trans = _trajectories()
    trans.action = np.arange(trans.n_rows, dtype=np.float32)[:, None]
    trans.validate()
    cfg = GraphConfig(
        graph_kind="state",
        mode_fn="none",
        bridge=BridgeConfig(
            k=4,
            radius=0.5,
            radius_unit="step",
            landing_steps=5,
            min_time_gap=20,
            cost=BridgeCostConfig(policy="sim_verified", sim_budget=10_000),
        ),
    )
    states, _, _, _ = state_table(trans)
    featurizer = StateFeaturizer.fit(trans, cfg.feature)
    phi = featurizer.transform(states)
    nodes, edges, meta = build_state_graph(trans, cfg, phi)
    modes = resolve_state_modes(trans, cfg.mode_fn)
    sim_state = SimStateSet(
        qpos=np.arange(nodes.n_nodes, dtype=np.float32)[:, None],
        qvel=np.zeros((nodes.n_nodes, 1), dtype=np.float32),
    )
    bridge = edges.kind == 1
    edge_dst = {
        (int(src), int(anchor)): int(dst)
        for src, anchor, dst in zip(
            edges.src[bridge], edges.anchor[bridge], edges.dst[bridge], strict=True
        )
    }
    return trans, cfg, featurizer, nodes, edges, modes, sim_state, meta, edge_dst


def test_sim_verified_keeps_successful_used_bridges_and_has_no_temporal_penalty():
    trans, cfg, featurizer, nodes, edges, modes, sim_state, meta, edge_dst = _fixture()
    verified, result = verify_bridge_edges(
        trans, nodes, edges, cfg, featurizer, sim_state, modes,
        d_step=meta["d_step"], verifier=_ReplayVerifier(nodes, edge_dst),
    )

    bridge = verified.kind == 1
    temporal = verified.kind == 0
    assert result["sim_bridges_verified"] > 0
    assert np.all(verified.n_pos[bridge] == 1)
    weights = edge_weights(verified, cfg, d_step=meta["d_step"], meta=meta)
    np.testing.assert_allclose(weights[temporal], step_cost(cfg.gamma) * verified.kappa[temporal])
    np.testing.assert_allclose(weights[bridge], step_cost(cfg.gamma) * verified.kappa[bridge])


def test_sim_verified_drops_failed_and_unverified_bridges():
    trans, cfg, featurizer, nodes, edges, modes, sim_state, meta, edge_dst = _fixture()
    verified, result = verify_bridge_edges(
        trans, nodes, edges, cfg, featurizer, sim_state, modes,
        d_step=meta["d_step"], verifier=_ReplayVerifier(nodes, edge_dst, fail=True),
    )

    assert result["sim_bridges_failed"] > 0
    assert not np.any(verified.kind == 1)
