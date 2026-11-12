"""v1.2b structural residual-head guarantees and direct-Q lower bound."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.train_action_quasimetric import _parse
from experiments.v08_runtime import iqe_distance_from_embeddings
from graphcritic.direct_q import (
    DirectQScorer,
    _make_action_head,
    parameterize_action_embedding,
)


def _distance(source, target, raw_alpha=0.0):
    return iqe_distance_from_embeddings(
        source, target, jnp.asarray(raw_alpha),
        n_components=64, component_dim=32, jnp=jnp,
    )


def test_residual_parameterization_satisfies_c2_and_dominates_source_distance():
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(13), 3)
    psi = jax.random.normal(key1, (7, 2048))
    raw_h = jax.random.normal(key2, (7, 2048))
    y = jax.random.normal(key3, (7, 2048))
    phi = parameterize_action_embedding(raw_h, psi, "residual")

    c2 = _distance(psi, phi)
    np.testing.assert_array_less(np.asarray(c2), np.full((7,), 1e-5))
    residual_distance = np.asarray(_distance(phi, y))
    source_distance = np.asarray(_distance(psi, y))
    assert np.all(residual_distance >= source_distance - 1e-5)


def test_residual_head_initialization_starts_near_the_state_distance():
    head = _make_action_head("residual")
    action_input = jnp.zeros((1, 17), dtype=jnp.float32)
    params = head.init(jax.random.PRNGKey(3), action_input)["params"]
    raw_h = head.apply({"params": params}, action_input)
    assert np.allclose(np.asarray(raw_h), -5.0)

    key1, key2 = jax.random.split(jax.random.PRNGKey(5))
    psi_start = jax.random.normal(key1, (1, 2048))
    psi_end = jax.random.normal(key2, (1, 2048))
    phi = parameterize_action_embedding(raw_h, psi_start, "residual")
    d_phi = float(_distance(phi, psi_end)[0])
    d_state = float(_distance(psi_start, psi_end)[0])
    assert abs(d_phi - d_state) < 0.1


def test_direct_q_residual_score_is_lower_bounded_by_state_distance_plus_cost():
    class IdentityEmbedder:
        @staticmethod
        def apply(_variables, values):
            return values

    class SmallResidualHead:
        @staticmethod
        def apply(_variables, values):
            return jnp.full((len(values), 2048), -5.0, dtype=jnp.float32)

    rng = np.random.default_rng(9)
    features = rng.normal(0.0, 0.1, size=(3, 2048)).astype(np.float32)
    boards = np.zeros((3, 1), dtype=np.uint8)
    costs = np.asarray([3.0, 1.0, 2.0], dtype=np.float32)
    head = SmallResidualHead()
    scorer = DirectQScorer(
        candidate_ids=np.arange(3, dtype=np.int64),
        candidate_features=features,
        candidate_boards=boards,
        candidate_embeddings=features,
        episode_ids=np.asarray([0, 1, 2], dtype=np.int64),
        times=np.asarray([0, 0, 0], dtype=np.int64),
        cost_steps=costs,
        state_mean=np.zeros(19), state_std=np.ones(19),
        n_buttons=1, action_dim=1, horizon=5,
        action_heads=[(head, {}) for _ in range(4)],
        raw_alpha=jnp.asarray(0.0),
        anchor_mode="same_object",
        action_parameterization="residual",
        state_embedder=IdentityEmbedder(), state_params={},
        k=32, successors=0, max_local_distance=1e6,
    )
    window = rng.normal(0.0, 0.1, size=(2048,)).astype(np.float32)
    chunks = np.zeros((2, 5, 1), dtype=np.float32)
    q_values, _ = scorer.score(window, chunks, board=np.zeros(1, dtype=np.uint8))

    psi = jnp.asarray(window[None, :])
    target = jnp.asarray(features)
    state_distances = np.asarray(_distance(jnp.broadcast_to(psi, target.shape), target))
    lower_bound = float(np.min(state_distances + costs))
    assert np.all(q_values >= lower_bound - 1e-5)


def test_action_head_parameterization_defaults_to_residual_and_accepts_free():
    args = _parse([
        "--transitions", "transitions.npz", "--sim-state", "sim.npz",
        "--state-checkpoint", "checkpoint", "--out", "out",
    ])
    assert args.parameterization == "residual"
    free = _parse([
        "--transitions", "transitions.npz", "--sim-state", "sim.npz",
        "--state-checkpoint", "checkpoint", "--out", "out",
        "--parameterization", "free",
    ])
    assert free.parameterization == "free"
