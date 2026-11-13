"""v1.6 gamma-power graph target checks."""

import json

import numpy as np

from envs.toy import make_chain
from experiments.build_judged_graph import _parse
from graphcritic.config import GraphConfig, TargetConfig
from graphcritic.io import load_targets, save_targets
from graphcritic.pipeline import build_artifact
from graphcritic.targets import critic_targets


def test_gamma_pow_is_complement_of_discounted_target_and_roundtrips(tmp_path):
    gamma = 0.999
    transitions, _ = make_chain(L=9, n_episodes=4, p_forward=1.0, seed=16)
    artifact = build_artifact(
        transitions,
        GraphConfig(graph_kind='node', radius=0.25, compress=False, weight_policy='time'),
        dataset_id='unit:gamma-pow', source='test',
    )
    graph_hash = artifact.content_hash
    discounted = critic_targets(
        transitions, artifact,
        TargetConfig(h=5, unit='steps', kind='discounted', discount_gamma=gamma),
    )
    gamma_pow = critic_targets(
        transitions, artifact,
        TargetConfig(h=5, unit='steps', kind='gamma_pow', discount_gamma=gamma),
    )

    assert artifact.content_hash == graph_hash
    assert gamma_pow.kind == 'gamma_pow'
    assert gamma_pow.h == discounted.h == 5
    assert gamma_pow.discount_gamma == gamma
    np.testing.assert_array_equal(gamma_pow.valid, discounted.valid)
    np.testing.assert_array_equal(gamma_pow.loss_mask, discounted.loss_mask)
    np.testing.assert_allclose(
        gamma_pow.y[gamma_pow.valid],
        1.0 - (1.0 - gamma) * discounted.y[discounted.valid],
        rtol=2e-5, atol=2e-6,
    )

    path = tmp_path / 'targets.npz'
    save_targets(gamma_pow, path)
    restored = load_targets(path, expected_dataset_id='unit:gamma-pow', require_loss_mask=True)
    np.testing.assert_array_equal(restored.y, gamma_pow.y)
    assert restored.kind == 'gamma_pow'
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert metadata['config']['kind'] == 'gamma_pow'
    assert metadata['config']['h'] == 5
    assert metadata['config']['gamma'] == gamma


def test_builder_accepts_gamma_pow_with_explicit_horizon_and_default_stays_regret():
    common = [
        '--judge', 'quasimetric', '--g0-graph', 'g0.npz',
        '--transitions', 'transitions.npz', '--sim-state', 'sim_state.npz',
        '--checkpoint', 'checkpoint', '--candidate-cache', 'candidates.npz', '--out', 'out',
    ]
    assert _parse(common).target_kind == 'regret'
    args = _parse([*common, '--target-kind', 'gamma_pow', '--target-h', '25',
                   '--discount-gamma', '0.999'])
    assert args.target_kind == 'gamma_pow'
    assert args.target_h == 25
    assert args.discount_gamma == 0.999
