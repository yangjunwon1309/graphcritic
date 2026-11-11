"""v1.1 observed object-change candidates and frozen-distance contracts."""

import numpy as np

from experiments.build_candidate_cache import (
    _robot_only_windows,
    select_object_candidate_pairs,
)
from experiments.build_object_change_set import build_change_set
from experiments.eval_bridges import (
    CANDIDATE_RULE,
    _array_sha256,
    load_or_build_window_candidate_pairs,
)
from graphcritic.bridges.object_state import object_change, observed_object_change_counts
from experiments.write_direct_q_config import _is_v11_same_object_graph
from envs.toy import make_chain
from experiments.train_action_quasimetric import embed_state_windows, flatten_state_windows


def test_observed_change_set_keeps_frequent_patterns_and_omits_rare_ones():
    states = []
    episodes = []
    times = []
    # Ten independent observations of pattern 1 and two of pattern 2.
    for episode_id, code in enumerate([1] * 10 + [2] * 2):
        before = np.zeros(9, dtype=np.uint8)
        after = before.copy()
        after[code - 1] = 1
        states.extend((before, after))
        episodes.extend((episode_id, episode_id))
        times.extend((0, 1))
    report = build_change_set(
        np.asarray(states), np.asarray(episodes), np.asarray(times),
        max_horizon=1, min_count=10,
    )
    assert [item["code"] for item in report["patterns"]] == [1]
    assert report["patterns"][0]["count"] == 10
    assert report["patterns"][0]["min_k"] == 1
    assert report["n_patterns_omitted_below_min_count"] == 1
    assert report["occurrences_omitted_below_min_count"] == 2
    np.testing.assert_array_equal(
        object_change(np.asarray([0, 1, 0]), np.asarray([1, 1, 1])), [1, 0, 1]
    )


def test_observed_change_counts_use_only_same_episode_frame_offsets():
    boards = np.asarray([[0], [1], [1], [0]], dtype=np.uint8)
    episodes = np.asarray([0, 0, 1, 1])
    times = np.asarray([0, 1, 0, 1])
    codes, counts, min_k = observed_object_change_counts(
        boards, episodes, times, max_horizon=1,
    )
    np.testing.assert_array_equal(codes, [1])
    np.testing.assert_array_equal(counts, [2])
    np.testing.assert_array_equal(min_k, [1])


def test_object_candidates_are_cross_episode_allowed_and_robot_only():
    frames = np.zeros((4, 5, 28), dtype=np.float32)
    frames[:, :, 0] = np.asarray([[0], [0.1], [0.2], [0.05]])
    frames[1, :, 19:] = 1.0
    frames[2, :, 19:] = 1.0
    frames[3, :, 19:] = 1.0
    robot = _robot_only_windows(frames.reshape(4, -1))
    np.testing.assert_allclose(robot[0], robot[1] - np.tile(
        np.r_[0.1, np.zeros(18)], 5
    ), atol=2e-8, rtol=0)
    # Changing only button bits must not affect the metric input.
    modified = frames.copy()
    modified[:, :, 19:] = 0.0
    np.testing.assert_array_equal(robot, _robot_only_windows(modified.reshape(4, -1)))

    pairs, deltas = select_object_candidate_pairs(
        robot, np.asarray([10, 11, 12, 13]), np.asarray([0, 1, 2, 3]),
        np.asarray([0, 0, 1, 2]), np.asarray([True, False, False, False]),
        np.asarray([0, 1]), knn=1, knn_changed=1,
    )
    np.testing.assert_array_equal(pairs, [[10, 11], [10, 12]])
    np.testing.assert_array_equal(deltas, [0, 1])
    np.testing.assert_array_equal(np.asarray([0, 0, 1, 2])[pairs[:, 0] - 10] !=
                                  np.asarray([0, 1, 2, 3])[pairs[:, 1] - 10], [True, True])
    assert np.all(np.isin(deltas, [0, 1]))


def test_legacy_candidate_rule_remains_pinned():
    # The new candidate mode is additive; the v0.8 cache rule string is a
    # compatibility field and must remain stable for existing cache hashes.
    assert CANDIDATE_RULE == "cross_episode_incl_goal_v2"


def test_legacy_candidate_cache_pair_hash_is_unchanged(tmp_path, monkeypatch):
    import experiments.eval_bridges as eval_bridges

    trans, _ = make_chain(L=8, n_episodes=3, p_forward=1.0, seed=19)
    terminal_rows = np.flatnonzero(trans.terminal)
    goals = np.concatenate((trans.success_cur, trans.success[terminal_rows]))
    features = np.arange(len(goals) * 3, dtype=np.float32).reshape(len(goals), 3)

    def deterministic_topk(values, episode_ids, *, k, batch_size, progress=False):
        rows = []
        for source in range(len(values)):
            eligible = np.flatnonzero(episode_ids != episode_ids[source])
            delta = values[eligible].astype(np.float64) - values[source].astype(np.float64)
            distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            order = np.lexsort((eligible, distance))[:k]
            rows.extend((source, int(eligible[index])) for index in order)
        return np.asarray(rows, dtype=np.int64)

    monkeypatch.setattr(eval_bridges, "_gpu_cross_episode_topk", deterministic_topk)
    path = tmp_path / "legacy_candidates.npz"
    pairs = load_or_build_window_candidate_pairs(
        features, trans, goals, dataset_id="legacy-hash-test", window=1,
        k=1, cache_path=path,
    )
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive["candidate_rule"].item()) == "cross_episode_incl_goal_v2_boundary_window1"
        assert "object_change_set_sha256" not in archive.files
    assert _array_sha256(pairs) == "b0b287d66bf2a31623887bf3a2b55afe6059df364d6433ab3a6ccbfa5e4af007"


def test_learned_v11_graph_is_eligible_for_same_object_anchors():
    graph_meta = {"target_provenance": {"kind": "learned"}}
    build_result = {
        "candidate_rule": "cross_episode_observed_object_change_v1_robot_window5_same32_changed32",
        "object_change_set_sha256": "a" * 64,
    }
    assert _is_v11_same_object_graph(graph_meta, build_result)
    assert not _is_v11_same_object_graph(
        {"target_provenance": {"kind": "oracle_upper_bound"}}, build_result,
    )


def test_action_head_windows_flatten_in_the_same_time_major_order():
    class Embedder:
        @staticmethod
        def apply(_variables, values):
            return values

    windows = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    flattened = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]],
                           dtype=np.float32)
    np.testing.assert_array_equal(flatten_state_windows(windows), flattened)
    np.testing.assert_array_equal(
        embed_state_windows(Embedder(), {}, windows),
        embed_state_windows(Embedder(), {}, flattened),
    )
