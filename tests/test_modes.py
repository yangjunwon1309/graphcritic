import numpy as np

import graphcritic.modes as modes


def test_ogbench_grasp_uses_scene_observation_layout(monkeypatch):
    monkeypatch.setattr(modes, "_valley_threshold", lambda values: 0.5)
    feat = np.zeros((2, 40), dtype=np.float32)
    feat[:, 12:15] = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    feat[:, 17] = [0.0, 1.0]  # gripper opening
    feat[:, 18] = [1.0, 0.0]  # gripper contact
    feat[:, 19:22] = np.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

    grasped = modes._ogbench_grasp(feat, n_cubes=1)

    np.testing.assert_array_equal(grasped, [True, False])
