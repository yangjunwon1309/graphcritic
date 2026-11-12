import numpy as np

from experiments.build_candidate_cache import _robot_only_windows


def test_robot_only_window_features_support_four_by_four_puzzle_width():
    # 4x4 Puzzle state frames are 19 robot values followed by 16 button bits.
    frames = np.arange(2 * 5 * 35, dtype=np.float32).reshape(2, 5, 35)
    flattened = frames.reshape(2, -1)
    robot = _robot_only_windows(flattened, frame_dim=35, robot_dim=19)
    expected = frames[:, :, :19].reshape(2, -1)
    np.testing.assert_array_equal(robot, expected)
