import numpy as np
from scipy.spatial.transform import Rotation

from apairo_huggingface.fomo.verify import driving_direction, pose_at


def _poses(yaw_deg, steps):
    """A robot driving along its lidar's -x, as on FoMo, with a fixed heading."""
    P = np.tile(np.eye(4), (len(steps), 1, 1))
    R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    P[:, :3, :3] = R
    P[:, :3, 3] = np.asarray(steps)[:, None] * (R @ [-1.0, 0.0, 0.0])
    return P


def test_driving_direction_is_expressed_in_the_lidar_frame():
    P = _poses(37.0, np.arange(20) * 0.2)
    np.testing.assert_allclose(driving_direction(P), [-1.0, 0.0, 0.0], atol=1e-9)


def test_pose_at_interpolates_between_bracketing_poses():
    P = _poses(10.0, [0.0, 1.0, 2.0])
    P[2, :3, :3] = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    T = pose_at(1.5, np.array([0.0, 1.0, 2.0]), P)
    np.testing.assert_allclose(T[:3, 3], 0.5 * (P[1, :3, 3] + P[2, :3, 3]))
    assert abs(Rotation.from_matrix(T[:3, :3]).as_euler("zyx", degrees=True)[0] - 20.0) < 1e-6
