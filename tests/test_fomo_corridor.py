"""The FoMo image corridor: ground returns never hide it, obstacles do."""

import numpy as np

from apairo_huggingface.fomo import config
from apairo_huggingface.fomo.calibration import INTRINSICS
from apairo_huggingface.fomo.labeling import _project_unclipped, corridor_mask, corridor_world

GROUND_Z = config.GROUND_OFFSET  # the poses are the lidar, LIDAR_HEIGHT above the ground
# A forward camera at the lidar origin, looking along the driving direction (-x).
CAM_FROM_WORLD = np.eye(4)
CAM_FROM_WORLD[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
ZED = INTRINSICS["zedx_left"]


def _straight_corridor():
    path = np.column_stack([-np.arange(0.5, 40.0, 0.5), np.zeros(79), np.zeros(79)])
    return corridor_world(path, np.zeros(3))


def _ground(x_max=45.0):
    x, y = np.meshgrid(-np.arange(1.0, x_max, 0.1), np.arange(-6.0, 6.0, 0.1))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, GROUND_Z)])


def _row(x):
    _, v, _ = _project_unclipped(np.array([[x, 0.0, GROUND_Z]]) @ CAM_FROM_WORLD[:3, :3].T, ZED)
    return int(v[0])


def test_dense_ground_does_not_cut_the_corridor():
    left, right = _straight_corridor()
    mask = corridor_mask(left, right, CAM_FROM_WORLD, _ground(), intrinsics=ZED)
    rows = np.flatnonzero(mask.any(axis=1))
    # the corridor reaches its full 30 m of path, not the first ground return
    assert rows.min() <= _row(-config.FORWARD_DISTANCE + 1.0)


def test_obstacle_hides_the_corridor_behind_it():
    left, right = _straight_corridor()
    yy, zz = np.meshgrid(np.arange(-0.6, 0.6, 0.05), np.arange(GROUND_Z + 0.4, 1.0, 0.05))
    wall = np.column_stack([np.full(yy.size, -12.0), yy.ravel(), zz.ravel()])
    clear = corridor_mask(left, right, CAM_FROM_WORLD, _ground(), intrinsics=ZED)
    blocked = corridor_mask(left, right, CAM_FROM_WORLD, np.vstack([_ground(), wall]), intrinsics=ZED)
    far = _row(-20.0)
    near = _row(-6.0)
    assert clear[far].any() and not blocked[far].any()
    assert blocked[near].any()


def test_pose_jitter_does_not_zigzag_the_edges():
    # 10 Hz poses on a straight road, with odometry noise of a few centimetres
    rng = np.random.default_rng(0)
    n = 200
    path = np.column_stack([-np.arange(1, n + 1) * 0.15,
                            rng.normal(0, 0.03, n), rng.normal(0, 0.05, n)])
    left, right = corridor_world(path, np.zeros(3))
    width = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
    assert np.allclose(width, 2 * config.ROBOT_HALF_WIDTH)
    # the edges stay within a few centimetres of the true road, laterally and in height
    assert np.abs(np.abs(left[:, 1]) - config.ROBOT_HALF_WIDTH).max() < 0.06
    assert np.abs(left[:, 2] - GROUND_Z).max() < 0.06
