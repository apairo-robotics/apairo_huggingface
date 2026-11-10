"""The FoMo image footprint: the scan labels' robot model, drawn in the image."""

import numpy as np
from apairo.core.sample import Sample
from apairo_preprocess import TraversabilityFromTrajectory

from apairo_huggingface.fomo import config
from apairo_huggingface.fomo.calibration import INTRINSICS
from apairo_huggingface.fomo.labeling import _project_unclipped, corridor_mask, footprint_boxes

GROUND_Z = config.GROUND_OFFSET  # the poses are the lidar, LIDAR_HEIGHT above the ground
# A forward camera at the lidar origin, looking along the driving direction (-x).
CAM_FROM_WORLD = np.eye(4)
CAM_FROM_WORLD[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
ZED = INTRINSICS["zedx_left"]


def _poses(positions, yaw=None):
    out = np.tile(np.eye(4), (len(positions), 1, 1))
    out[:, :3, 3] = positions
    a = np.zeros(len(positions)) if yaw is None else np.asarray(yaw)
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
    return out


def _straight(n=300, step=0.15):
    return _poses(np.column_stack([-np.arange(1, n + 1) * step, np.zeros(n), np.zeros(n)]))


def _ground(x_max=45.0, y_max=6.0, step=0.1):
    x, y = np.meshgrid(-np.arange(1.0, x_max, step), np.arange(-y_max, y_max, step))
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, GROUND_Z)])


def _row(x):
    _, v, _ = _project_unclipped(np.array([[x, 0.0, GROUND_Z]]) @ CAM_FROM_WORLD[:3, :3].T, ZED)
    return int(v[0])


def _pixels(points):
    u, v, z = _project_unclipped(points @ CAM_FROM_WORLD[:3, :3].T, ZED)
    ok = (z > 0.5) & (u >= 0) & (u < 1920) & (v >= 0) & (v < 1200)
    return u.astype(int), v.astype(int), ok


def test_dense_ground_does_not_cut_the_footprint():
    mask = corridor_mask(footprint_boxes(_straight()), CAM_FROM_WORLD, _ground(), intrinsics=ZED)
    rows = np.flatnonzero(mask.any(axis=1))
    # the footprint reaches its full 30 m of path, not the first ground return
    assert rows.min() <= _row(-config.FORWARD_DISTANCE + 1.0)


def test_obstacle_hides_the_footprint_behind_it():
    boxes = footprint_boxes(_straight())
    yy, zz = np.meshgrid(np.arange(-0.6, 0.6, 0.05), np.arange(GROUND_Z + 0.4, 1.0, 0.05))
    wall = np.column_stack([np.full(yy.size, -12.0), yy.ravel(), zz.ravel()])
    clear = corridor_mask(boxes, CAM_FROM_WORLD, _ground(), intrinsics=ZED)
    blocked = corridor_mask(boxes, CAM_FROM_WORLD, np.vstack([_ground(), wall]), intrinsics=ZED)
    assert clear[_row(-20.0)].any() and not blocked[_row(-20.0)].any()
    assert blocked[_row(-6.0)].any()


def test_image_footprint_is_the_scan_labels_footprint():
    # a curving path with yaw noise; the scan labeler and the image use one robot model
    rng = np.random.default_rng(0)
    n = 250
    s = np.arange(1, n + 1) * 0.15
    heading = np.pi + 0.02 * s
    pos = np.column_stack([np.cumsum(np.cos(heading) * 0.15), np.cumsum(np.sin(heading) * 0.15), np.zeros(n)])
    poses = _poses(pos, yaw=heading + rng.normal(0, np.radians(1), n))
    ground = _ground(x_max=35.0, y_max=10.0, step=0.05)
    labeler = TraversabilityFromTrajectory(
        robot_length=config.ROBOT_LENGTH, robot_width=config.ROBOT_WIDTH,
        height_min=config.HEIGHT_MIN, height_max=config.HEIGHT_MAX,
        forward_distance=config.FORWARD_DISTANCE)
    frames = [Sample(data={"lidar": ground, "poses": np.eye(4)})]
    frames += [Sample(data={"lidar": np.zeros((0, 3)), "poses": p}) for p in poses]
    driven = np.asarray(labeler(iter(frames))[0], bool)
    mask = corridor_mask(footprint_boxes(poses), CAM_FROM_WORLD, ground, intrinsics=ZED)
    u, v, ok = _pixels(ground)
    in_mask = np.zeros(len(ground), bool)
    in_mask[ok] = mask[v[ok], u[ok]]
    near = ok & (-ground[:, 0] < 15.0)
    assert driven[near].sum() > 1000
    assert in_mask[ok & driven].mean() > 0.99
    # Beyond ~15 m one pixel row spans about a metre of ground at grazing
    # angles, so pixels on the box edges also hold points just outside them.
    assert driven[near & in_mask].mean() > 0.98


def test_height_jitter_does_not_zigzag_the_footprint():
    rng = np.random.default_rng(0)
    poses = _straight()
    poses[:, 2, 3] += rng.normal(0, 0.05, len(poses))
    boxes = footprint_boxes(poses)
    assert np.abs(boxes[:, 0, 2] - GROUND_Z).max() < 0.05
