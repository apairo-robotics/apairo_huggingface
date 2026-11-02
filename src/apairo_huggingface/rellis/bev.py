"""Robot-centred bird's-eye-view grids for every RELLIS-3D scan, as apairo derived channels.

RELLIS-3D ships no BEV labels, yet top-down grids are what planners consume.
From the scans, their per-point labels and ``poses.txt``, one grid per scan,
centred on that scan's pose (never on a time grid of its own):

- ``bev_elevation``  ``(3, H, W)`` float16: min / max / mean height of the
                     points in each cell, relative to the ground under the robot
                     (lidar z + ``LIDAR_HEIGHT``); NaN where no point fell.
- ``bev_semantic``   ``(H, W)`` uint8: the majority ontology id of the points in
                     each cell (void only where nothing else fell); 255 unknown.
- ``bev_trav``       ``(H, W)`` uint8: 1 driven (inside the robot's footprint
                     along the trajectory), 0 seen (a point fell) but not driven,
                     255 unknown.

Grid frame (see ``config``): gravity-aligned, origin at the lidar, +x the
robot's heading projected on the horizontal plane, +y left. Row 0 is the far
front, column 0 the far left: the robot drives up the page.

Pipeline, per sequence:

1. ``make_view``      a symlink view under ``out_root/<seq>`` (camera, scans,
                      labels, ``poses/poses.txt``) plus ``poses/timestamps.txt``
                      written from the camera filenames. apairo writes its
                      registry and the derived channels there, never in the
                      shared source tree.
2. ``BevGrids``       an apairo ``FramePreprocessor`` with three
                      ``output_keys``, materialised by ``run_preprocess``.
                      Elevation layers go through apairo_preprocess's
                      ``BEVRasterisation``; the majority class and the footprint
                      are computed here (it has no such reductions).

Accumulation is past-only: the grid of scan ``j`` uses scans ``j, j - s, ...,
j - (n - 1) s`` and nothing later, so the inputs never leak the future. The
trav channel is a label and does use the future path, like
``TraversabilityFromTrajectory``.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from importlib import resources
from pathlib import Path

import numpy as np
from apairo import RawDataset
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample
from apairo_preprocess import BEVRasterisation
from scipy.spatial import cKDTree

from . import config

logger = logging.getLogger(__name__)

BEV_SOURCE_DIRS = [config.CAMERA, "os1_cloud_node_kitti_bin", "os1_cloud_node_semantickitti_label_id"]
INPUT_KEYS = [config.LIDAR, config.LABELS, config.BEV_POSES]
OUTPUT_KEYS = [config.BEV_ELEVATION, config.BEV_SEMANTIC, config.BEV_TRAV]

# Ontology ids -> a compact index for the majority vote; anything unknown counts as void.
CLASS_IDS = np.array(sorted(config.ONTOLOGY), dtype=np.int64)
_COMPACT = np.zeros(1 << 16, dtype=np.int64)
_COMPACT[CLASS_IDS] = np.arange(len(CLASS_IDS))
_VOID = int(_COMPACT[0])


def declaration() -> Path:
    """The packaged ``apairo_bev.yaml`` describing one sequence view."""
    return Path(str(resources.files(__package__) / "apairo_bev.yaml"))


def make_view(root, seq: str, out_root) -> Path:
    """``out_root/<seq>``: symlinks to the source channels + the poses channel and its clock."""
    src = Path(root) / "Rellis-3D" / seq
    view = Path(out_root) / seq
    (view / config.BEV_POSES).mkdir(parents=True, exist_ok=True)
    for name in BEV_SOURCE_DIRS:
        link = view / name
        if not link.is_symlink():
            os.symlink(src / name, link)
    link = view / config.BEV_POSES / "poses.txt"
    if not link.is_symlink():
        os.symlink(src / "poses.txt", link)
    stamps = view / config.BEV_POSES / "timestamps.txt"
    if not stamps.exists():
        camera = RawDataset(view, declare=declaration(), keys=[config.CAMERA])
        np.savetxt(stamps, camera.timestamps[config.CAMERA], fmt="%.3f")
    return view


def open_view(view, keys=INPUT_KEYS) -> RawDataset:
    return RawDataset(view, declare=declaration(), keys=list(keys))


# ------------------------------------------------------------------ geometry

def grid_shape() -> tuple[int, int]:
    r = config.BEV_RESOLUTION
    return (int(round((config.BEV_X_MAX - config.BEV_X_MIN) / r)),
            int(round((config.BEV_Y_MAX - config.BEV_Y_MIN) / r)))


# The grid in BEVRasterisation's convention (row 0 at its ymax, column 0 at its
# xmin) is the BEV plane seen as (u, v) = (-y, x): column 0 = far left, row 0 = far front.
PLANE_BOUNDS = (-config.BEV_Y_MAX, -config.BEV_Y_MIN, config.BEV_X_MIN, config.BEV_X_MAX)


def to_plane(points_bev: np.ndarray) -> np.ndarray:
    return np.column_stack([-points_bev[:, 1], points_bev[:, 0]])


def cell_index(points_bev: np.ndarray) -> np.ndarray:
    """Flat cell index of each point, -1 outside the grid (BEVRasterisation's binning)."""
    h, w = grid_shape()
    r = config.BEV_RESOLUTION
    row = np.floor((config.BEV_X_MAX - points_bev[:, 0]) / r).astype(np.int64)
    col = np.floor((config.BEV_Y_MAX - points_bev[:, 1]) / r).astype(np.int64)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    return np.where(inside, row * w + col, -1)


def cell_centres() -> np.ndarray:
    """``(H * W, 2)`` x, y of every cell centre in the BEV frame, row-major."""
    h, w = grid_shape()
    r = config.BEV_RESOLUTION
    x = config.BEV_X_MAX - (np.arange(h) + 0.5) * r
    y = config.BEV_Y_MAX - (np.arange(w) + 0.5) * r
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def poses_from_rows(rows: np.ndarray) -> np.ndarray:
    """``(N, 12)`` row-major 3x4 rows of poses.txt -> ``(N, 4, 4)`` T_world_lidar."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3] = rows
    return poses


def world_from_bev(T_world_lidar: np.ndarray) -> np.ndarray:
    """``(4, 4)`` T_world_bev: the lidar origin, world z up, x = robot heading (lidar -x) on the ground plane."""
    forward = -T_world_lidar[:3, 0]
    yaw = np.arctan2(forward[1], forward[0])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = T_world_lidar[:3, 3]
    return T


def in_self_box(points_lidar: np.ndarray) -> np.ndarray:
    """Points of the robot itself behind the lidar (``config.SELF_BOX_*``, sensor frame)."""
    x, y, z = points_lidar[:, 0], points_lidar[:, 1], points_lidar[:, 2]
    return ((x >= config.SELF_BOX_X[0]) & (x <= config.SELF_BOX_X[1]) & (y >= config.SELF_BOX_Y[0])
            & (y <= config.SELF_BOX_Y[1]) & (z >= config.SELF_BOX_Z[0]) & (z <= config.SELF_BOX_Z[1]))


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


# -------------------------------------------------------------------- layers

_RASTERS = {
    name: BEVRasterisation(bounds=PLANE_BOUNDS, resolution=config.BEV_RESOLUTION,
                           points_key="uv", values_key="value", reduce=reduce, background=np.nan)
    for name, reduce in (("max", "max"), ("mean", "mean"))
}


def elevation(points_bev: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """``(3, H, W)`` float16 min / max / mean height per cell, NaN where empty."""
    uv = to_plane(points_bev)
    top = _RASTERS["max"](Sample(data={"uv": uv, "value": heights}))
    low = -_RASTERS["max"](Sample(data={"uv": uv, "value": -heights}))
    mean = _RASTERS["mean"](Sample(data={"uv": uv, "value": heights}))
    return np.stack([low, top, mean]).astype(np.float16)


def semantic(cells: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``(H, W)`` uint8 majority ontology id per cell; void only if nothing else fell; 255 empty.

    Ties go to the lower ontology id."""
    h, w = grid_shape()
    k = len(CLASS_IDS)
    inside = cells >= 0
    counts = np.bincount(cells[inside] * k + _COMPACT[labels[inside]],
                         minlength=h * w * k).reshape(h * w, k)
    seen = counts.sum(axis=1) > 0
    counts[:, _VOID] = 0                     # void wins only an all-void cell
    best = np.where(counts.sum(axis=1) > 0, CLASS_IDS[counts.argmax(axis=1)], 0)
    out = np.where(seen, best, config.NO_DATA).astype(np.uint8)
    return out.reshape(h, w)


def densify(path_xy: np.ndarray, step: float = 0.1) -> np.ndarray:
    """Points every ``step`` metres along a polyline."""
    if len(path_xy) < 2:
        return path_xy
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.arange(0.0, s[-1] + step, step)
    return np.column_stack([np.interp(t, s, path_xy[:, 0]), np.interp(t, s, path_xy[:, 1])])


def footprint(path_bev_xy: np.ndarray, centres: np.ndarray, half_width: float) -> np.ndarray:
    """``(H, W)`` bool: cells whose centre is within ``half_width`` of the path."""
    h, w = grid_shape()
    if len(path_bev_xy) == 0:
        return np.zeros((h, w), dtype=bool)
    d, _ = cKDTree(densify(path_bev_xy)).query(centres, distance_upper_bound=half_width + 1.0)
    return (d <= half_width).reshape(h, w)


# -------------------------------------------------------------- preprocessor

class BevGrids(FramePreprocessor):
    """One robot-centred elevation / semantic / trav grid per scan.

    Frame-wise for the runner, but a grid needs the trajectory (the footprint,
    the past scans' placement) and the past scans themselves: those are read
    from the view's loaders by scan index, located by the sample's stamp, so
    the output never depends on the order frames are asked in. Placed past
    scans are cached, so an in-order run reads each scan once.
    """

    output_keys = OUTPUT_KEYS
    output_loader = "npys"
    input_keys = INPUT_KEYS
    timestamps_from = config.LIDAR
    sources = INPUT_KEYS

    def __init__(self, dataset: RawDataset, *, accumulate: int = config.BEV_ACCUMULATE,
                 stride: int = config.BEV_STRIDE) -> None:
        # Public scalars enter the recipe hash (run_preprocess(reuse=True)).
        self.accumulate = int(accumulate)
        self.stride = int(stride)
        self.resolution = config.BEV_RESOLUTION
        self.bounds = str((config.BEV_X_MIN, config.BEV_X_MAX, config.BEV_Y_MIN, config.BEV_Y_MAX))
        self.lidar_height = config.LIDAR_HEIGHT
        self.min_range = config.MIN_RANGE
        self.self_box = str((config.SELF_BOX_X, config.SELF_BOX_Y, config.SELF_BOX_Z))
        self.half_width = config.ROBOT_HALF_WIDTH
        self.path_back = config.BEV_PATH_BACK
        self.path_forward = config.FORWARD_DISTANCE
        self.min_future_path = config.MIN_FUTURE_PATH

        self._lidar = dataset.loaders[config.LIDAR]
        self._labels = dataset.loaders[config.LABELS]
        poses_loader = dataset.loaders[config.BEV_POSES]
        self._poses = poses_from_rows(np.stack([poses_loader[i] for i in range(len(poses_loader))]))
        self._stamps = np.asarray(dataset.timestamps[config.LIDAR], dtype=np.float64)
        if len(self._poses) != len(self._stamps) or len(self._lidar) != len(self._stamps):
            raise ValueError(f"{len(self._poses)} poses, {len(self._lidar)} scans, {len(self._stamps)} stamps")
        steps = np.linalg.norm(np.diff(self._poses[:, :3, 3], axis=0), axis=1)
        self._path = np.concatenate([[0.0], np.cumsum(steps)])
        self._centres = cell_centres()
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def index_of(self, stamp: float) -> int:
        j = int(np.argmin(np.abs(self._stamps - stamp)))
        if abs(self._stamps[j] - stamp) > 1e-3:
            raise ValueError(f"no scan at t={stamp:.3f}")
        return j

    def _world_scan(self, k: int, points=None, labels=None) -> tuple[np.ndarray, np.ndarray]:
        """Scan ``k`` in the world frame (self-returns dropped) with its labels, cached."""
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        if points is None:
            points, labels = self._lidar[k], self._labels[k]
        points = np.asarray(points)[:, :3].astype(np.float64)
        labels = np.asarray(labels)
        keep = (np.linalg.norm(points, axis=1) >= self.min_range) & ~in_self_box(points)
        out = (transform(self._poses[k], points[keep]).astype(np.float32), labels[keep])
        self._cache[k] = out
        while len(self._cache) > self.accumulate * self.stride + 1:
            self._cache.popitem(last=False)
        return out

    def scans_used(self, j: int) -> list[int]:
        """The scan indices accumulated into the grid of scan ``j``: ``j`` and earlier only."""
        return [k for k in range(j, j - self.accumulate * self.stride, -self.stride) if k >= 0]

    def __call__(self, sample: Sample) -> dict:
        j = self.index_of(sample.timestamp)
        pose = poses_from_rows(sample.data[config.BEV_POSES])[0]
        if not np.allclose(pose, self._poses[j]):
            raise ValueError(f"scan {j}: the sample's pose is not poses.txt row {j}")
        bev_from_world = np.linalg.inv(world_from_bev(pose))

        points, labels = [], []
        for k in self.scans_used(j):
            if k == j:
                p, lab = self._world_scan(k, sample.data[config.LIDAR], sample.data[config.LABELS])
            else:
                p, lab = self._world_scan(k)
            points.append(p)
            labels.append(lab)
        points = transform(bev_from_world, np.concatenate(points).astype(np.float64))
        labels = np.concatenate(labels)
        cells = cell_index(points)
        inside = cells >= 0
        points, labels, cells = points[inside], labels[inside], cells[inside]

        elev = elevation(points, points[:, 2] + self.lidar_height)
        sem = semantic(cells, labels)

        h, w = grid_shape()
        if self._path[-1] - self._path[j] >= self.min_future_path:
            window = (self._path >= self._path[j] - self.path_back) & (self._path <= self._path[j] + self.path_forward)
            path_bev = transform(bev_from_world, self._poses[window, :3, 3])[:, :2]
            driven = footprint(path_bev, self._centres, self.half_width)
            trav = np.full((h, w), config.NO_DATA, dtype=np.uint8)
            trav[sem != config.NO_DATA] = 0
            trav[driven] = 1
        else:
            trav = np.full((h, w), config.NO_DATA, dtype=np.uint8)
        return {config.BEV_ELEVATION: elev, config.BEV_SEMANTIC: sem, config.BEV_TRAV: trav}


def process_sequence(root, seq: str, out_root, *, overwrite: bool = False,
                     accumulate: int = config.BEV_ACCUMULATE, stride: int = config.BEV_STRIDE) -> dict:
    """Materialise the three BEV channels of one sequence in its view."""
    t0 = time.time()
    view = make_view(root, seq, out_root)
    ds = open_view(view)
    grids = BevGrids(ds, accumulate=accumulate, stride=stride)
    RawDataset.run_preprocess(grids, view, declare=declaration(), overwrite=overwrite, reuse=not overwrite)
    report = {"sequence": seq, "scans": len(ds.timestamps[config.LIDAR]),
              "accumulate": accumulate, "stride": stride, "seconds": round(time.time() - t0)}
    logger.info("%s", report)
    return report
