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
from importlib import resources
from pathlib import Path

import numpy as np
from apairo import RawDataset
from apairo.core.sample import Sample

from ..bev_grid import (PLANE_BOUNDS, BevGridBase, cell_centres, cell_index, densify, driven_window,  # noqa: F401
                        elevation, footprint, grid_shape, scans_used, to_plane, trav_grid, transform,
                        world_from_heading)
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
# The grid itself (shape, binning, frame, elevation, footprint) is the shared
# one; RELLIS adds where its heading points and its own self-returns.

def poses_from_rows(rows: np.ndarray) -> np.ndarray:
    """``(N, 12)`` row-major 3x4 rows of poses.txt -> ``(N, 4, 4)`` T_world_lidar."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3] = rows
    return poses


def world_from_bev(T_world_lidar: np.ndarray) -> np.ndarray:
    """``(4, 4)`` T_world_bev: the lidar origin, world z up, x = robot heading (lidar -x) on the ground plane."""
    return world_from_heading(T_world_lidar, -T_world_lidar[:3, 0])


def in_self_box(points_lidar: np.ndarray) -> np.ndarray:
    """Points of the robot itself behind the lidar (``config.SELF_BOX_*``, sensor frame)."""
    x, y, z = points_lidar[:, 0], points_lidar[:, 1], points_lidar[:, 2]
    return ((x >= config.SELF_BOX_X[0]) & (x <= config.SELF_BOX_X[1]) & (y >= config.SELF_BOX_Y[0])
            & (y <= config.SELF_BOX_Y[1]) & (z >= config.SELF_BOX_Z[0]) & (z <= config.SELF_BOX_Z[1]))


# -------------------------------------------------------------------- layers

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


# -------------------------------------------------------------- preprocessor

class BevGrids(BevGridBase):
    """One robot-centred elevation / semantic / trav grid per scan (shared
    machinery: :class:`~apairo_huggingface.bev_grid.BevGridBase`)."""

    output_keys = OUTPUT_KEYS
    input_keys = INPUT_KEYS
    timestamps_from = config.LIDAR
    sources = INPUT_KEYS

    def __init__(self, dataset: RawDataset, *, accumulate: int = config.BEV_ACCUMULATE,
                 stride: int = config.BEV_STRIDE) -> None:
        poses_loader = dataset.loaders[config.BEV_POSES]
        poses = poses_from_rows(np.stack([poses_loader[i] for i in range(len(poses_loader))]))
        super().__init__(poses, np.asarray(dataset.timestamps[config.LIDAR], dtype=np.float64),
                         accumulate=accumulate, stride=stride, lidar_height=config.LIDAR_HEIGHT,
                         min_range=config.MIN_RANGE, half_width=config.ROBOT_HALF_WIDTH,
                         path_back=config.BEV_PATH_BACK, path_forward=config.FORWARD_DISTANCE,
                         min_future_path=config.MIN_FUTURE_PATH)
        self.self_box = str((config.SELF_BOX_X, config.SELF_BOX_Y, config.SELF_BOX_Z))
        self._lidar = dataset.loaders[config.LIDAR]
        self._labels = dataset.loaders[config.LABELS]
        if len(self._lidar) != len(self._stamps):
            raise ValueError(f"{len(self._lidar)} scans for {len(self._stamps)} stamps")

    def index_of(self, stamp: float) -> int:
        j = int(np.argmin(np.abs(self._stamps - stamp)))
        if abs(self._stamps[j] - stamp) > 1e-3:
            raise ValueError(f"no scan at t={stamp:.3f}")
        return j

    def place(self, k: int, data=None) -> tuple[np.ndarray, np.ndarray]:
        """Scan ``k`` in the world frame (self-returns dropped) with its labels; ``data`` = (points, labels)."""
        points, labels = data if data is not None else (self._lidar[k], self._labels[k])
        points = np.asarray(points)[:, :3].astype(np.float64)
        labels = np.asarray(labels)
        keep = (np.linalg.norm(points, axis=1) >= self.min_range) & ~in_self_box(points)
        return transform(self._poses[k], points[keep]).astype(np.float32), labels[keep]

    def __call__(self, sample: Sample) -> dict:
        j = self.index_of(sample.timestamp)
        pose = poses_from_rows(sample.data[config.BEV_POSES])[0]
        if not np.allclose(pose, self._poses[j]):
            raise ValueError(f"scan {j}: the sample's pose is not poses.txt row {j}")
        bev_from_world = np.linalg.inv(world_from_bev(pose))

        points, labels = [], []
        for k in self.scans_used(j):
            data = (sample.data[config.LIDAR], sample.data[config.LABELS]) if k == j else None
            p, lab = self.world_scan(k, data)
            points.append(p)
            labels.append(lab)
        points = transform(bev_from_world, np.concatenate(points).astype(np.float64))
        labels = np.concatenate(labels)
        cells = cell_index(points)
        inside = cells >= 0
        points, labels, cells = points[inside], labels[inside], cells[inside]

        elev = elevation(points, points[:, 2] + self.lidar_height)
        sem = semantic(cells, labels)
        trav = trav_grid(sem != config.NO_DATA, self.driven(j, bev_from_world))
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
