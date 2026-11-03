"""Robot-centred BEV grids for every TartanDrive 2.0 scan: the RELLIS BEV recipe, ported.

Same grid (:mod:`apairo_huggingface.bev_grid`), same accumulation, same
driven corridor as :mod:`apairo_huggingface.rellis.bev`. One grid per scan,
centred on that scan's pose:

- ``bev_elevation``  ``(3, H, W)`` float16: min / max / mean height of the
                     points in each cell above the ground under the robot
                     (lidar z + ``LIDAR_HEIGHT``); NaN where no point fell.
- ``bev_geometry``   ``(2, H, W)`` float16, replacing RELLIS's
                     ``bev_semantic`` (TartanDrive has no per-point labels):
                     ``roughness``, the standard deviation of the point
                     heights in the cell (NaN under ``MIN_CELL_POINTS``), and
                     ``step``, the largest jump of the mean height to a known
                     8-neighbour (NaN if the cell or all its neighbours are
                     unknown). Both in metres, both from the inputs only.
- ``bev_trav``       ``(H, W)`` uint8: 1 driven (inside the vehicle's
                     footprint along the trajectory), 0 seen (a point fell)
                     but not driven, 255 unknown.

What the port changes, and nothing else: the declaration (``apairo_bev.yaml``:
four raw TartanDrive channels, the GICP poses on the lidar clock), the heading
(vehicle +x, where RELLIS's is the Ouster's -x), the lidar height (1.77 m),
the vehicle width, the self-return filter (a 2 m radius covers the roof) and
the semantic layer swapped for the geometric one.

Pipeline, per recording:

1. ``make_view``     a symlink view under ``out_root/<seq>`` of the four
                     source channels; apairo writes its registry and the
                     derived channels there, never in the shared source tree.
2. ``BevGrids``      an apairo ``FramePreprocessor`` with three
                     ``output_keys``, materialised by ``run_preprocess`` on the
                     asynchronous recording (scan and pose share a clock, so
                     the runner zips them into one sample).

Accumulation is past-only: the grid of scan ``j`` uses scans ``j, j - s, ...,
j - (n - 1) s`` and nothing later. The trav channel is a label and does use
the future path.
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

from ..bev_grid import (cell_centres, cell_index, driven_window, elevation, footprint, grid_shape, scans_used,
                        trav_grid, transform, world_from_heading)
from . import config

logger = logging.getLogger(__name__)

INPUT_KEYS = [config.LIDAR, config.POSES]
OUTPUT_KEYS = [config.BEV_ELEVATION, config.BEV_GEOMETRY, config.BEV_TRAV]
GEOMETRY_NAMES = ["roughness", "step"]


def declaration() -> Path:
    """The packaged ``apairo_bev.yaml`` describing one recording view."""
    return Path(str(resources.files(__package__) / "apairo_bev.yaml"))


def make_view(root, seq: str, out_root) -> Path:
    """``out_root/<seq>``: symlinks to the recording's source channels."""
    src = Path(root) / seq
    view = Path(out_root) / seq
    view.mkdir(parents=True, exist_ok=True)
    for name in config.SOURCE_DIRS:
        if not (src / name).is_dir():
            raise FileNotFoundError(f"{src / name} missing")
        link = view / name
        if not link.is_symlink():
            os.symlink(src / name, link)
    return view


def open_view(view, keys=INPUT_KEYS) -> RawDataset:
    return RawDataset(view, declare=declaration(), keys=list(keys))


def world_from_bev(T_world_vehicle: np.ndarray) -> np.ndarray:
    """``(4, 4)`` T_world_bev: the lidar origin, world z up, x = the vehicle's +x on the ground plane."""
    return world_from_heading(T_world_vehicle, T_world_vehicle[:3, 0])


# -------------------------------------------------------------------- layers

def geometry(cells: np.ndarray, heights: np.ndarray, mean_height: np.ndarray) -> np.ndarray:
    """``(2, H, W)`` float16 roughness / step per cell, NaN unknown.

    ``cells`` are the flat cell indices of in-grid points, ``mean_height`` the
    ``(H, W)`` mean elevation layer (NaN unknown)."""
    h, w = grid_shape()
    count = np.bincount(cells, minlength=h * w)
    total = np.bincount(cells, weights=heights, minlength=h * w)
    square = np.bincount(cells, weights=heights * heights, minlength=h * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        var = np.maximum(square / count - mean * mean, 0.0)
    rough = np.where(count >= config.MIN_CELL_POINTS, np.sqrt(var), np.nan).reshape(h, w)

    m = np.pad(mean_height.astype(np.float64), 1, constant_values=np.nan)
    jumps = [np.abs(m[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc] - m[1:h + 1, 1:w + 1])
             for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    step = np.fmax.reduce(np.stack(jumps), axis=0)   # fmax skips NaN: NaN only if every jump is unknown
    return np.stack([rough, step]).astype(np.float16)


# -------------------------------------------------------------- preprocessor

class BevGrids(FramePreprocessor):
    """One robot-centred elevation / geometry / trav grid per scan.

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
        self.lidar_height = config.LIDAR_HEIGHT
        self.min_range = config.MIN_RANGE
        self.half_width = config.ROBOT_HALF_WIDTH
        self.path_back = config.BEV_PATH_BACK
        self.path_forward = config.FORWARD_DISTANCE
        self.min_future_path = config.MIN_FUTURE_PATH
        self.min_cell_points = config.MIN_CELL_POINTS

        self._lidar = dataset.loaders[config.LIDAR]
        poses_loader = dataset.loaders[config.POSES]
        self._poses = np.stack([np.asarray(poses_loader[i], dtype=np.float64) for i in range(len(poses_loader))])
        self._stamps = np.asarray(dataset.timestamps[config.LIDAR], dtype=np.float64)
        if len(self._poses) != len(self._stamps) or len(self._lidar) != len(self._stamps):
            raise ValueError(f"{len(self._poses)} poses, {len(self._lidar)} scans, {len(self._stamps)} stamps")
        steps = np.linalg.norm(np.diff(self._poses[:, :3, 3], axis=0), axis=1)
        self._path = np.concatenate([[0.0], np.cumsum(steps)])
        self._centres = cell_centres()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def index_of(self, stamp: float, pose: np.ndarray) -> int:
        """The scan at ``stamp``. TartanDrive repeats some scans under the same
        stamp (identical points, re-registered pose): the pose tells them apart."""
        candidates = np.flatnonzero(np.abs(self._stamps - stamp) <= 1e-3)
        for j in candidates:
            if np.allclose(pose, self._poses[j]):
                return int(j)
        raise ValueError(f"no scan with this pose at t={stamp:.3f} ({len(candidates)} scans at that stamp)")

    def _world_scan(self, k: int, points=None) -> np.ndarray:
        """Scan ``k`` in the world frame, self-returns and non-finite points dropped, cached."""
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        if points is None:
            points = self._lidar[k]
        points = np.asarray(points)[:, :3].astype(np.float64)
        keep = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) >= self.min_range)
        out = transform(self._poses[k], points[keep]).astype(np.float32)
        self._cache[k] = out
        while len(self._cache) > self.accumulate * self.stride + 1:
            self._cache.popitem(last=False)
        return out

    def scans_used(self, j: int) -> list[int]:
        """The scan indices accumulated into the grid of scan ``j``: ``j`` and earlier only."""
        return scans_used(j, self.accumulate, self.stride)

    def __call__(self, sample: Sample) -> dict:
        pose = np.asarray(sample.data[config.POSES], dtype=np.float64)
        j = self.index_of(sample.timestamp, pose)
        bev_from_world = np.linalg.inv(world_from_bev(pose))

        points = [self._world_scan(k, sample.data[config.LIDAR] if k == j else None) for k in self.scans_used(j)]
        points = transform(bev_from_world, np.concatenate(points).astype(np.float64))
        cells = cell_index(points)
        inside = cells >= 0
        points, cells = points[inside], cells[inside]
        heights = points[:, 2] + self.lidar_height

        elev = elevation(points, heights)
        geom = geometry(cells, heights, elev[2])

        driven = None
        if self._path[-1] - self._path[j] >= self.min_future_path:
            window = driven_window(self._path, j, self.path_back, self.path_forward)
            path_bev = transform(bev_from_world, self._poses[window, :3, 3])[:, :2]
            driven = footprint(path_bev, self._centres, self.half_width)
        trav = trav_grid(np.isfinite(elev[2]), driven)
        return {config.BEV_ELEVATION: elev, config.BEV_GEOMETRY: geom, config.BEV_TRAV: trav}


def process_sequence(root, seq: str, out_root, *, overwrite: bool = False,
                     accumulate: int = config.BEV_ACCUMULATE, stride: int = config.BEV_STRIDE) -> dict:
    """Materialise the three BEV channels of one recording in its view."""
    t0 = time.time()
    view = make_view(root, seq, out_root)
    ds = open_view(view)
    grids = BevGrids(ds, accumulate=accumulate, stride=stride)
    RawDataset.run_preprocess(grids, view, declare=declaration(), overwrite=overwrite, reuse=not overwrite)
    report = {"sequence": seq, "scans": len(ds.timestamps[config.LIDAR]),
              "accumulate": accumulate, "stride": stride, "seconds": round(time.time() - t0)}
    logger.info("%s", report)
    return report
