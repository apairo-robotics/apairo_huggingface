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
five raw TartanDrive channels, the GICP poses and their validity mask on the
lidar clock), the heading (vehicle +x, where RELLIS's is the Ouster's -x),
the up axis (gravity from the INS attitude at the scan, where RELLIS uses its
poses' world z: the GICP world is the first scan's vehicle frame and its z
drifts by up to 26 deg), the lidar height (1.77 m), the vehicle width, the
self-return filter (a 2 m radius covers the roof) and the semantic layer
swapped for the geometric one. Scans whose GICP pose is flagged invalid, or
with no INS attitude within ``ATTITUDE_TOLERANCE``, get all-unknown grids and
enter no other grid nor the driven corridor.

Pipeline, per recording:

1. ``make_view``     a symlink view under ``out_root/<seq>`` of the five
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
from importlib import resources
from pathlib import Path

import numpy as np
from apairo import RawDataset
from scipy.spatial.transform import Rotation
from apairo.core.sample import Sample

from ..bev_grid import BevGridBase, cell_index, elevation, grid_shape, level_frame, trav_grid, transform
from . import config

logger = logging.getLogger(__name__)

INPUT_KEYS = [config.LIDAR, config.POSES]                          # zipped per scan by the runner
VIEW_KEYS = INPUT_KEYS + [config.POSE_VALID, config.ATTITUDE]      # all that BevGrids reads
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


def open_view(view, keys=VIEW_KEYS) -> RawDataset:
    return RawDataset(view, declare=declaration(), keys=list(keys))


def scan_poses(dataset: RawDataset) -> np.ndarray:
    """``(N, 4, 4)`` GICP T_world_vehicle, one per scan."""
    loader = dataset.loaders[config.POSES]
    return np.stack([np.asarray(loader[i], dtype=np.float64) for i in range(len(loader))])


def scan_up(dataset: RawDataset) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 3)`` gravity up in the vehicle frame of every scan, from the INS
    attitude row nearest the scan, and ``(N,)`` bool: whether that row is
    within ``config.ATTITUDE_TOLERANCE``. Rows hold ``qx qy qz qw`` of the
    vehicle in a local level frame, so up in the vehicle frame is R^T e_z."""
    loader = dataset.loaders[config.ATTITUDE]
    rows = np.stack([np.asarray(loader[i], dtype=np.float64) for i in range(len(loader))])
    times = np.asarray(dataset.timestamps[config.ATTITUDE], dtype=np.float64)
    stamps = np.asarray(dataset.timestamps[config.POSES], dtype=np.float64)   # the scan clock
    i = np.clip(np.searchsorted(times, stamps), 1, len(times) - 1)
    i = np.where(np.abs(times[i - 1] - stamps) <= np.abs(times[i] - stamps), i - 1, i)
    up = Rotation.from_quat(rows[i, 3:7]).inv().apply([0.0, 0.0, 1.0])
    return up, np.abs(times[i] - stamps) <= config.ATTITUDE_TOLERANCE


def valid_scans(dataset: RawDataset) -> np.ndarray:
    """``(N,)`` bool: the scans with a valid GICP pose (``valid_mask.npy``) and an INS attitude."""
    loader = dataset.loaders[config.POSE_VALID]
    valid = np.array([bool(loader[i]) for i in range(len(loader))])
    return valid & scan_up(dataset)[1]


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

class BevGrids(BevGridBase):
    """One robot-centred elevation / geometry / trav grid per scan (shared
    machinery: :class:`~apairo_huggingface.bev_grid.BevGridBase`).

    The grid of scan ``j`` is levelled by the INS gravity at ``j``; past scans
    and the driven path are placed relative to ``j`` by the GICP poses (exact
    over the few metres and seconds a grid spans, where the GICP world's own z
    is not). A scan without a valid pose or attitude gets all-unknown grids
    and never enters another scan's grid.
    """

    output_keys = OUTPUT_KEYS
    input_keys = INPUT_KEYS
    timestamps_from = config.LIDAR
    sources = VIEW_KEYS

    def __init__(self, dataset: RawDataset, *, accumulate: int = config.BEV_ACCUMULATE,
                 stride: int = config.BEV_STRIDE) -> None:
        super().__init__(scan_poses(dataset), np.asarray(dataset.timestamps[config.LIDAR], dtype=np.float64),
                         valid=valid_scans(dataset), accumulate=accumulate, stride=stride,
                         lidar_height=config.LIDAR_HEIGHT, min_range=config.MIN_RANGE,
                         half_width=config.ROBOT_HALF_WIDTH, path_back=config.BEV_PATH_BACK,
                         path_forward=config.FORWARD_DISTANCE, min_future_path=config.MIN_FUTURE_PATH)
        self.up = "ins_gravity"
        self.attitude_tolerance = config.ATTITUDE_TOLERANCE
        self.min_cell_points = config.MIN_CELL_POINTS
        self._up = scan_up(dataset)[0]
        self._lidar = dataset.loaders[config.LIDAR]
        if len(self._lidar) != len(self._stamps):
            raise ValueError(f"{len(self._lidar)} scans for {len(self._stamps)} stamps")

    def index_of(self, stamp: float, pose: np.ndarray) -> int:
        """The scan at ``stamp``. TartanDrive repeats some scans under the same
        stamp (identical points, re-registered pose): the pose tells them apart."""
        candidates = np.flatnonzero(np.abs(self._stamps - stamp) <= 1e-3)
        for j in candidates:
            if np.allclose(pose, self._poses[j]):
                return int(j)
        raise ValueError(f"no scan with this pose at t={stamp:.3f} ({len(candidates)} scans at that stamp)")

    def world_from_bev(self, j: int) -> np.ndarray:
        """``(4, 4)`` T_world_bev of scan ``j``: the lidar origin, z = INS gravity up, x = the vehicle's +x levelled."""
        R = self._poses[j][:3, :3]
        return level_frame(self._poses[j], R[:, 0], R @ self._up[j])

    def place(self, k: int, points=None) -> np.ndarray:
        """Scan ``k`` in the world frame, self-returns and non-finite points dropped."""
        if points is None:
            points = self._lidar[k]
        points = np.asarray(points)[:, :3].astype(np.float64)
        keep = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) >= self.min_range)
        return transform(self._poses[k], points[keep]).astype(np.float32)

    def __call__(self, sample: Sample) -> dict:
        j = self.index_of(sample.timestamp, np.asarray(sample.data[config.POSES], dtype=np.float64))
        h, w = grid_shape()
        if not self._valid[j]:
            unknown = np.full((h, w), np.nan, dtype=np.float16)
            return {config.BEV_ELEVATION: np.stack([unknown] * 3), config.BEV_GEOMETRY: np.stack([unknown] * 2),
                    config.BEV_TRAV: trav_grid(np.zeros((h, w), dtype=bool), None)}
        bev_from_world = np.linalg.inv(self.world_from_bev(j))

        points = [self.world_scan(k, sample.data[config.LIDAR] if k == j else None) for k in self.scans_used(j)]
        points = transform(bev_from_world, np.concatenate(points).astype(np.float64))
        cells = cell_index(points)
        inside = cells >= 0
        points, cells = points[inside], cells[inside]
        heights = points[:, 2] + self.lidar_height

        elev = elevation(points, heights)
        geom = geometry(cells, heights, elev[2])
        trav = trav_grid(np.isfinite(elev[2]), self.driven(j, bev_from_world))
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
