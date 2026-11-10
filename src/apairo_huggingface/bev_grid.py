"""The robot-centred BEV grid shared by every per-dataset BEV recipe.

One grid, one set of conventions, used unchanged by :mod:`.rellis.bev` and
:mod:`.tartan_bev.bev`, so a model trained on one dataset's grids reads the
other's:

- 0.2 m cells, 40 x 40 m, forward-biased: 30 m ahead, 10 m behind, 20 m
  either side, i.e. 200 x 200 cells;
- level frame: origin at the lidar, +z up, +x the robot's heading projected
  on the plane normal to +z, +y left. Which "up" is the dataset's call: world
  z of the poses (:func:`world_from_heading`, RELLIS) or gravity measured at
  the scan (:func:`level_frame`, TartanDrive's INS attitude);
- row 0 is the far front, column 0 the far left: the robot drives up the page.

:class:`BevGridBase` holds what every per-dataset ``BevGrids`` preprocessor
shares: the recipe scalars (grid included), the path length along the poses,
the past-only scan selection, the cache of placed scans and the driven
corridor. What differs per dataset (where the heading points in the sensor
frame, which way is up, the lidar height, the self-returns, the per-cell
layers beyond elevation) stays in the dataset's own module.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from apairo.core.preprocessor import FramePreprocessor
from apairo.core.sample import Sample
from apairo_preprocess import BEVRasterisation
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

RESOLUTION = 0.2              # m per cell
X_MIN, X_MAX = -10.0, 30.0    # m, behind / ahead of the lidar
Y_MIN, Y_MAX = -20.0, 20.0    # m, right / left
NO_DATA = 255                 # unknown, in every uint8 grid


# ------------------------------------------------------------------ geometry

def grid_shape() -> tuple[int, int]:
    return (int(round((X_MAX - X_MIN) / RESOLUTION)), int(round((Y_MAX - Y_MIN) / RESOLUTION)))


# The grid in BEVRasterisation's convention (row 0 at its ymax, column 0 at its
# xmin) is the BEV plane seen as (u, v) = (-y, x): column 0 = far left, row 0 = far front.
PLANE_BOUNDS = (-Y_MAX, -Y_MIN, X_MIN, X_MAX)


def to_plane(points_bev: np.ndarray) -> np.ndarray:
    return np.column_stack([-points_bev[:, 1], points_bev[:, 0]])


def cell_index(points_bev: np.ndarray) -> np.ndarray:
    """Flat cell index of each point, -1 outside the grid (BEVRasterisation's binning)."""
    h, w = grid_shape()
    row = np.floor((X_MAX - points_bev[:, 0]) / RESOLUTION).astype(np.int64)
    col = np.floor((Y_MAX - points_bev[:, 1]) / RESOLUTION).astype(np.int64)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    return np.where(inside, row * w + col, -1)


def cell_centres() -> np.ndarray:
    """``(H * W, 2)`` x, y of every cell centre in the BEV frame, row-major."""
    h, w = grid_shape()
    x = X_MAX - (np.arange(h) + 0.5) * RESOLUTION
    y = Y_MAX - (np.arange(w) + 0.5) * RESOLUTION
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def world_from_heading(T_world_lidar: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """``(4, 4)`` T_world_bev: the lidar origin, world z up, x = ``forward`` (a world vector) on the ground plane."""
    yaw = np.arctan2(forward[1], forward[0])
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = T_world_lidar[:3, 3]
    return T


def level_frame(T_world_lidar: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``(4, 4)`` T_world_bev: the lidar origin, z = ``up``, x = ``forward`` projected on
    the plane normal to ``up`` (both world vectors), y = z x x."""
    z = up / np.linalg.norm(up)
    x = forward - np.dot(forward, z) * z
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = T_world_lidar[:3, 3]
    return T


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def scans_used(j: int, accumulate: int, stride: int) -> list[int]:
    """The scans accumulated into the grid of scan ``j``: ``j, j - stride, ...``, never later than ``j``."""
    return [k for k in range(j, j - accumulate * stride, -stride) if k >= 0]


# -------------------------------------------------------------------- layers

_RASTERS = {
    reduce: BEVRasterisation(bounds=PLANE_BOUNDS, resolution=RESOLUTION,
                             points_key="uv", values_key="value", reduce=reduce, background=np.nan)
    for reduce in ("max", "mean")
}


def elevation(points_bev: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """``(3, H, W)`` float16 min / max / mean height per cell, NaN where empty."""
    uv = to_plane(points_bev)
    top = _RASTERS["max"](Sample(data={"uv": uv, "value": heights}))
    low = -_RASTERS["max"](Sample(data={"uv": uv, "value": -heights}))
    mean = _RASTERS["mean"](Sample(data={"uv": uv, "value": heights}))
    return np.stack([low, top, mean]).astype(np.float16)


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


def driven_window(path: np.ndarray, j: int, back: float, forward: float) -> np.ndarray:
    """Bool mask of the poses within ``back`` metres of path behind scan ``j`` and ``forward`` ahead."""
    return (path >= path[j] - back) & (path <= path[j] + forward)


def trav_grid(seen: np.ndarray, driven: np.ndarray | None) -> np.ndarray:
    """``(H, W)`` uint8: 1 driven, 0 seen and not driven, 255 unknown; all 255 if ``driven`` is None."""
    trav = np.full(seen.shape, NO_DATA, dtype=np.uint8)
    if driven is None:
        return trav
    trav[seen] = 0
    trav[driven] = 1
    return trav


# -------------------------------------------------------------- preprocessor

def path_length(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Distance travelled at each pose, along the valid poses only (an invalid
    pose holds the length of the last valid one before it)."""
    idx = np.flatnonzero(valid)
    path = np.zeros(len(poses))
    path[idx] = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poses[idx, :3, 3], axis=0), axis=1))])
    return np.maximum.accumulate(path)


class BevGridBase(FramePreprocessor):
    """What every per-dataset ``BevGrids`` shares; a subclass adds ``place``
    (one scan in the world frame) and ``__call__`` (its layers).

    Frame-wise for the runner, but a grid needs the trajectory (the footprint,
    the past scans' placement) and the past scans themselves: subclasses read
    those from the view's loaders by scan index, so the output never depends
    on the order frames are asked in. Placed scans are cached, so an in-order
    run reads each scan once. Scans whose pose is not ``valid`` never enter a
    grid, neither as a past scan nor in the driven corridor.
    """

    output_loader = "npys"

    def __init__(self, poses: np.ndarray, stamps: np.ndarray, *, valid: np.ndarray | None = None,
                 accumulate: int, stride: int, lidar_height: float, min_range: float, half_width: float,
                 path_back: float, path_forward: float, min_future_path: float) -> None:
        # Public scalars enter the recipe hash (run_preprocess(reuse=True)),
        # the grid's resolution and bounds included: change one and the next
        # run recomputes instead of reusing the old grids.
        self.accumulate = int(accumulate)
        self.stride = int(stride)
        self.resolution = RESOLUTION
        self.bounds = str((X_MIN, X_MAX, Y_MIN, Y_MAX))
        self.lidar_height = lidar_height
        self.min_range = min_range
        self.half_width = half_width
        self.path_back = path_back
        self.path_forward = path_forward
        self.min_future_path = min_future_path

        if len(poses) != len(stamps):
            raise ValueError(f"{len(poses)} poses for {len(stamps)} scan stamps")
        self._poses = poses
        self._stamps = stamps
        self._valid = np.ones(len(poses), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self._path = path_length(poses, self._valid)
        self._centres = cell_centres()
        self._cache: OrderedDict = OrderedDict()

    def place(self, k: int, data=None):
        """Scan ``k`` in the world frame (``data``: its already loaded inputs, else read)."""
        raise NotImplementedError

    def world_scan(self, k: int, data=None):
        """``place(k, data)``, cached for the next grids."""
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        out = self.place(k, data)
        self._cache[k] = out
        while len(self._cache) > self.accumulate * self.stride + 1:
            self._cache.popitem(last=False)
        return out

    def scans_used(self, j: int) -> list[int]:
        """The valid scans accumulated into the grid of scan ``j``: ``j`` and earlier only."""
        return [k for k in scans_used(j, self.accumulate, self.stride) if self._valid[k]]

    def driven(self, j: int, bev_from_world: np.ndarray) -> np.ndarray | None:
        """``(H, W)`` bool footprint of the valid path around scan ``j``; None with too little path ahead."""
        if self._path[-1] - self._path[j] < self.min_future_path:
            return None
        window = driven_window(self._path, j, self.path_back, self.path_forward) & self._valid
        path_bev = transform(bev_from_world, self._poses[window, :3, 3])[:, :2]
        return footprint(path_bev, self._centres, self.half_width)


# ----------------------------------------------------------------- rendering

HEIGHT_RANGE = (-1.0, 3.0)   # m above the ground under the robot
_RAMP = np.array([[48, 18, 59], [50, 136, 250], [27, 229, 181], [190, 245, 52],
                  [250, 140, 30], [122, 4, 3]], dtype=np.float64)   # turbo-like stops
TRAV_COLORS = np.zeros((256, 3), dtype=np.uint8)
TRAV_COLORS[0] = (220, 50, 50)
TRAV_COLORS[1] = (50, 220, 50)


def height_rgb(height: np.ndarray, lo: float = HEIGHT_RANGE[0], hi: float = HEIGHT_RANGE[1]) -> np.ndarray:
    """A float grid on a fixed colour ramp from ``lo`` to ``hi``, black where NaN."""
    t = np.clip((np.nan_to_num(height.astype(np.float64), nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    pos = t * (len(_RAMP) - 1)
    rgb = np.stack([np.interp(pos, np.arange(len(_RAMP)), _RAMP[:, c]) for c in range(3)], axis=-1)
    rgb[np.isnan(height)] = 0
    return rgb.astype(np.uint8)


def with_robot(panel: np.ndarray, scale: int, half_width: float, half_length: float) -> Image.Image:
    """The grid panel upscaled, the robot drawn as a white rectangle at the origin, heading up."""
    img = Image.fromarray(panel).resize((panel.shape[1] * scale, panel.shape[0] * scale), Image.NEAREST)
    r = RESOLUTION / scale
    row0, col0 = X_MAX / r, Y_MAX / r    # pixel of the lidar
    ImageDraw.Draw(img).rectangle(
        [col0 - half_width / r, row0 - half_length / r, col0 + half_width / r, row0 + half_length / r],
        outline=(255, 255, 255), width=2)
    return img


def compose(panels: list[Image.Image], image: np.ndarray | None, out_png, title: str = "") -> Path:
    """Panels side by side, the camera image (resized to their height) first, a title on top."""
    panels = list(panels)
    height = panels[0].size[1]
    if image is not None:
        cam = Image.fromarray(np.asarray(image)).convert("RGB")
        cam = cam.resize((int(cam.width * height / cam.height), height), Image.BILINEAR)
        panels.insert(0, cam)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1), height + 20), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 20))
        x += p.width + 10
    ImageDraw.Draw(canvas).text((4, 4), title, fill=(0, 0, 0))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png
