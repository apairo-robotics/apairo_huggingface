"""PNGs of the TartanDrive BEV grids beside the camera image, for checking by eye.

One PNG per scan, five panels: the camera image nearest the scan, the mean
height (fixed scale ``bev_grid.HEIGHT_RANGE``, -1 to 3 m), the roughness and
the step (both on a 0 to ``GEOMETRY_MAX`` m scale), and the trav grid (green
driven, red seen not driven). Black is unknown everywhere; the vehicle is the
white rectangle at the origin, heading up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..bev_grid import TRAV_COLORS, compose, height_rgb, with_robot
from . import config
from .bev import open_view

GEOMETRY_MAX = 0.5   # m, top of the roughness / step colour scale


def render(elevation: np.ndarray, geometry: np.ndarray, trav: np.ndarray, image: np.ndarray | None,
           out_png, scale: int = 3, title: str = "") -> Path:
    layers = [height_rgb(elevation[2]), height_rgb(geometry[0], 0.0, GEOMETRY_MAX),
              height_rgb(geometry[1], 0.0, GEOMETRY_MAX), TRAV_COLORS[trav]]
    panels = [with_robot(p, scale, config.ROBOT_HALF_WIDTH, config.ROBOT_HALF_LENGTH) for p in layers]
    return compose(panels, image, out_png, title)


def render_view(view, indices, png_dir) -> list[Path]:
    """Render scans of a processed view; the image is the camera frame nearest each scan."""
    view = Path(view)
    ds = open_view(view, [config.CAMERA, config.BEV_ELEVATION, config.BEV_GEOMETRY, config.BEV_TRAV])
    camera_stamps = np.asarray(ds.timestamps[config.CAMERA])
    out = []
    for k in indices:
        t = float(ds.timestamps[config.BEV_TRAV][k])
        c = int(np.argmin(np.abs(camera_stamps - t)))
        image = ds.loaders[config.CAMERA][c] if abs(camera_stamps[c] - t) <= config.TOLERANCE else None
        out.append(render(ds.loaders[config.BEV_ELEVATION][k], ds.loaders[config.BEV_GEOMETRY][k],
                          ds.loaders[config.BEV_TRAV][k], image, Path(png_dir) / f"{view.name}_{k:06d}.png",
                          title=f"{view.name} scan {k}  |  camera | mean height -1..3 m | roughness 0..0.5 m"
                                f" | step 0..0.5 m | trav: green driven, red seen, black unknown"))
    return out
