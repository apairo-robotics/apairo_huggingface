"""PNGs of the BEV grids beside the camera image, for checking by eye.

One PNG per frame, four panels: the camera image, the mean height (fixed
colour scale ``HEIGHT_RANGE``, black = unknown), the majority class (ontology
colours, black = unknown) and the trav grid (green driven, red seen not
driven, black unknown). Every grid panel has the robot drawn as a white
rectangle at the origin, heading up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..bev_grid import TRAV_COLORS, compose, height_rgb, with_robot
from . import config
from .overlay import _PALETTE

HALF_LENGTH = 0.75   # Warthog: ~1.5 m long, 1.4 m wide, lidar near its centre


def render(elevation: np.ndarray, sem: np.ndarray, trav: np.ndarray, image: np.ndarray | None,
           out_png, scale: int = 3, title: str = "") -> Path:
    panels = [with_robot(p, scale, config.ROBOT_HALF_WIDTH, HALF_LENGTH)
              for p in (height_rgb(elevation[2]), _PALETTE[sem], TRAV_COLORS[trav])]
    return compose(panels, image, out_png, title)


def render_view(view, indices, png_dir) -> list[Path]:
    """Render frames of a processed view (the BEV channels must exist)."""
    from apairo import RawDataset

    from .bev import declaration

    view = Path(view)
    ds = RawDataset(view, declare=declaration(),
                    keys=[config.CAMERA, config.BEV_ELEVATION, config.BEV_SEMANTIC, config.BEV_TRAV])
    out = []
    for k in indices:
        out.append(render(ds.loaders[config.BEV_ELEVATION][k], ds.loaders[config.BEV_SEMANTIC][k],
                          ds.loaders[config.BEV_TRAV][k], ds.loaders[config.CAMERA][k],
                          Path(png_dir) / f"{view.name}_{k:06d}.png", title=f"{view.name} scan {k}"))
    return out
