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
from PIL import Image, ImageDraw

from . import config
from .bev import grid_shape
from .overlay import _PALETTE

HEIGHT_RANGE = (-1.0, 3.0)   # m above the ground under the robot
_RAMP = np.array([[48, 18, 59], [50, 136, 250], [27, 229, 181], [190, 245, 52],
                  [250, 140, 30], [122, 4, 3]], dtype=np.float64)   # turbo-like stops
TRAV_COLORS = np.zeros((256, 3), dtype=np.uint8)
TRAV_COLORS[0] = (220, 50, 50)
TRAV_COLORS[1] = (50, 220, 50)


def height_rgb(height: np.ndarray, lo: float = HEIGHT_RANGE[0], hi: float = HEIGHT_RANGE[1]) -> np.ndarray:
    t = np.clip((np.nan_to_num(height.astype(np.float64), nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    pos = t * (len(_RAMP) - 1)
    rgb = np.stack([np.interp(pos, np.arange(len(_RAMP)), _RAMP[:, c]) for c in range(3)], axis=-1)
    rgb[np.isnan(height)] = 0
    return rgb.astype(np.uint8)


def _robot(panel: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray(panel).resize((panel.shape[1] * scale, panel.shape[0] * scale), Image.NEAREST)
    r = config.BEV_RESOLUTION / scale
    row0 = config.BEV_X_MAX / r             # pixel row of x = 0
    col0 = config.BEV_Y_MAX / r             # pixel column of y = 0
    # Warthog: ~1.5 m long, 1.4 m wide, lidar near its centre.
    ImageDraw.Draw(img).rectangle(
        [col0 - config.ROBOT_HALF_WIDTH / r, row0 - 0.75 / r, col0 + config.ROBOT_HALF_WIDTH / r, row0 + 0.75 / r],
        outline=(255, 255, 255), width=2)
    return img


def render(elevation: np.ndarray, sem: np.ndarray, trav: np.ndarray, image: np.ndarray | None,
           out_png, scale: int = 3, title: str = "") -> Path:
    h, w = grid_shape()
    panels = [_robot(height_rgb(elevation[2]), scale), _robot(_PALETTE[sem], scale),
              _robot(TRAV_COLORS[trav], scale)]
    size = panels[0].size
    if image is not None:
        cam = Image.fromarray(np.asarray(image)).convert("RGB")
        cam = cam.resize((int(cam.width * size[1] / cam.height), size[1]), Image.BILINEAR)
        panels.insert(0, cam)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1), size[1] + 20), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 20))
        x += p.width + 10
    ImageDraw.Draw(canvas).text((4, 4), title, fill=(0, 0, 0))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return out_png


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
