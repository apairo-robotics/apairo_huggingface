"""Overlay PNGs of the derived masks on their images, for checking by eye.

One PNG per frame, three panels side by side: the projected semantic mask
(ontology colours), the trav mask (green driven, red seen not driven) and,
where one exists, the official 2D annotation -- all over the dimmed image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import config

_PALETTE = np.zeros((256, 3), dtype=np.uint8)
for _id, (_name, _rgb) in config.ONTOLOGY.items():
    _PALETTE[_id] = _rgb


def _blend(image: np.ndarray, mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    out = (image * 0.45).astype(np.uint8)
    has = mask != config.NO_DATA
    out[has] = colors[mask[has]]
    return out


def overlay(view, stem: str, out_png, scale: int = 2) -> Path:
    view = Path(view)
    image = np.asarray(Image.open(view / config.CAMERA / f"{stem}.jpg").convert("RGB"))
    sem = np.asarray(Image.open(view / config.SEM_MASK / f"{stem}.png"))
    trav = np.asarray(Image.open(view / config.TRAV_MASK / f"{stem}.png"))
    trav_colors = np.zeros((256, 3), dtype=np.uint8)
    trav_colors[0] = (230, 40, 40)
    trav_colors[1] = (40, 230, 40)
    panels = [_blend(image, sem, _PALETTE), _blend(image, trav, trav_colors)]
    label_2d = view / config.LABEL_2D / f"{stem}.png"
    if label_2d.exists():
        ref = np.asarray(Image.open(label_2d))
        panels.append((0.5 * image + 0.5 * _PALETTE[ref]).astype(np.uint8))
    else:
        panels.append(image)
    grid = Image.fromarray(np.hstack(panels))
    grid = grid.resize((grid.width // scale, grid.height // scale), Image.BILINEAR)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)
    return out_png
