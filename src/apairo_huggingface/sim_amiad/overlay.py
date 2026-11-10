"""Overlay PNGs for checking the geometry and the labels by eye.

One PNG per scan, a 2x2 grid over the camera image:

- top left:     every lidar point projected (camera ZED2_pcl), coloured by
                depth, over the RGB image -- the calibration check;
- top right:    the sim segmentation (palette of semantic_classes.json);
- bottom left:  the visible points coloured by their transferred class
                (``point_sem``), hidden / outside points not drawn;
- bottom right: ``trav_mask`` (green driven, red seen not driven) with the
                ``trav_traj`` positives as white dots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from apairo import RawDataset
from apairo.core.sample import Sample
from apairo_preprocess import LidarCameraProjection

from . import config
from .labeling import POINT_SEM, TRAV_MASK, TRAV_TRAJ

PALETTE = np.zeros((256, 3), dtype=np.uint8)
for _id, (_name, _rgb, _trav) in config.CLASSES.items():
    PALETTE[_id] = _rgb


def _dots(image: np.ndarray, uv: np.ndarray, colors: np.ndarray, r: int = 2) -> np.ndarray:
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    for (u, v), c in zip(uv, colors):
        draw.ellipse([u - r, v - r, u + r, v + r], fill=tuple(int(x) for x in c))
    return np.asarray(im)


def _depth_colors(z: np.ndarray, zmax: float = 20.0) -> np.ndarray:
    a = np.clip(z / zmax, 0, 1)
    return np.stack([255 * (1 - a), 255 * a, np.full_like(a, 255)], axis=1).astype(np.uint8)


def overlay(seq_dir, k: int, out_png, scale: int = 2) -> Path:
    """Grid for raw lidar index ``k`` (the derived files share its name)."""
    seq_dir = Path(seq_dir)
    lidar_ts = np.loadtxt(seq_dir / config.LIDAR / "timestamps.txt", ndmin=1)
    cam_ts = np.loadtxt(seq_dir / config.RGB / "timestamps.txt", ndmin=1)
    c = int(np.argmin(np.abs(cam_ts - lidar_ts[k])))
    if abs(cam_ts[c] - lidar_ts[k]) > config.SYNC_TOLERANCE:
        raise ValueError(f"scan {k}: no image on its stamp")
    cal = RawDataset(seq_dir, keys=[config.LIDAR]).calibration
    proj = LidarCameraProjection(intrinsics=cal.get_intrinsics(config.CAMERA_FRAME),
                                 extrinsics=cal.get_tf(config.LIDAR_FRAME, config.CAMERA_FRAME),
                                 image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    pts = np.load(seq_dir / config.LIDAR / f"{k:06d}.npy")
    uv = proj(Sample(data={"lidar": pts}))
    ok = np.isfinite(uv[:, 0])
    rgb = np.asarray(Image.open(seq_dir / config.RGB / f"{c:06d}.png").convert("RGB"))
    sem_img = np.asarray(Image.open(seq_dir / config.SEMANTIC / f"{c:06d}.png"))
    sem = np.load(seq_dir / POINT_SEM / f"{k:06d}.npy")
    trav = np.load(seq_dir / TRAV_TRAJ / f"{k:06d}.npy")
    tm = np.asarray(Image.open(seq_dir / TRAV_MASK / f"{k:06d}.png"))
    dim = (rgb * 0.5).astype(np.uint8)

    p1 = _dots(rgb, uv[ok, :2], _depth_colors(uv[ok, 2]))
    p2 = PALETTE[sem_img]
    has = sem != config.NO_DATA
    p3 = _dots(dim, uv[has, :2], PALETTE[sem[has]])
    p4 = dim.copy()
    p4[tm == 1] = (0.5 * p4[tm == 1] + [0, 120, 0]).astype(np.uint8)
    p4[tm == 0] = (0.5 * p4[tm == 0] + [120, 0, 0]).astype(np.uint8)
    pos = ok & (trav == 1)
    p4 = _dots(p4, uv[pos, :2], np.full((pos.sum(), 3), 255), r=1)
    grid = Image.fromarray(np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])]))
    grid = grid.resize((grid.width // scale, grid.height // scale), Image.BILINEAR)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_png)
    return out_png
