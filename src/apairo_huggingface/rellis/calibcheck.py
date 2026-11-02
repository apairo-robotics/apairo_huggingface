"""Check the lidar -> camera calibration against the official 2D annotations.

For a random sample of annotated frames of each sequence, every candidate chain
projects the raw scan, and the score is the share of projected points whose
lidar label equals the 2D label at their pixel (void excluded on both sides).
A wrong chain scores far lower; the right one is also checked by eye on the
overlays (points coloured by label over the image).

Candidates: the chosen chain (inverse of ``transforms.yaml``, pinhole, no
distortion), the stored transform used without inversion, and the plumb-bob
coefficients hard-coded in the official ``lidar2img`` notebook. The chosen
chain is also scored under vertical pixel shifts, to measure any residual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import config
from .calibration import IMAGE_HEIGHT, IMAGE_WIDTH, camera_from_lidar, intrinsics, lidar_from_camera

NOTEBOOK_DISTORTION = np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0.0])
SHIFTS = [-16, -8, 0, 8, 16, 24, 32]


def project(xyz, T, K, dist=None):
    """Pixel coordinates (u, v) and row indices of the points that land in the image."""
    cam = xyz @ T[:3, :3].T + T[:3, 3]
    front = cam[:, 2] > 0.5
    x, y = cam[front, 0] / cam[front, 2], cam[front, 1] / cam[front, 2]
    if dist is not None:
        k1, k2, p1, p2, k3 = dist
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    u, v = K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]
    inside = (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    return u[inside], v[inside], np.flatnonzero(front)[inside]


def _frame(root, seq, label_path):
    idx = int(label_path.name[5:11])
    sdir = Path(root) / "Rellis-3D" / seq
    xyz = np.fromfile(sdir / "os1_cloud_node_kitti_bin" / f"{idx:06d}.bin", np.float32).reshape(-1, 4)[:, :3]
    labels = np.fromfile(sdir / "os1_cloud_node_semantickitti_label_id" / f"{idx:06d}.label", np.uint32) & 0xFFFF
    return xyz.astype(np.float64), labels, np.asarray(Image.open(label_path))


def _agree(u, v, labels, ref, dv=0):
    g = ref[np.clip((v + dv).astype(int), 0, IMAGE_HEIGHT - 1), u.astype(int)]
    keep = (g != 0) & (labels != 0)
    return float((g[keep] == labels[keep]).mean()) if keep.any() else float("nan")


def calibration_check(root, sequences, frames_per_seq: int = 8, seed: int = 0, png_dir=None) -> dict:
    rng = np.random.default_rng(seed)
    report = {}
    for seq in sequences:
        K = intrinsics(root, seq)
        chains = {
            "chosen: inv(transforms.yaml), no distortion": (camera_from_lidar(root, seq), None),
            "transforms.yaml not inverted": (lidar_from_camera(root, seq), None),
            "inv(transforms.yaml), notebook distortion": (camera_from_lidar(root, seq), NOTEBOOK_DISTORTION),
        }
        refs = sorted((Path(root) / "Rellis-3D" / seq / config.LABEL_2D).iterdir())
        picks = [refs[i] for i in rng.choice(len(refs), frames_per_seq, replace=False)]
        scores = {name: [] for name in chains}
        shift = np.zeros(len(SHIFTS))
        for path in picks:
            xyz, labels, ref = _frame(root, seq, path)
            for name, (T, dist) in chains.items():
                u, v, rows = project(xyz, T, K, dist)
                scores[name].append(_agree(u, v, labels[rows], ref))
                if name.startswith("chosen"):
                    shift += [_agree(u, v, labels[rows], ref, dv) for dv in SHIFTS]
            if png_dir is not None:
                overlay_points(root, seq, path, xyz, labels, Path(png_dir) / f"calib_{seq}_{path.stem}.png")
        report[seq] = {name: float(np.nanmean(s)) for name, s in scores.items()}
        report[seq]["chosen, vertical shift dv (px) -> agreement"] = dict(zip(SHIFTS, np.round(shift / len(picks), 4).tolist()))
    return report


def overlay_points(root, seq, label_path, xyz, labels, out_png):
    """The image with the projected points coloured by their lidar label, next
    to the official 2D annotation."""
    image_path = next((Path(root) / "Rellis-3D" / seq / config.CAMERA).glob(f"{label_path.stem}.jpg"))
    image = np.asarray(Image.open(image_path).convert("RGB"))
    u, v, rows = project(xyz, camera_from_lidar(root, seq), intrinsics(root, seq))
    canvas = Image.fromarray((image * 0.5).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for a, b, lab in zip(u, v, labels[rows]):
        rgb = config.ONTOLOGY.get(int(lab), ("", (255, 255, 255)))[1]
        draw.ellipse([a - 3, b - 3, a + 3, b + 3], fill=tuple(rgb))
    palette = np.zeros((256, 3), np.uint8)
    for cid, (_, rgb) in config.ONTOLOGY.items():
        palette[cid] = rgb
    ref = (0.5 * image + 0.5 * palette[np.asarray(Image.open(label_path))]).astype(np.uint8)
    grid = Image.fromarray(np.hstack([np.asarray(canvas), ref]))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid.resize((grid.width // 2, grid.height // 2), Image.BILINEAR).save(out_png)
