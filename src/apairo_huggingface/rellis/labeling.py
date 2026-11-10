"""Lidar labels and trajectory traversability, projected into every RELLIS image.

RELLIS-3D annotates 13556 Ouster scans point by point but only 6235 of the
camera frames, about one in two. Every scan is co-captured with an image of the
same index, so projecting the scan labels gives a (sparse) semantic mask for
every image, and the driven path gives a traversability mask for every image.

Per sequence, in one pass over the camera clock:

1. ``make_view``  a directory under ``out_root`` that symlinks the source
                  channels: apairo reads through it and writes its registry and
                  the derived channels there, never into the shared source tree.
2. ``label_sequence``
   - ``synchronize()`` pairs each image with its scan and point labels
     (nearest stamp within ``TOLERANCE``: the scan borrows the image's stamp).
   - ``TraversabilityFromTrajectory`` labels the points inside the robot's
     future footprint from ``poses.txt`` (scans are labelled in chunks with a
     pose-only tail covering the look-ahead, so chunked = whole-sequence).
   - ``LidarCameraProjection`` (pinhole, ``T_camera_from_lidar``) then
     ``ImageMaskFromPointLabels`` paint ``sem_mask`` (ontology id) and
     ``trav_mask`` (0/1); the driven corridor is drawn on top of ``trav_mask``
     where it is not occluded. 255 = no lidar point projects there.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from apairo import RawDataset
from apairo.core.sample import Sample
from apairo_preprocess import (
    ImageMaskFromPointLabels,
    LidarCameraProjection,
    TraversabilityFromTrajectory,
)

from . import config
from .calibration import IMAGE_HEIGHT, IMAGE_WIDTH, camera_from_lidar, intrinsics

logger = logging.getLogger(__name__)

_EMPTY_SCAN = np.zeros((0, 3), dtype=np.float32)


def declaration() -> Path:
    """The packaged ``apairo.yaml`` describing one RELLIS-3D sequence."""
    return Path(str(resources.files(__package__) / "apairo.yaml"))


def make_view(root, seq: str, out_root) -> Path:
    """``out_root/<seq>``: symlinks to the source channels of one sequence."""
    src = Path(root) / "Rellis-3D" / seq
    view = Path(out_root) / seq
    view.mkdir(parents=True, exist_ok=True)
    for name in config.SOURCE_DIRS:
        link = view / name
        if not link.is_symlink():
            os.symlink(src / name, link)
    return view


def open_view(view, keys) -> RawDataset:
    return RawDataset(view, declare=declaration(), keys=list(keys))


def load_poses(root, seq: str) -> np.ndarray:
    """``(N, 4, 4)`` T_world_lidar, one per scan, from ``poses.txt``."""
    rows = np.loadtxt(Path(root) / "Rellis-3D" / seq / "poses.txt").reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3] = rows
    return poses


def path_length(poses: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))])


# --------------------------------------------------------------- corridor

def _project(points_cam: np.ndarray, K: np.ndarray):
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return u, v, z


def corridor_world(future: np.ndarray, step: float = 0.3):
    """Left/right ground edges of the footprint along ``future`` positions."""
    empty = np.empty((0, 3))
    seg = np.linalg.norm(np.diff(future[:, :2], axis=0), axis=1)
    pos = future[np.concatenate([[True], seg > 1e-3])]
    if len(pos) < 2:
        return empty, empty
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))])
    end = int(np.searchsorted(path, config.FORWARD_DISTANCE, side="right"))
    pos, path = pos[:end], path[:end]
    s = np.arange(0.0, path[-1], step)
    if len(s) < 2:
        return empty, empty
    x, y = np.interp(s, path, pos[:, 0]), np.interp(s, path, pos[:, 1])
    z = np.interp(s, path, pos[:, 2]) + config.GROUND_OFFSET
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    lx, ly = -dy / norm, dx / norm
    hw = config.ROBOT_HALF_WIDTH
    return (np.column_stack([x + hw * lx, y + hw * ly, z]),
            np.column_stack([x - hw * lx, y - hw * ly, z]))


def corridor_mask(left_w, right_w, cam_from_world, K, uv_lidar, *, bin_px=16, margin=1.0):
    """(H, W) bool: the corridor as filled quads, skipping occluded ones.

    A quad is drawn only if no lidar return in its pixel bins is more than
    ``margin`` metres nearer than the quad itself (an obstacle in front)."""
    mask = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    if len(left_w) < 2:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)

    def to_cam(p):
        return p @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]

    ul, vl, zl = _project(to_cam(left_w), K)
    ur, vr, zr = _project(to_cam(right_w), K)
    bx, by = -(-IMAGE_WIDTH // bin_px), -(-IMAGE_HEIGHT // bin_px)
    nearest = np.full(by * bx, np.inf)
    ok = np.isfinite(uv_lidar[:, 0])
    ids = (uv_lidar[ok, 1].astype(int) // bin_px) * bx + uv_lidar[ok, 0].astype(int) // bin_px
    np.minimum.at(nearest, ids, uv_lidar[ok, 2].astype(float))
    nearest = nearest.reshape(by, bx)

    def depth_at(u, v):
        return nearest[np.clip(int(v) // bin_px, 0, by - 1), np.clip(int(u) // bin_px, 0, bx - 1)]

    draw = ImageDraw.Draw(mask)
    for i in range(len(ul) - 1):
        zs = (zl[i], zr[i], zl[i + 1], zr[i + 1])
        if min(zs) < 0.5:
            continue
        quad = [(ul[i], vl[i]), (ur[i], vr[i]), (ur[i + 1], vr[i + 1]), (ul[i + 1], vl[i + 1])]
        if not all(np.isfinite(c) for q in quad for c in q):
            continue
        if min(depth_at(u, v) for u, v in quad) + margin < min(zs):
            continue
        draw.polygon(quad, fill=1)
    return np.asarray(mask, dtype=bool)


# ------------------------------------------------------------------ masks

def label_sequence(root, seq: str, out_root, *, overwrite: bool = False,
                   limit: int | None = None, chunk: int = 64) -> dict:
    """Write ``sem_mask`` and ``trav_mask`` for every image of one sequence."""
    t0 = time.time()
    view = make_view(root, seq, out_root)
    if (view / config.TRAV_MASK / "timestamps.txt").exists() and not overwrite:
        logger.info("%s: masks exist, skipping", seq)
        return {}
    for name in (config.SEM_MASK, config.TRAV_MASK):
        try:
            RawDataset.remove_channel(view, name, data=True)
        except (KeyError, FileNotFoundError):  # not registered yet / fresh view
            pass
        shutil.rmtree(view / name, ignore_errors=True)
        (view / name).mkdir()

    ds = open_view(view, [config.CAMERA, config.LIDAR, config.LABELS])
    sync = ds.synchronize(
        reference=config.CAMERA,
        method={config.LIDAR: "nearest", config.LABELS: "nearest"},
        tolerance=config.TOLERANCE,
    )
    scan_of = np.asarray(sync.frame_indices[config.LIDAR])
    camera_files = ds.loaders[config.CAMERA].files
    # synchronize() drops the images without a scan within TOLERANCE; masks
    # are indexed by image below, so every image must have kept its scan.
    if len(sync) != len(camera_files):
        raise ValueError(f"{seq}: {len(camera_files) - len(sync)} of {len(camera_files)} images "
                         f"have no scan within {config.TOLERANCE} s")
    stamps = np.asarray(ds.timestamps[config.CAMERA], dtype=np.float64)

    poses = load_poses(root, seq)
    if len(poses) != len(ds.timestamps[config.LIDAR]):
        raise ValueError(f"{seq}: {len(poses)} poses for {len(ds.timestamps[config.LIDAR])} scans")
    path = path_length(poses)

    K = intrinsics(root, seq)
    E = camera_from_lidar(root, seq)
    projector = LidarCameraProjection(intrinsics=K, extrinsics=E,
                                      image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), lidar_key="lidar")
    sem_masker = ImageMaskFromPointLabels(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), labels_key="labels",
        radius=config.SPLAT_RADIUS, occlusion_bin_px=config.OCCLUSION_BIN_PX,
    )
    trav_masker = ImageMaskFromPointLabels(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), labels_key="trav",
        radius=config.SPLAT_RADIUS, occlusion_bin_px=config.OCCLUSION_BIN_PX,
    )
    labeler = TraversabilityFromTrajectory(
        robot_radius=config.ROBOT_HALF_WIDTH,
        height_min=config.HEIGHT_MIN,
        height_max=config.HEIGHT_MAX,
        forward_distance=config.FORWARD_DISTANCE,
    )

    n = len(sync) if limit is None else min(limit, len(sync))
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        samples = [sync[k] for k in range(a, b)]
        scans = []
        for s in samples:
            p = np.asarray(s.data[config.LIDAR])[:, :3]
            keep = np.linalg.norm(p, axis=1) >= config.MIN_RANGE
            scans.append((p, keep))
        # Labels depend on future poses only: a pose-only tail covering the
        # look-ahead makes chunked labels equal whole-sequence ones.
        j0, j1 = int(scan_of[a]), int(scan_of[b - 1])
        tail = min(len(poses), int(np.searchsorted(path, path[j1] + config.FORWARD_DISTANCE)) + 1)
        frames = [Sample(data={"lidar": p[keep], "poses": poses[int(scan_of[k])]})
                  for k, (p, keep) in zip(range(a, b), scans)]
        frames += [Sample(data={"lidar": _EMPTY_SCAN, "poses": poses[j]}) for j in range(j1 + 1, tail)]
        trav_all = labeler(iter(frames))[: b - a]

        for k, s, (p, keep), trav in zip(range(a, b), samples, scans, trav_all):
            j = int(scan_of[k])
            pts = p[keep].astype(np.float32)
            labels = np.asarray(s.data[config.LABELS])[keep]
            uv = projector(Sample(data={"lidar": pts}))
            # LidarCameraProjection bounds u, v in float64 but returns float32,
            # where 1199.99997 rounds to 1200.0 -- one row past the image.
            uv[:, 0] = np.minimum(uv[:, 0], IMAGE_WIDTH - 1)
            uv[:, 1] = np.minimum(uv[:, 1], IMAGE_HEIGHT - 1)
            sem = sem_masker(Sample(data={"labels": labels, "lidar_uv": uv}))
            if path[-1] - path[j] >= config.MIN_FUTURE_PATH:
                tm = trav_masker(Sample(data={"trav": np.asarray(trav, dtype=np.uint8), "lidar_uv": uv}))
                future = poses[j:, :3, 3]   # from the current pose on
                left, right = corridor_world(future)
                tm[corridor_mask(left, right, E @ np.linalg.inv(poses[j]), K, uv)] = 1
            else:
                tm = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), config.NO_DATA, dtype=np.uint8)
            stem = Path(camera_files[k]).stem
            Image.fromarray(sem, mode="L").save(view / config.SEM_MASK / f"{stem}.png")
            Image.fromarray(tm, mode="L").save(view / config.TRAV_MASK / f"{stem}.png")
        logger.info("%s: frames [%d, %d) / %d  %.0f s", seq, a, b, n, time.time() - t0)

    for name in (config.SEM_MASK, config.TRAV_MASK):
        np.savetxt(view / name / "timestamps.txt", stamps[:n], fmt="%.3f")
        RawDataset.register_channel(view, name, "img", sources=[config.LIDAR, config.LABELS, config.CAMERA])
    report = {"sequence": seq, **coverage(view), "seconds": round(time.time() - t0)}
    logger.info("%s", report)
    return report


def coverage(view) -> dict:
    """Mask statistics of one processed view, read back from the written masks.

    ``sem_mask`` coverage is averaged over every image. The ``trav_mask``
    figures are averaged over the frames that have one (any pixel other than
    255; the others are all-255 for lack of path ahead), so both are shares of
    the same images and driven <= with data holds per frame and on average."""
    view = Path(view)
    sem_cov = [float((np.asarray(Image.open(f)) != config.NO_DATA).mean())
               for f in sorted((view / config.SEM_MASK).glob("*.png"))]
    trav_cov, trav_pos = [], []
    for f in sorted((view / config.TRAV_MASK).glob("*.png")):
        tm = np.asarray(Image.open(f))
        if (tm != config.NO_DATA).any():
            trav_cov.append(float((tm != config.NO_DATA).mean()))
            trav_pos.append(float((tm == 1).mean()))
    return {
        "images": len(sem_cov),
        "sem_mask_pixel_coverage_mean": float(np.mean(sem_cov)) if sem_cov else 0.0,
        "trav_mask_frames_with_data": len(trav_cov),
        "trav_mask_pixel_coverage_mean": float(np.mean(trav_cov)) if trav_cov else 0.0,
        "trav_mask_driven_fraction_mean": float(np.mean(trav_pos)) if trav_pos else 0.0,
    }
