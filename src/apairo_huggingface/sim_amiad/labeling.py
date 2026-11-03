"""Sim semantic ground truth transferred to the lidar, and trajectory traversability.

Per extracted sequence, three derived channels are written beside the raw ones
(one file per lidar scan, on the lidar clock):

- ``point_sem``  (N,) uint8: training class id of each point, read from the
  perfect segmentation image of the same stamp. A point gets a label only if
  it is VISIBLE: it projects into the image (``LidarCameraProjection``,
  ``T_ZED2_pcl<-sensor`` and the camera_info K and D) and its camera depth agrees
  with the sim depth image at that pixel within ``DEPTH_TOLERANCE``. Points
  outside the image, or hidden behind something the camera sees, get 255.
- ``trav_traj``  (N,) uint8: 1 if the point lies in the robot's future
  footprint (``TraversabilityFromTrajectory``, oriented rectangle, points and
  poses in ``base_link``), else 0. The usual self-supervised PU label.
- ``trav_mask``  (H, W) uint8 PNG on the camera image: 1 driven (points of
  ``trav_traj`` plus the corridor of the future path where the depth image
  shows it is not hidden), 0 seen by the lidar but not driven, 255 no data.
  Frames with less than ``MIN_FUTURE_PATH`` of path ahead are all 255.

Files are named by the raw lidar index; ``scored_frames.npy`` lists the raw
indices of the scans with enough path ahead to be scored.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from apairo import RawDataset
from apairo.core.sample import Sample
from apairo_preprocess import ImageMaskFromPointLabels, LidarCameraProjection, TraversabilityFromTrajectory
from apairo_preprocess.projection.lidar_to_camera import _distort_plumb_bob

from . import config
from .extract import pose7_to_matrix

logger = logging.getLogger(__name__)

POINT_SEM = "point_sem"
TRAV_TRAJ = "trav_traj"
TRAV_MASK = "trav_mask"

# Visibility test against the depth image: |z_point - z_image| <= tol.
DEPTH_TOLERANCE = 0.15         # m, plus DEPTH_TOLERANCE_REL * z
DEPTH_TOLERANCE_REL = 0.02

# Barakuda: wheel contacts at x +-0.525, y +-0.4775 in base_link; tyres add
# about 0.2 m around them.
ROBOT_LENGTH = 1.45
ROBOT_WIDTH = 1.15
FORWARD_DISTANCE = 30.0        # m of path
MIN_FUTURE_PATH = 10.0         # m; less path ahead -> the frame is not scored
# base_link sits BASE_HEIGHT above the ground under the wheels: suspension
# links at z -0.11 plus the wheel radius; the lidar agrees (5th percentile of
# return z in a 1.5-3 m ring around the base, ``ground_z_in_base`` in the report).
BASE_HEIGHT = 0.41
HEIGHT_MIN = -BASE_HEIGHT - 0.3
HEIGHT_MAX = -BASE_HEIGHT + 0.5
# Returns off the robot itself: sensor-frame cells of SELF_VOXEL metres hit in
# at least SELF_MIN_SHARE of SELF_SAMPLES scans spread over the sequence (the
# body does not move in the sensor frame, the world does) AND higher than
# SELF_MIN_Z in base_link. The height gate matters: on flat ground the fixed
# beams hit the ground just ahead in the same sensor-frame cells every scan,
# and those returns are visible in the camera.
SELF_VOXEL = 0.05
SELF_SAMPLES = 100
SELF_MIN_SHARE = 0.8
SELF_MIN_Z = -BASE_HEIGHT + 0.15

SPLAT_RADIUS = 2
OCCLUSION_BIN_PX = 8


def open_sequence(seq_dir, keys) -> RawDataset:
    return RawDataset(seq_dir, keys=list(keys))


def _cells(points: np.ndarray) -> np.ndarray:
    v = np.floor(points / SELF_VOXEL).astype(np.int64) + 2**20
    return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]


def self_cells(scans: list[np.ndarray]) -> np.ndarray:
    """Sensor-frame voxel keys occupied in >= SELF_MIN_SHARE of sampled scans."""
    picks = np.linspace(0, len(scans) - 1, min(SELF_SAMPLES, len(scans))).astype(int)
    keys, hits = np.unique(np.concatenate([np.unique(_cells(scans[k])) for k in picks]), return_counts=True)
    return keys[hits >= SELF_MIN_SHARE * len(picks)]


def self_mask(points_sensor: np.ndarray, cells: np.ndarray, base_from_lidar: np.ndarray) -> np.ndarray:
    z_base = points_sensor @ base_from_lidar[2, :3] + base_from_lidar[2, 3]
    return np.isin(_cells(points_sensor), cells) & (z_base > SELF_MIN_Z)


def visible_labels(uv: np.ndarray, semantic: np.ndarray, depth_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class id per point (255 if not visible) and the visibility mask."""
    labels = np.full(len(uv), config.NO_DATA, dtype=np.uint8)
    inside = np.isfinite(uv[:, 0])
    u = uv[inside, 0].astype(np.intp)
    v = uv[inside, 1].astype(np.intp)
    z = uv[inside, 2].astype(np.float64)
    z_img = depth_mm[v, u].astype(np.float64) / 1000.0
    ok = (z_img > 0) & (np.abs(z - z_img) <= DEPTH_TOLERANCE + DEPTH_TOLERANCE_REL * z)
    idx = np.flatnonzero(inside)[ok]
    labels[idx] = semantic[v[ok], u[ok]]
    visible = np.zeros(len(uv), dtype=bool)
    visible[idx] = True
    return labels, visible


def path_length(positions: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])


def corridor_world(future_poses: np.ndarray, step: float = 0.25):
    """Left/right ground edges of the footprint along the future base poses."""
    pos = future_poses[:, :3, 3]
    path = path_length(pos)
    end = int(np.searchsorted(path, FORWARD_DISTANCE, side="right"))
    pos, path = pos[:end], path[:end]
    keep = np.concatenate([[True], np.diff(path) > 1e-3])
    pos, path = pos[keep], path[keep]
    if len(pos) < 2:
        return np.empty((0, 3)), np.empty((0, 3))
    s = np.arange(0.0, path[-1], step)
    x, y = np.interp(s, path, pos[:, 0]), np.interp(s, path, pos[:, 1])
    z = np.interp(s, path, pos[:, 2]) - BASE_HEIGHT
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    lx, ly = -dy / norm, dx / norm
    hw = ROBOT_WIDTH / 2
    return np.column_stack([x + hw * lx, y + hw * ly, z]), np.column_stack([x - hw * lx, y - hw * ly, z])


def camera_project(points_cam: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera optical-frame points -> continuous ``u, v`` and depth ``z``, NOT
    clipped to the image (a corridor quad may straddle its border).

    The same camera model as ``LidarCameraProjection`` (K with skew, plumb-bob
    distortion of ``cam``), so the corridor and the lidar points land on the
    same pixels. ``u, v`` are meaningless where ``z <= 0``.
    """
    if cam.model != "plumb_bob" and len(cam.distortion):
        raise ValueError(f"only the plumb_bob distortion model is supported, got {cam.model!r}")
    K = np.asarray(cam.K, dtype=np.float64)
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = points_cam[:, 0] / z, points_cam[:, 1] / z
    if len(cam.distortion):
        d = np.asarray(cam.distortion, dtype=np.float64).ravel()
        x, y = _distort_plumb_bob(x, y, np.append(d, 0.0) if d.size == 4 else d)
    return K[0, 0] * x + K[0, 1] * y + K[0, 2], K[1, 1] * y + K[1, 2], z


def corridor_mask(left_w, right_w, cam_from_world, cam, depth_mm, margin: float = 0.5) -> np.ndarray:
    """(H, W) bool: corridor quads, each drawn only where the depth image does
    not show something nearer (by ``margin``) at the quad's corners. ``cam`` is
    the ``CameraIntrinsics`` the lidar projection uses (see ``camera_project``)."""
    H, W = depth_mm.shape
    mask = Image.new("L", (W, H), 0)
    if len(left_w) < 2:
        return np.zeros((H, W), dtype=bool)

    def project(p):
        return camera_project(p @ cam_from_world[:3, :3].T + cam_from_world[:3, 3], cam)

    ul, vl, zl = project(left_w)
    ur, vr, zr = project(right_w)
    depth = depth_mm.astype(np.float64) / 1000.0
    depth[depth == 0] = np.inf
    draw = ImageDraw.Draw(mask)
    for i in range(len(ul) - 1):
        corners = [(ul[i], vl[i], zl[i]), (ur[i], vr[i], zr[i]), (ur[i + 1], vr[i + 1], zr[i + 1]),
                   (ul[i + 1], vl[i + 1], zl[i + 1])]
        if min(c[2] for c in corners) < 0.3:
            continue
        hidden = 0
        for u, v, z in corners:
            if 0 <= u < W and 0 <= v < H and depth[int(v), int(u)] + margin < z:
                hidden += 1
        if hidden >= 2:
            continue
        draw.polygon([(u, v) for u, v, _ in corners], fill=1)
    return np.asarray(mask, dtype=bool)


def label_sequence(seq_dir, *, overwrite: bool = False, limit: int | None = None) -> dict:
    """Write point_sem, trav_traj and trav_mask for every lidar scan of ``seq_dir``."""
    t0 = time.time()
    seq_dir = Path(seq_dir)
    if (seq_dir / TRAV_MASK / "timestamps.txt").exists() and not overwrite:
        logger.info("%s: labels exist, skipping", seq_dir)
        return {}
    for name in (POINT_SEM, TRAV_TRAJ, TRAV_MASK):
        try:
            RawDataset.remove_channel(seq_dir, name, data=True)
        except (KeyError, FileNotFoundError):
            pass
        shutil.rmtree(seq_dir / name, ignore_errors=True)
        (seq_dir / name).mkdir()

    keys = [config.LIDAR, config.SEMANTIC, config.DEPTH, config.POSE]
    ds = open_sequence(seq_dir, keys)
    sync = ds.synchronize(reference=config.LIDAR,
                          method={k: "nearest" for k in keys[1:]}, tolerance=config.SYNC_TOLERANCE)
    for k in keys[1:]:
        off = np.abs(sync.time_offsets(k))
        if off.max() > config.SYNC_TOLERANCE:
            raise ValueError(f"{k}: a scan has no {k} within {config.SYNC_TOLERANCE} s (max {off.max():.4f})")
    n = len(sync) if limit is None else min(limit, len(sync))
    # Derived files are named by the RAW lidar index, so point_sem/000123.npy
    # is row-aligned with lidar/000123.npy even when sync drops edge scans.
    raw_index = np.asarray(sync.frame_indices[config.LIDAR])[:n]
    stamps = np.asarray(ds.timestamps[config.LIDAR], dtype=np.float64)[raw_index]

    cal = ds.calibration
    cam = cal.get_intrinsics(config.CAMERA_FRAME)
    cam_from_lidar = cal.get_tf(config.LIDAR_FRAME, config.CAMERA_FRAME)
    base_from_lidar = cal.get_tf(config.LIDAR_FRAME, config.BASE_FRAME)
    cam_from_base = cal.get_tf(config.BASE_FRAME, config.CAMERA_FRAME)
    H, W = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    projector = LidarCameraProjection(intrinsics=cam, extrinsics=cam_from_lidar, image_size=(H, W))
    trav_masker = ImageMaskFromPointLabels(image_size=(H, W), labels_key="trav", radius=SPLAT_RADIUS,
                                           occlusion_bin_px=OCCLUSION_BIN_PX)
    labeler = TraversabilityFromTrajectory(robot_length=ROBOT_LENGTH, robot_width=ROBOT_WIDTH,
                                           height_min=HEIGHT_MIN, height_max=HEIGHT_MAX,
                                           forward_distance=FORWARD_DISTANCE)

    # Pass 1: points (base frame, self returns removed) and poses of every scan.
    scans, poses = [], []
    for k in range(n):
        s = sync[k]
        scans.append(np.asarray(s.data[config.LIDAR], dtype=np.float64))
        poses.append(pose7_to_matrix(s.data[config.POSE]))
    poses = np.stack(poses)
    cells = self_cells(scans)
    keeps = [~self_mask(p, cells, base_from_lidar) for p in scans]
    path = path_length(poses[:, :3, 3])
    frames = [Sample(data={"lidar": (scans[k][keeps[k]] @ base_from_lidar[:3, :3].T + base_from_lidar[:3, 3]),
                           "poses": poses[k]}) for k in range(n)]
    trav_kept = labeler(iter(frames))
    del frames
    logger.info("%s: trajectory labels for %d scans  %.0f s", seq_dir.name, n, time.time() - t0)

    # Ground height under the base, for the record (BASE_HEIGHT is a constant).
    near_z = []
    for k in range(0, n, max(1, n // 50)):
        pb = scans[k] @ base_from_lidar[:3, :3].T + base_from_lidar[:3, 3]
        r = np.hypot(pb[:, 0], pb[:, 1])
        sel = (r > 1.5) & (r < 3.0) & keeps[k]
        if sel.any():
            near_z.append(float(np.percentile(pb[sel, 2], 5)))

    # Pass 2: visibility-tested semantic labels, trav mask.
    counts = np.zeros(256, dtype=np.int64)
    n_points = n_inside = n_visible = n_self = 0
    scored = np.zeros(n, dtype=bool)
    for k in range(n):
        s = sync[k]
        p, keep = scans[k], keeps[k]
        uv = projector(Sample(data={"lidar": p.astype(np.float32)}))
        uv[:, 0] = np.minimum(uv[:, 0], W - 1)      # float32 rounding at the far edge
        uv[:, 1] = np.minimum(uv[:, 1], H - 1)
        semantic = np.asarray(s.data[config.SEMANTIC])
        depth_mm = np.asarray(s.data[config.DEPTH])
        labels, visible = visible_labels(uv, semantic, depth_mm)
        labels[~keep] = config.NO_DATA
        trav = np.zeros(len(p), dtype=np.uint8)
        trav[keep] = np.asarray(trav_kept[k], dtype=np.uint8)

        if path[-1] - path[k] >= MIN_FUTURE_PATH:
            scored[k] = True
            uv_keep = uv.copy()
            uv_keep[~keep] = np.nan
            tm = trav_masker(Sample(data={"trav": trav, "lidar_uv": uv_keep}))
            left, right = corridor_world(poses[k:])
            cam_from_world = cam_from_base @ np.linalg.inv(poses[k])
            tm[corridor_mask(left, right, cam_from_world, cam, depth_mm)] = 1
        else:
            tm = np.full((H, W), config.NO_DATA, dtype=np.uint8)

        np.save(seq_dir / POINT_SEM / f"{raw_index[k]:06d}.npy", labels)
        np.save(seq_dir / TRAV_TRAJ / f"{raw_index[k]:06d}.npy", trav)
        Image.fromarray(tm).save(seq_dir / TRAV_MASK / f"{raw_index[k]:06d}.png")
        counts += np.bincount(labels, minlength=256)
        n_points += len(p)
        n_inside += int(np.isfinite(uv[:, 0]).sum())
        n_visible += int((visible & keep).sum())
        n_self += int((~keep).sum())
        if k % 500 == 0:
            logger.info("%s: %d / %d  %.0f s", seq_dir.name, k, n, time.time() - t0)

    for name, loader in ((POINT_SEM, "npys"), (TRAV_TRAJ, "npys"), (TRAV_MASK, "img")):
        np.savetxt(seq_dir / name / "timestamps.txt", stamps, fmt="%.9f")
        RawDataset.register_channel(seq_dir, name, loader, timestamps_from=config.LIDAR,
                                    sources=[config.LIDAR, config.SEMANTIC, config.DEPTH, config.POSE])
    np.save(seq_dir / "scored_frames.npy", raw_index[scored])

    report = {
        "sequence": seq_dir.name,
        "scans": n,
        "scored_scans": int(scored.sum()),
        "path_length_m": float(path[-1]),
        "points": n_points,
        "points_self_removed": n_self,
        "points_in_image": n_inside,
        "points_visible_labeled": n_visible,
        "point_sem_counts": {config.CLASSES[c][0]: int(counts[c]) for c in config.CLASSES},
        "point_sem_no_label": int(counts[config.NO_DATA]),
        "ground_z_in_base": float(np.median(near_z)) if near_z else None,
        "self_cells": int(len(cells)),
        "params": {"depth_tolerance": [DEPTH_TOLERANCE, DEPTH_TOLERANCE_REL], "robot": [ROBOT_LENGTH, ROBOT_WIDTH],
                   "height_window": [HEIGHT_MIN, HEIGHT_MAX], "forward_distance": FORWARD_DISTANCE,
                   "min_future_path": MIN_FUTURE_PATH},
        "seconds": round(time.time() - t0),
    }
    (seq_dir / "label_report.json").write_text(json.dumps(report, indent=2))
    logger.info("%s", report)
    return report
