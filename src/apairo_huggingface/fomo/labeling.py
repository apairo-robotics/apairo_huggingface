"""Self-supervised traversability labels for FoMo trajectories.

Where the robot drove is free ground truth. On FoMo the camera and the lidar
run on their own triggers, so unlike the TartanDrive pipeline the image mask
is computed **at the camera instant**, not borrowed from the nearest scan:

1. ``odometry``    KISS-ICP (deskewed with per-point time) -> ``kissicp_poses``,
                   one ``T_world_lidar`` per scan.
2. ``label_scans`` per-point labels inside the future footprint ->
                   ``trav_traj``, on the lidar clock.
3. ``label_images`` for each Basler frame, ``synchronize()`` picks the nearest
                   scan (and its labels) and *interpolates* the pose at the
                   frame's own timestamp. The scan is moved by the motion
                   between its stamp and the frame's (``inv(T_cam) @ T_scan``),
                   projected through the distorted Basler model, splatted, and
                   the driven corridor is drawn on top -> ``trav_mask``, on the
                   camera clock.

The FoMo tree is read through a declaration passed at load time; only the
derived channels and apairo's ``.apairo/`` registry are written into it.
"""

from __future__ import annotations

import logging
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from apairo import ChannelWriter, RawDataset
from apairo.core.sample import Sample
from apairo_preprocess import (
    ImageMaskFromPointLabels,
    KissICPOdometry,
    LidarCameraProjection,
    TraversabilityFromTrajectory,
)
from apairo_transform.interp import Se3Interp

from . import config
from .calibration import BASLER, IMAGE_HEIGHT, IMAGE_WIDTH, camera_from_lidar

logger = logging.getLogger(__name__)

_EMPTY_SCAN = np.zeros((0, 3), dtype=np.float32)


def declaration() -> Path:
    """The packaged ``apairo.yaml`` describing one FoMo trajectory."""
    return Path(str(resources.files(__package__) / "apairo.yaml"))


def open_trajectory(traj_dir, keys) -> RawDataset:
    return RawDataset(traj_dir, declare=declaration(), keys=list(keys))


def _drop(traj_dir: Path, channel: str) -> None:
    try:
        RawDataset.remove_channel(traj_dir, channel, data=True)
    except KeyError:
        pass
    shutil.rmtree(traj_dir / channel, ignore_errors=True)


# ------------------------------------------------------------------ odometry

def odometry(traj_dir, *, overwrite: bool = False) -> None:
    traj_dir = Path(traj_dir)
    if (traj_dir / config.POSES).exists() and not overwrite:
        logger.info("%s: %s exists, skipping", traj_dir.name, config.POSES)
        return
    ds = open_trajectory(traj_dir, [config.LIDAR])
    ds.run_preprocess(
        KissICPOdometry(
            lidar_key=config.LIDAR,
            max_range=config.ODOM_MAX_RANGE,
            min_range=config.ODOM_MIN_RANGE,
            voxel_size=config.ODOM_VOXEL,
            deskew=True,
        ),
        overwrite=overwrite,
        declare=declaration(),  # the instance's own declare= is not forwarded
    )


def load_poses(ds: RawDataset) -> tuple[np.ndarray, np.ndarray]:
    loader = ds.loaders[config.POSES]
    poses = np.stack([np.asarray(loader[i], dtype=np.float64) for i in range(len(loader))])
    return poses, np.asarray(ds.timestamps[config.POSES], dtype=np.float64)


# ------------------------------------------------------------- scan labels

def label_scans(traj_dir, *, overwrite: bool = False, chunk: int = 128) -> None:
    """``trav_traj``: per-point future-footprint labels, one file per scan."""
    traj_dir = Path(traj_dir)
    if (traj_dir / config.TRAV_TRAJ / "timestamps.txt").exists() and not overwrite:
        logger.info("%s: %s exists, skipping", traj_dir.name, config.TRAV_TRAJ)
        return
    _drop(traj_dir, config.TRAV_TRAJ)

    ds = open_trajectory(traj_dir, [config.LIDAR, config.POSES])
    poses, poses_ts = load_poses(ds)
    lidar_ts = np.asarray(ds.timestamps[config.LIDAR], dtype=np.float64)
    if len(poses) != len(lidar_ts) or np.abs(poses_ts - lidar_ts).max() > 1e-3:
        raise ValueError(f"{config.POSES} is not on the {config.LIDAR} clock")
    traj = poses[:, :3, 3]
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
    labeler = TraversabilityFromTrajectory(
        robot_radius=config.ROBOT_HALF_WIDTH,
        height_min=config.HEIGHT_MIN,
        height_max=config.HEIGHT_MAX,
        forward_distance=config.FORWARD_DISTANCE,
    )
    lidar = ds.loaders[config.LIDAR]
    n = len(lidar_ts)
    with ChannelWriter(
        traj_dir, config.TRAV_TRAJ, loader="npys",
        timestamps_from=config.LIDAR, sources=[config.LIDAR, config.POSES],
    ) as writer:
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            # Labels depend on future poses only: a pose-only tail covering the
            # look-ahead makes chunked labels equal whole-sequence ones.
            tail = min(n, int(np.searchsorted(path, path[b - 1] + config.FORWARD_DISTANCE)) + 1)
            scans = [np.asarray(lidar[i])[:, :3] for i in range(a, b)]
            frames = [Sample(data={"lidar": s, "poses": poses[i]}) for i, s in zip(range(a, b), scans)]
            frames += [Sample(data={"lidar": _EMPTY_SCAN, "poses": poses[i]}) for i in range(b, tail)]
            labels = labeler(iter(frames))[: b - a]
            for i, lab in zip(range(a, b), labels):
                writer.add(np.asarray(lab, dtype=np.uint8), stem=f"{i:06d}", timestamp=float(lidar_ts[i]))
            logger.info("%s: scans [%d, %d) / %d", traj_dir.name, a, b, n)


# ------------------------------------------------------------ image labels

def _project_unclipped(points_cam: np.ndarray):
    """Pixel coordinates of camera-frame points through the distorted model."""
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = points_cam[:, 0] / z, points_cam[:, 1] / z
    k1, k2, p1, p2, k3 = BASLER.distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = BASLER.K
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2], z


def corridor_world(traj: np.ndarray, start: np.ndarray, step: float = 0.3):
    """Left/right edges of the future footprint from ``start``, world frame."""
    empty = np.empty((0, 3))
    pos = np.vstack([start[None], traj])
    seg = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    pos = pos[np.concatenate([[True], seg > 1e-3])]
    if len(pos) < 2:
        return empty, empty
    seg = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])
    end = int(np.searchsorted(path, config.FORWARD_DISTANCE, side="right"))
    if end < 2:
        return empty, empty
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


def corridor_mask(left_w, right_w, cam_from_world, uv_lidar, *, bin_px=16, margin=1.0):
    """(H, W) bool: the corridor as filled quads, skipping occluded ones."""
    mask = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    if len(left_w) < 2:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)

    def to_cam(p):
        return p @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]

    ul, vl, zl = _project_unclipped(to_cam(left_w))
    ur, vr, zr = _project_unclipped(to_cam(right_w))
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


def label_images(traj_dir, *, overwrite: bool = False, limit: int | None = None) -> int:
    """``trav_mask``: one PNG per Basler frame that has a scan within tolerance."""
    traj_dir = Path(traj_dir)
    out_dir = traj_dir / config.TRAV_MASK
    if (out_dir / "timestamps.txt").exists() and not overwrite:
        logger.info("%s: %s exists, skipping", traj_dir.name, config.TRAV_MASK)
        return 0
    _drop(traj_dir, config.TRAV_MASK)

    ds = open_trajectory(traj_dir, [config.CAMERA, config.LIDAR, config.TRAV_TRAJ, config.POSES])
    poses, poses_ts = load_poses(ds)
    # The camera clock is the reference: each frame gets the nearest scan and
    # its labels, and the pose *at the frame's own instant*.
    sync = ds.synchronize(
        reference=config.CAMERA,
        method={
            config.LIDAR: "nearest",
            config.TRAV_TRAJ: "nearest",
            config.POSES: Se3Interp(),
        },
        # One scalar for every channel: the 10 Hz poses bracket a frame with
        # events up to a period away, so the tight scan check is done below.
        tolerance=config.POSE_TOLERANCE,
    )
    scan_of = np.asarray(sync.frame_indices[config.LIDAR])
    fresh = np.abs(sync.time_offsets(config.LIDAR)) <= config.TOLERANCE
    E = camera_from_lidar(traj_dir)
    projector = LidarCameraProjection(intrinsics=BASLER, extrinsics=E, lidar_key="lidar")
    masker = ImageMaskFromPointLabels(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        radius=config.SPLAT_RADIUS,
        occlusion_bin_px=config.OCCLUSION_BIN_PX,
    )
    traj = poses[:, :3, 3]
    out_dir.mkdir()
    stamps = []
    n = len(sync) if limit is None else min(limit, len(sync))
    for k in range(n):
        if not fresh[k]:
            continue
        sample = sync[k]
        t_cam = float(sample.timestamp)
        j = int(scan_of[k])
        T_cam = np.asarray(sample.data[config.POSES], dtype=np.float64)  # world <- lidar @ t_cam
        motion = np.linalg.inv(T_cam) @ poses[j]                         # lidar@t_cam <- lidar@t_scan
        pts = np.asarray(sample.data[config.LIDAR])[:, :3].astype(np.float64)
        pts = pts @ motion[:3, :3].T + motion[:3, 3]
        uv = projector(Sample(data={"lidar": pts.astype(np.float32)}))
        mask = masker(Sample(data={"trav_traj": sample.data[config.TRAV_TRAJ], "lidar_uv": uv}))
        future = traj[np.searchsorted(poses_ts, t_cam, side="right"):]
        left, right = corridor_world(future, T_cam[:3, 3])
        mask[corridor_mask(left, right, E @ np.linalg.inv(T_cam), uv)] = 1
        stem = f"{k:06d}"
        Image.fromarray(mask, mode="L").save(out_dir / f"{stem}.png")
        stamps.append(t_cam)
        if k % 200 == 0:
            logger.info("%s: frames %d / %d", traj_dir.name, k, n)
    np.savetxt(out_dir / "timestamps.txt", np.asarray(stamps), fmt="%.6f")
    RawDataset.register_channel(
        traj_dir, config.TRAV_MASK, "img",
        sources=[config.TRAV_TRAJ, config.POSES, config.LIDAR],
    )
    logger.info("%s: %d image masks", traj_dir.name, len(stamps))
    return len(stamps)
