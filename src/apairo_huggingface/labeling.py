"""Self-supervised traversability labeling of TartanDrive recordings.

Where the robot actually drove is free ground truth: the future trajectory,
swept over the robot footprint and projected into the camera, labels the
image pixels the platform traversed.  This module materializes three derived
channels on a raw recording, using the ``apairo_preprocess`` primitives:

  ``trav_traj``  per-point labels — points inside the future footprint
                 (:class:`TraversabilityFromTrajectory`, path-length capped)
  ``lidar_uv``   per-point pixel coordinates in the left camera
                 (:class:`LidarCameraProjection`, rig from :mod:`.calibration`)
  ``trav_mask``  (H, W) uint8 image mask — the sparse point labels
                 (:class:`ImageMaskFromPointLabels`) plus the *driven
                 corridor* rasterized as filled quads, so the labeled passage
                 is dense, not a set of lidar rings.  Corridor quads hidden
                 behind nearer lidar returns are skipped (occlusion).

Channels are persisted with ``ChannelWriter`` (the multi-channel steps cannot
run through ``run_preprocess`` on an asynchronous dataset — see apairo
``IDEAS.md``) and registered in ``.apairo``, so they load like any raw
channel and feed straight into the LeRobot export (:mod:`.export`).

Memory stays bounded: scans are processed in chunks, and the look-ahead the
labeler needs beyond a chunk is provided as poses with *empty* scans — labels
only depend on future positions, so chunked labels are exact.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from apairo import ChannelWriter, TartanKittiDataset
from apairo.core.sample import Sample
from apairo_preprocess import (
    ImageMaskFromPointLabels,
    LidarCameraProjection,
    TraversabilityFromTrajectory,
)

from . import config
from .calibration import CAMERA, IMAGE_HEIGHT, IMAGE_WIDTH, T_OPTICAL_FROM_VELODYNE

logger = logging.getLogger(__name__)

LIDAR = "velodyne_0"
_EMPTY_SCAN = np.zeros((0, 3), dtype=np.float32)


# --------------------------------------------------------------------- poses

def load_poses(seq_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 4, 4)`` poses and their timestamps from ``<seq>/<key>/``.

    Accepts the stacked layouts found in the wild (``poses.npy`` or
    ``<key>.npy``) as well as one ``.npy`` per frame.
    """
    channel = Path(seq_dir) / key
    if not channel.is_dir():
        raise FileNotFoundError(
            f"'{key}' not found in {seq_dir} — run an odometry preprocessor "
            f"first (e.g. apairo_preprocess.KissICPOdometry)."
        )
    stacked = [p for name in ("poses.npy", f"{key}.npy") if (p := channel / name).exists()]
    if stacked:
        poses = np.load(stacked[0])
    else:
        files = sorted(p for p in channel.glob("*.npy") if not p.stem.endswith("_mask"))
        poses = np.stack([np.load(p) for p in files])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N, 4, 4), got {poses.shape}")
    timestamps = np.loadtxt(channel / "timestamps.txt")
    return poses, timestamps


# ------------------------------------------------------------------ corridor

def corridor_edges(traj, j, max_path, half_width, ground_offset, step_l=0.3):
    """Left/right edges of the future footprint from frame j, world frame.

    Future positions are interpolated linearly along arc length (poses at
    ~10 Hz are far denser than the footprint width, so the corridor follows
    curves smoothly) and offset laterally by ``half_width`` at ground height
    (pose height + ``ground_offset``), for ``max_path`` metres of travel —
    the same cap as ``forward_distance`` in the point labeler.
    """
    empty = np.empty((0, 3))
    pos = traj[j:]
    if len(pos) < 2:
        return empty, empty
    seg = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    pos = pos[np.concatenate([[True], seg > 1e-3])]  # drop stationary duplicates
    if len(pos) < 2:
        return empty, empty
    seg = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])
    end = int(np.searchsorted(path, max_path, side="right"))
    if end < 2:
        return empty, empty
    pos, path = pos[:end], path[:end]

    s = np.arange(0.0, path[-1], step_l)
    x = np.interp(s, path, pos[:, 0])
    y = np.interp(s, path, pos[:, 1])
    z = np.interp(s, path, pos[:, 2]) + ground_offset
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    lx, ly = -dy / norm, dx / norm  # unit left vector per sample
    left = np.column_stack([x + half_width * lx, y + half_width * ly, z])
    right = np.column_stack([x - half_width * lx, y - half_width * ly, z])
    return left, right


def corridor_mask(traj, j, pose, uv_lidar, *, max_path, half_width, ground_offset,
                  occlusion_bin_px=16, occlusion_margin=1.0) -> np.ndarray:
    """(H, W) bool — the future footprint corridor, rasterized as filled quads.

    Edges are projected *unclipped* so quads reaching out of the frame still
    fill their visible part.  A quad is skipped when the nearest lidar return
    around its vertices is more than ``occlusion_margin`` metres in front of
    it: corridor hidden behind an obstacle stays unlabeled.
    """
    from PIL import Image, ImageDraw

    height, width = IMAGE_HEIGHT, IMAGE_WIDTH
    left_w, right_w = corridor_edges(traj, j, max_path, half_width, ground_offset)
    if len(left_w) < 2:
        return np.zeros((height, width), dtype=bool)

    def project_unclipped(pts_world):
        pts = (pts_world - pose[:3, 3]) @ pose[:3, :3]  # world -> sensor frame
        cam = pts @ T_OPTICAL_FROM_VELODYNE[:3, :3].T + T_OPTICAL_FROM_VELODYNE[:3, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = CAMERA.K[0, 0] * cam[:, 0] / z + CAMERA.K[0, 2]
            v = CAMERA.K[1, 1] * cam[:, 1] / z + CAMERA.K[1, 2]
        return u, v, z

    ul, vl, zl = project_unclipped(left_w)
    ur, vr, zr = project_unclipped(right_w)

    # Coarse nearest-return grid from the lidar projection, for occlusion.
    bins_x = (width + occlusion_bin_px - 1) // occlusion_bin_px
    bins_y = (height + occlusion_bin_px - 1) // occlusion_bin_px
    nearest = np.full(bins_y * bins_x, np.inf)
    lv = np.isfinite(uv_lidar[:, 0])
    bin_id = (
        (uv_lidar[lv, 1].astype(int) // occlusion_bin_px) * bins_x
        + uv_lidar[lv, 0].astype(int) // occlusion_bin_px
    )
    np.minimum.at(nearest, bin_id, uv_lidar[lv, 2].astype(float))
    nearest = nearest.reshape(bins_y, bins_x)

    def lidar_depth_at(u, v):
        bx = np.clip(int(u) // occlusion_bin_px, 0, bins_x - 1)
        by = np.clip(int(v) // occlusion_bin_px, 0, bins_y - 1)
        return nearest[by, bx]

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for i in range(len(ul) - 1):
        zs = (zl[i], zr[i], zl[i + 1], zr[i + 1])
        if min(zs) < 0.5:  # behind or grazing the camera
            continue
        quad = [(ul[i], vl[i]), (ur[i], vr[i]), (ur[i + 1], vr[i + 1]), (ul[i + 1], vl[i + 1])]
        if min(lidar_depth_at(u, v) for u, v in quad) + occlusion_margin < min(zs):
            continue  # hidden behind a nearer return
        draw.polygon(quad, fill=1)
    return np.asarray(canvas, dtype=bool)


# ------------------------------------------------------------------ pipeline

def label_sequence(seq_dir, *, poses_key: str = config.POSES, overwrite: bool = False,
                   chunk: int = 256) -> bool:
    """Materialize ``trav_traj``, ``lidar_uv`` and ``trav_mask`` on one recording.

    Returns ``True`` when channels were written, ``False`` when they already
    existed and ``overwrite`` was not set.  Tunables live in :mod:`.config`.
    """
    seq_dir = Path(seq_dir)
    channels = (config.TRAV_TRAJ, config.LIDAR_UV, config.TRAV_MASK)
    existing = [c for c in channels if (seq_dir / c).exists()]
    if existing and not overwrite:
        logger.info("%s: already labeled, skipping (overwrite=True to redo)", seq_dir.name)
        return False
    for c in existing:
        TartanKittiDataset.remove_channel(seq_dir, c, data=True)

    ds = TartanKittiDataset(seq_dir, keys=[LIDAR])
    lidar_ts = np.asarray(ds.timestamps[LIDAR])
    poses, poses_ts = load_poses(seq_dir, poses_key)
    n = min(len(lidar_ts), len(poses))
    if np.abs(poses_ts[:n] - lidar_ts[:n]).max() > 0.02:
        raise ValueError(
            f"'{poses_key}' is not on the lidar clock (max skew "
            f"{np.abs(poses_ts[:n] - lidar_ts[:n]).max():.3f}s); resample it first."
        )
    traj = poses[:, :3, 3]
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])

    labeler = TraversabilityFromTrajectory(
        robot_radius=config.ROBOT_HALF_WIDTH,
        height_min=config.HEIGHT_MIN,
        height_max=config.HEIGHT_MAX,
        forward_distance=config.FORWARD_DISTANCE,
    )
    projector = LidarCameraProjection(
        intrinsics=CAMERA, extrinsics=T_OPTICAL_FROM_VELODYNE, lidar_key=LIDAR,
    )
    masker = ImageMaskFromPointLabels(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        radius=config.SPLAT_RADIUS,
        occlusion_bin_px=config.OCCLUSION_BIN_PX,
    )

    writer_args = dict(loader="npys", timestamps_from=LIDAR)
    with (
        ChannelWriter(seq_dir, config.TRAV_TRAJ, sources=[LIDAR, poses_key], **writer_args) as w_traj,
        ChannelWriter(seq_dir, config.LIDAR_UV, sources=[LIDAR], **writer_args) as w_uv,
        ChannelWriter(seq_dir, config.TRAV_MASK,
                      sources=[config.TRAV_TRAJ, config.LIDAR_UV, poses_key], **writer_args) as w_mask,
    ):
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            # Labels depend on future *poses* only: extend the chunk with a
            # pose-only tail (empty scans) covering the look-ahead cap, so
            # chunked labels equal whole-sequence labels at bounded memory.
            tail = min(n, int(np.searchsorted(path, path[b - 1] + config.FORWARD_DISTANCE)) + 1)
            scans = [np.asarray(ds.load(LIDAR, i)) for i in range(a, b)]
            frames = [
                Sample(data={"lidar": s, "poses": poses[i]})
                for i, s in zip(range(a, b), scans)
            ] + [
                Sample(data={"lidar": _EMPTY_SCAN, "poses": poses[i]})
                for i in range(b, tail)
            ]
            labels = labeler(iter(frames))[: b - a]

            for j, i in enumerate(range(a, b)):
                uv = projector(Sample(data={LIDAR: scans[j]}))
                mask = masker(Sample(data={"trav_traj": labels[j], "lidar_uv": uv}))
                # The driven corridor on top: the robot passed there —
                # strongest traversability evidence, denser than any ring.
                mask[corridor_mask(
                    traj, i, poses[i], uv,
                    max_path=config.FORWARD_DISTANCE,
                    half_width=config.ROBOT_HALF_WIDTH,
                    ground_offset=config.GROUND_OFFSET,
                )] = 1
                stem = f"{i:06d}"
                ts = float(lidar_ts[i])
                w_traj.add(np.asarray(labels[j], dtype=np.uint8), stem=stem, timestamp=ts)
                w_uv.add(uv, stem=stem, timestamp=ts)
                w_mask.add(mask, stem=stem, timestamp=ts)
            logger.debug("%s: [%d, %d) / %d", seq_dir.name, a, b, n)
    logger.info("%s: labeled (%d frames)", seq_dir.name, n)
    return True


def label_root(root, *, overwrite: bool = False) -> None:
    """Label every recording under ``root`` (or ``root`` itself, if a sequence)."""
    probe = TartanKittiDataset(root, keys=[LIDAR])
    if getattr(probe, "_is_root", False):
        seq_dirs = [s.root_dir for s in probe.sequences]
    else:
        seq_dirs = [Path(root)]
    for seq_dir in seq_dirs:
        label_sequence(seq_dir, overwrite=overwrite)
