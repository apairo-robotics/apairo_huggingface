"""Self-supervised traversability labels for FoMo trajectories.

Where the robot drove is free ground truth. On FoMo the camera and the lidar
run on their own triggers, so unlike the TartanDrive pipeline the image mask
is computed **at the camera instant**, not borrowed from the nearest scan:

1. ``odometry``    KISS-ICP (deskewed with per-point time) -> ``kissicp_poses``,
                   one ``T_world_lidar`` per scan.
2. ``label_scans`` per-point labels inside the future footprint ->
                   ``trav_traj``, and inside the past footprint ->
                   ``trav_traj_past``, on the lidar clock.
3. ``label_images`` for each camera frame, ``synchronize()`` picks the nearest
                   scan (and its labels) and *interpolates* the pose at the
                   frame's own timestamp. The scan is moved by the motion
                   between its stamp and the frame's (``inv(T_cam) @ T_scan``),
                   projected through the camera's distorted model, splatted,
                   and the footprint ribbon is drawn on top -> one mask channel
                   per camera (``config.TRAV_MASKS``), on the camera clock.
                   The frame stamps come from the bucket listing, so no image
                   is needed. Forward ZED X cameras get the future footprint;
                   the Basler looks backward and gets the past one.

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
from .calibration import FOOTPRINT, IMAGE_HEIGHT, IMAGE_WIDTH, INTRINSICS, camera_from_lidar

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

def label_scans(traj_dir, *, footprint: str = "future", overwrite: bool = False,
                chunk: int = 128) -> None:
    """Per-point footprint labels, one file per scan.

    ``footprint="future"`` -> ``trav_traj``: points under the footprint swept
    along the trajectory *ahead* of each scan. ``"past"`` -> ``trav_traj_past``:
    the same along the trajectory *behind* it (the labeler run on the scans
    in reverse order), for the rear-facing camera.
    """
    traj_dir = Path(traj_dir)
    channel = config.POINT_LABELS[footprint]
    if (traj_dir / channel / "timestamps.txt").exists() and not overwrite:
        logger.info("%s: %s exists, skipping", traj_dir.name, channel)
        return
    _drop(traj_dir, channel)

    ds = open_trajectory(traj_dir, [config.LIDAR, config.POSES])
    poses, poses_ts = load_poses(ds)
    lidar_ts = np.asarray(ds.timestamps[config.LIDAR], dtype=np.float64)
    if len(poses) != len(lidar_ts) or np.abs(poses_ts - lidar_ts).max() > 1e-3:
        raise ValueError(f"{config.POSES} is not on the {config.LIDAR} clock")
    n = len(lidar_ts)
    # Processing order: time order for the future footprint, reversed for the past.
    order = np.arange(n) if footprint == "future" else np.arange(n)[::-1]
    traj = poses[order, :3, 3]
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
    labeler = TraversabilityFromTrajectory(
        robot_radius=config.ROBOT_HALF_WIDTH,
        height_min=config.HEIGHT_MIN,
        height_max=config.HEIGHT_MAX,
        forward_distance=config.FORWARD_DISTANCE,
    )
    lidar = ds.loaders[config.LIDAR]
    tmp = traj_dir / f"{channel}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        # Labels depend on the poses ahead (in processing order) only: a
        # pose-only tail covering the look-ahead makes chunked labels equal
        # whole-sequence ones.
        tail = min(n, int(np.searchsorted(path, path[b - 1] + config.FORWARD_DISTANCE)) + 1)
        frames = [Sample(data={"lidar": np.asarray(lidar[order[r]])[:, :3], "poses": poses[order[r]]})
                  for r in range(a, b)]
        frames += [Sample(data={"lidar": _EMPTY_SCAN, "poses": poses[order[r]]}) for r in range(b, tail)]
        labels = labeler(iter(frames))[: b - a]
        for r, lab in zip(range(a, b), labels):
            np.save(tmp / f"{order[r]:06d}.npy", np.asarray(lab, dtype=np.uint8))
        logger.info("%s: %s scans [%d, %d) / %d", traj_dir.name, channel, a, b, n)
    # Written in processing order, registered in time order.
    with ChannelWriter(
        traj_dir, channel, loader="npys",
        timestamps_from=config.LIDAR, sources=[config.LIDAR, config.POSES],
    ) as writer:
        for i in range(n):
            writer.add(np.load(tmp / f"{i:06d}.npy"), stem=f"{i:06d}", timestamp=float(lidar_ts[i]))
    shutil.rmtree(tmp)


def check_scans(traj_dir) -> int:
    """Check that the raw scans on disk are the ones ``trav_traj`` labelled.

    Same stamps (file names vs ``trav_traj/timestamps.txt``) and, scan by scan,
    as many records as labels (file size / record size vs label length).
    Returns the number of scans; raises ``ValueError`` on any mismatch.
    """
    from .format import RECORD

    traj_dir = Path(traj_dir)
    files = sorted((traj_dir / config.LIDAR).glob("*.bin"))
    labels_dir = traj_dir / config.TRAV_TRAJ
    stamps = np.round(np.loadtxt(labels_dir / "timestamps.txt", ndmin=1) * 1e6).astype(np.int64)
    file_stamps = np.array([int(f.stem) for f in files], dtype=np.int64)
    if len(file_stamps) != len(stamps) or not np.array_equal(np.sort(file_stamps), stamps):
        raise ValueError(f"{traj_dir.name}: {len(files)} scans on disk, {len(stamps)} labelled, "
                         "stamps differ")
    for i, f in enumerate(sorted(files, key=lambda f: int(f.stem))):
        size = f.stat().st_size
        n_labels = np.load(labels_dir / f"{i:06d}.npy", mmap_mode="r").shape[0]
        if size % RECORD.itemsize or size // RECORD.itemsize != n_labels:
            raise ValueError(f"{traj_dir.name}: {f.name} holds {size / RECORD.itemsize:g} points, "
                             f"{n_labels} labels")
    return len(files)


# ------------------------------------------------------------ image labels

def _project_unclipped(points_cam: np.ndarray, intrinsics=INTRINSICS[config.CAMERA]):
    """Pixel coordinates of camera-frame points through the distorted model."""
    z = points_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = points_cam[:, 0] / z, points_cam[:, 1] / z
    k1, k2, p1, p2, k3 = intrinsics.distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = intrinsics.K
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


def corridor_mask(left_w, right_w, cam_from_world, points_world, *, margin=0.5, above=0.3,
                  intrinsics=INTRINSICS[config.CAMERA]):
    """(H, W) bool: the corridor as filled quads, minus the quads an obstacle hides.

    A quad is hidden only by a lidar return that projects inside it, lies more
    than ``margin`` metres in front of it and more than ``above`` metres above
    the corridor surface there. Ground returns never hide the ground: at
    grazing angles a nearer ground return shares the pixels of a farther quad.
    Far quads are drawn first, so where two quads overlap in the image the
    nearer one owns the pixels.
    """
    empty = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    if len(left_w) < 2:
        return empty

    def to_cam(p):
        return p @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]

    ul, vl, zl = _project_unclipped(to_cam(left_w), intrinsics)
    ur, vr, zr = _project_unclipped(to_cam(right_w), intrinsics)

    def quad(i):
        return [(ul[i], vl[i]), (ur[i], vr[i]), (ur[i + 1], vr[i + 1]), (ul[i + 1], vl[i + 1])]

    quads = [
        i for i in range(len(ul) - 1)
        if min(zl[i], zr[i], zl[i + 1], zr[i + 1]) >= 0.5
        and all(np.isfinite(c) for q in quad(i) for c in q)
    ]
    if not quads:
        return empty
    ids_img = Image.new("I", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(ids_img)
    for i in sorted(quads, key=lambda i: -(zl[i] + zr[i])):
        draw.polygon(quad(i), fill=i + 1)
    ids = np.asarray(ids_img, dtype=np.int64) - 1

    u, v, z = _project_unclipped(to_cam(points_world), intrinsics)
    inside = (z > 0.5) & (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    q = ids[v[inside].astype(int), u[inside].astype(int)]
    hit = q >= 0
    q, z_pt, h_pt = q[hit], z[inside][hit], points_world[inside][hit, 2]
    depth = 0.25 * (zl[q] + zr[q] + zl[q + 1] + zr[q + 1])
    ground = 0.25 * (left_w[q, 2] + right_w[q, 2] + left_w[q + 1, 2] + right_w[q + 1, 2])
    hidden = np.unique(q[(z_pt < depth - margin) & (h_pt > ground + above)])
    return (ids >= 0) & ~np.isin(ids, hidden)


def camera_stamps(traj: str, camera: str) -> np.ndarray:
    """Sorted microsecond stamps of a camera's raw images, from the bucket listing.

    The masks need the camera geometry and its frame instants, not the pixels.
    """
    from .fetch import list_keys

    keys = list_keys(f"data/{traj}/{camera}/")
    return np.array(sorted(int(Path(k).stem) for k, _ in keys if k.endswith(".png")), dtype=np.int64)


def label_images(traj_dir, *, camera: str = config.CAMERA, stamps_us=None,
                 overwrite: bool = False, limit: int | None = None) -> int:
    """``config.TRAV_MASKS[camera]``: one PNG per camera frame with a scan within tolerance.

    ``stamps_us`` are the camera's frame stamps (e.g. :func:`camera_stamps`);
    by default they are read from the camera images on disk. Each mask is
    named by the raw image's microsecond stamp, so it joins with FoMo's own
    ``<camera>/<us>.png``. The footprint is the one the camera can see
    (``calibration.FOOTPRINT``): future for the forward ZED X, past for the
    rear Basler.
    """
    traj_dir = Path(traj_dir)
    channel = config.TRAV_MASKS[camera]
    footprint = FOOTPRINT[camera]
    point_labels = config.POINT_LABELS[footprint]
    intrinsics = INTRINSICS[camera]
    out_dir = traj_dir / channel
    if (out_dir / "timestamps.txt").exists() and not overwrite:
        logger.info("%s: %s exists, skipping", traj_dir.name, channel)
        return 0
    _drop(traj_dir, channel)

    keys = [config.LIDAR, point_labels, config.POSES]
    if stamps_us is None:
        ds = open_trajectory(traj_dir, [camera, *keys])
        stamps_us = np.round(np.asarray(ds.timestamps[camera], dtype=np.float64) * 1e6).astype(np.int64)
    else:
        ds = open_trajectory(traj_dir, keys)
        stamps_us = np.asarray(stamps_us, dtype=np.int64)
    poses, poses_ts = load_poses(ds)
    # The camera clock is the reference: each frame gets the nearest scan and
    # its labels, and the pose *at the frame's own instant*.
    sync = ds.synchronize(
        reference=stamps_us * 1e-6,
        method={
            config.LIDAR: "nearest",
            point_labels: "nearest",
            config.POSES: Se3Interp(),
        },
        # One scalar for every channel: the 10 Hz poses bracket a frame with
        # events up to a period away, so the tight scan check is done below.
        tolerance=config.POSE_TOLERANCE,
    )
    scan_of = np.asarray(sync.frame_indices[config.LIDAR])
    fresh = np.abs(sync.time_offsets(config.LIDAR)) <= config.TOLERANCE
    E = camera_from_lidar(traj_dir, camera)
    projector = LidarCameraProjection(intrinsics=intrinsics, extrinsics=E, lidar_key="lidar")
    masker = ImageMaskFromPointLabels(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        radius=config.SPLAT_RADIUS,
        occlusion_bin_px=config.OCCLUSION_BIN_PX,
    )
    traj = poses[:, :3, 3]
    tmp = traj_dir / f"{channel}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    stamps = []
    n = len(sync) if limit is None else min(limit, len(sync))
    for k in range(n):
        if not fresh[k]:
            continue
        sample = sync[k]
        t_cam = float(sample.timestamp)
        us = int(stamps_us[np.argmin(np.abs(stamps_us - round(t_cam * 1e6)))])
        j = int(scan_of[k])
        T_cam = np.asarray(sample.data[config.POSES], dtype=np.float64)  # world <- lidar @ t_cam
        motion = np.linalg.inv(T_cam) @ poses[j]                         # lidar@t_cam <- lidar@t_scan
        pts = np.asarray(sample.data[config.LIDAR])[:, :3].astype(np.float64)
        pts = pts @ motion[:3, :3].T + motion[:3, 3]
        uv = projector(Sample(data={"lidar": pts.astype(np.float32)}))
        mask = masker(Sample(data={"trav_traj": sample.data[point_labels], "lidar_uv": uv}))
        # Lidar returns only say "seen"; "driven" is the geometric corridor, so
        # ring splats never stick out of it.
        mask[mask == 1] = 0
        now = int(np.searchsorted(poses_ts, t_cam, side="right"))
        path = traj[now:] if footprint == "future" else traj[:now][::-1]
        left, right = corridor_world(path, T_cam[:3, 3])
        pts_world = pts @ T_cam[:3, :3].T + T_cam[:3, 3]
        mask[corridor_mask(left, right, E @ np.linalg.inv(T_cam), pts_world, intrinsics=intrinsics)] = 1
        Image.fromarray(mask, mode="L").save(tmp / f"{us}.png")
        stamps.append(us)
        if k % 500 == 0:
            logger.info("%s: %s frames %d / %d", traj_dir.name, channel, k, n)
    np.savetxt(tmp / "timestamps.txt", np.asarray(stamps) * 1e-6, fmt="%.6f")
    tmp.rename(out_dir)
    RawDataset.register_channel(
        traj_dir, channel, "img",
        sources=[point_labels, config.POSES, config.LIDAR],
    )
    logger.info("%s: %d %s masks (%s footprint)", traj_dir.name, len(stamps), camera, footprint)
    return len(stamps)
