"""A tiny synthetic dataset in TartanDrive's on-disk layout, so the labeling
and the export can be exercised end to end without the (large, undistributed)
real recordings.

The lidar/pose geometry is consistent with the *real* rig calibration baked in
:mod:`apairo_huggingface.calibration`: the robot drives along +x with the
ground plane 1.77 m below the velodyne, so the trajectory labeler and the
corridor rasterizer produce non-trivial labels in the 1024x544 left camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apairo_huggingface import label_sequence

H, W, N = 48, 64, 6           # small, even dimensions (video encoder friendly)
BASE, STEP = 1_700_000_000.0, 0.1
DRIVE_STEP = 1.0              # metres between consecutive poses (along +x)
GROUND_Z = -1.77              # ground height below the velodyne


def _timestamps(path, n):
    # Fixed-point: "%.9e" kept 10 significant digits, 1 s at BASE, so every
    # frame got the same stamp and every channel synced to its last row.
    path.write_text("".join(f"{BASE + i * STEP:.6f}\n" for i in range(n)))


def _scan():
    """Sensor-frame ground grid ahead of the robot, plus a small obstacle."""
    gx, gy = np.meshgrid(np.arange(1.0, 15.0, 0.5), np.arange(-3.0, 3.0, 0.25))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, GROUND_Z)])
    obstacle = np.column_stack(
        [np.full(20, 6.0), np.full(20, 2.0), np.linspace(GROUND_Z, 0.5, 20)]
    )
    return np.vstack([ground, obstacle]).astype(np.float32)


def _write_sequence(seq):
    img = seq / "image_left_color"
    img.mkdir(parents=True)
    for i in range(N):
        Image.fromarray(np.full((H, W, 3), (i * 30) % 255, dtype=np.uint8)).save(img / f"{i:06d}.png")
    _timestamps(img / "timestamps.txt", N)

    odom = seq / "gps_odom"
    odom.mkdir()
    rows = np.zeros((N, 13), dtype=np.float64)
    rows[:, 0] = np.arange(N)     # x advances
    rows[:, 6] = 1.0              # quaternion w = 1
    np.save(odom / "odometry.npy", rows)
    _timestamps(odom / "timestamps.txt", N)

    cmd = seq / "cmd"
    cmd.mkdir()
    twist = np.column_stack([np.full(N, 1.5), np.linspace(-0.3, 0.3, N)]).astype(np.float64)
    np.save(cmd / "twist.npy", twist)
    _timestamps(cmd / "timestamps.txt", N)

    lidar = seq / "velodyne_0"
    lidar.mkdir()
    for i in range(N):
        np.save(lidar / f"{i:06d}.npy", _scan())
    _timestamps(lidar / "timestamps.txt", N)

    poses_dir = seq / "gicp_poses"
    poses_dir.mkdir()
    poses = np.tile(np.eye(4), (N, 1, 1))
    poses[:, 0, 3] = np.arange(N) * DRIVE_STEP
    np.save(poses_dir / "poses.npy", poses)
    _timestamps(poses_dir / "timestamps.txt", N)


GT_INDICES = (0, 2)  # a hand-labeled subset -- ground_truth is sparse by nature


def _write_ground_truth(seq, indices=GT_INDICES):
    """A synthetic toaster import: sparse per-point 0/1/2 labels on a subset
    of scans, row-aligned with ``velodyne_0`` -- matches
    apairo_experiments/scripts/import_toaster_gt.py's on-disk convention
    (including the ``ground_truth`` channel registration it performs)."""
    from apairo import TartanKittiDataset

    # Bootstrap .apairo/channels.yaml from the raw layout *before* registering
    # ground_truth -- once the sidecar exists, apairo stops auto-discovering
    # channels from disk, same ordering apairo_experiments' toaster importer
    # relies on (it constructs a RawDataset before it registers anything).
    TartanKittiDataset(seq, keys=["velodyne_0"])

    lidar_ts = np.loadtxt(seq / "velodyne_0" / "timestamps.txt").reshape(-1)
    gt = seq / "ground_truth"
    gt.mkdir()
    stamps = []
    for i in indices:
        scan = np.load(seq / "velodyne_0" / f"{i:06d}.npy")
        ground = scan[:, 2] == np.float32(GROUND_Z)
        labels = np.where(ground, 1, 2).astype(np.int32)  # ground traversable, obstacle not
        labels[0] = 0  # one point left unlabeled, like a real partial pass
        np.save(gt / f"{i:06d}.npy", labels)
        stamps.append(lidar_ts[i])
    np.savetxt(gt / "timestamps.txt", np.asarray(stamps))
    TartanKittiDataset.register_channel(
        seq, "ground_truth", "npys", timestamps_from="velodyne_0", sources=["velodyne_0"]
    )


@pytest.fixture
def tartan_root_raw(tmp_path):
    """A TartanDrive root with two raw (unlabeled) recordings."""
    for ep in ("forest_00", "forest_01"):
        _write_sequence(tmp_path / ep)
    return str(tmp_path)


@pytest.fixture
def tartan_root_gt(tartan_root_raw):
    """The raw root, plus hand-labeled ``ground_truth`` on part of one recording."""
    _write_ground_truth(Path(tartan_root_raw) / "forest_00")
    return tartan_root_raw


@pytest.fixture
def tartan_root(tartan_root_raw):
    """The same root, labeled (=> exportable with the traversability mask)."""
    for seq in sorted(Path(tartan_root_raw).iterdir()):
        if seq.is_dir():
            label_sequence(seq)
    return tartan_root_raw


@pytest.fixture
def n_episodes():
    return 2


@pytest.fixture
def n_frames():
    return 2 * N
