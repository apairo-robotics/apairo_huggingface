"""A tiny synthetic dataset in TartanDrive's on-disk layout, so the export can be
exercised end to end without the (large, undistributed) real recordings."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

H, W, N = 48, 64, 6           # small, even dimensions (video encoder friendly)
BASE, STEP = 1_700_000_000.0, 0.1


def _timestamps(path, n):
    path.write_text("".join(f"{BASE + i * STEP:.9e}\n" for i in range(n)))


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


@pytest.fixture
def tartan_root(tmp_path):
    """A TartanDrive root with two recordings (=> two episodes)."""
    for ep in ("forest_00", "forest_01"):
        _write_sequence(tmp_path / ep)
    return str(tmp_path)


@pytest.fixture
def n_episodes():
    return 2


@pytest.fixture
def n_frames():
    return 2 * N
