"""The pose diagnostic measures what it claims, on synthetic trajectories."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from apairo_huggingface.diagnostics.poses import Trajectory, compare, load_rows, path_ahead


def _drive(n=600, step=0.5, yaw_bias=0.0):
    """A curving drive, one pose per scan; ``yaw_bias`` tilts each pose's
    heading without moving its position (a heading error, as an INS has)."""
    yaw = 0.3 * np.sin(np.arange(n) / 60.0)
    xy = np.cumsum(np.c_[np.cos(yaw), np.sin(yaw)] * step, axis=0)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_euler("z", (yaw + yaw_bias).reshape(-1, 1)).as_matrix()
    poses[:, :2, 3] = xy
    return poses


def test_identical_sources_agree_exactly():
    p = _drive()
    c = compare(Trajectory("a", p), Trajectory("b", p.copy()))
    assert len(c.lateral) > 10
    assert np.abs(c.lateral).max() < 1e-9 and np.abs(c.vertical).max() < 1e-9
    assert c.summary()["per_distance"]["40"]["agreeing_share"] == 1.0


def test_a_heading_error_shows_as_a_lateral_offset_growing_with_distance():
    # Same positions, headings off by 1 degree: the path d metres ahead is
    # seen d * sin(1 deg) to the side (to first order on a gentle curve).
    a = Trajectory("truth", _drive())
    b = Trajectory("biased", _drive(yaw_bias=np.radians(1.0)))
    c = compare(a, b)
    for k, d in enumerate(c.distances):
        np.testing.assert_allclose(np.median(c.lateral[:, k]), d * np.sin(np.radians(1.0)), rtol=0.02)
    assert np.abs(c.vertical).max() < 1e-9


def test_the_path_ahead_stops_at_a_scan_without_a_pose():
    # 0.5 m per scan, scan 30 without a pose: from scan 0 there are under
    # 15 m of path before the hole, enough for 10 m, not for 20 m.
    p = _drive()
    p[30] = np.nan
    assert path_ahead(p, 0, (10.0,)) is not None
    assert path_ahead(p, 0, (20.0,)) is None
    assert path_ahead(p, 40, (20.0,)) is not None


def test_delivered_rows_get_their_clock_offset_back(tmp_path):
    # Rows recorded 0.05 s late: looked up against the scans' own trajectory,
    # the estimated clock offset is -0.05 s.
    scans = np.arange(0, 60, 0.1)
    yaw = 0.3 * np.sin(scans / 6.0)
    ref = np.tile(np.eye(4), (len(scans), 1, 1))
    ref[:, :3, :3] = Rotation.from_euler("z", yaw.reshape(-1, 1)).as_matrix()
    t = np.arange(-1, 61, 0.02)
    q = Rotation.from_euler("z", (0.3 * np.sin((t + 0.05) / 6.0)).reshape(-1, 1)).as_quat()
    rows = np.c_[t, np.zeros_like(t), np.zeros_like(t), q]   # x = t (any motion)
    np.save(tmp_path / "odometry.npy", rows)
    (tmp_path / "timestamps.txt").write_text("".join(f"{x:.6f}\n" for x in t))
    got = load_rows(tmp_path, scans, against=Trajectory("ref", ref))
    assert got.clock_offset == -0.05
