"""TartanDrive labels from the vehicle's INS, with validity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apairo_huggingface import config, label_sequence
from apairo_huggingface.cli import label_main
from apairo_huggingface.reference_poses import load_validity, write_ins_poses
from conftest import BASE, DRIVE_STEP, N, STEP


@pytest.fixture
def seq(tartan_root_raw):
    return Path(tartan_root_raw) / "forest_00"


def _poses(seq):
    return np.stack([np.load(f) for f in sorted((seq / config.POSES).glob("*.npy"))])


def _valid(seq):
    return np.array([bool(np.load(f)) for f in sorted((seq / config.POSE_VALID).glob("*.npy"))])


def _hole(seq, start, stop):
    """Remove the INS rows between two times, as a dropout would."""
    odom = seq / config.STATE
    rows = np.load(odom / "odometry.npy")
    t = np.loadtxt(odom / "timestamps.txt")
    keep = (t < start) | (t > stop)
    np.save(odom / "odometry.npy", rows[keep])
    (odom / "timestamps.txt").write_text("".join(f"{x:.6f}\n" for x in t[keep]))


def test_ins_poses_are_the_lidar_poses_on_its_clock(seq):
    # The fixture drives DRIVE_STEP per scan along +x. Relative to the first
    # scan, the clock offset shifts every lookup by the same amount, so scan i
    # sits i * DRIVE_STEP ahead. Stamps are written to the microsecond around
    # 1.7e9 s: at 10 m/s that is a few micrometres.
    assert write_ins_poses(seq) is True
    poses, valid = _poses(seq), _valid(seq)
    assert poses.shape == (N, 4, 4) and valid.all()
    np.testing.assert_allclose(poses[:, 0, 3], np.arange(N) * DRIVE_STEP, atol=1e-5)
    np.testing.assert_allclose(poses[0], np.eye(4), atol=1e-12)
    assert write_ins_poses(seq) is False  # already there


def test_a_hole_in_the_ins_leaves_its_scans_without_a_pose(seq):
    # 0.12 s without INS rows around scan 3's lookup: the rows bracketing it
    # are 0.14 s apart, more than INS_MAX_GAP. No pose, no label, an unknown
    # mask; the other scans keep theirs.
    t3 = BASE + 3 * STEP + config.INS_CLOCK_OFFSET
    _hole(seq, t3 - 0.06, t3 + 0.06)
    label_sequence(seq)
    valid = _valid(seq)
    assert np.flatnonzero(~valid).tolist() == [3]
    assert np.isnan(_poses(seq)[3]).all()

    labels = np.load(seq / config.TRAV_TRAJ / "000003.npy")
    assert (labels == config.UNKNOWN).all()
    mask = np.asarray(Image.open(seq / config.TRAV_MASK / "000003.png"))
    assert (mask == config.UNKNOWN).all()  # no pixel comes from a scan without a pose


def test_the_look_ahead_stops_at_a_scan_without_a_pose(seq):
    # Scan 1 has no pose: scan 0 sees no future across it, so none of its
    # points is traversable, while scan 2 still labels the path ahead.
    t1 = BASE + 1 * STEP + config.INS_CLOCK_OFFSET
    _hole(seq, t1 - 0.06, t1 + 0.06)
    label_sequence(seq)
    assert np.flatnonzero(~_valid(seq)).tolist() == [1]
    assert not np.load(seq / config.TRAV_TRAJ / "000000.npy").any()
    assert (np.load(seq / config.TRAV_TRAJ / "000002.npy") == 1).any()


def test_a_source_without_a_mask_is_still_bounded_by_the_top_speed(seq):
    # gicp_poses of the fixture has no valid_mask.npy: only the physical bound
    # applies, and a 100 m jump fails it.
    poses = np.load(seq / "gicp_poses" / "poses.npy")
    poses[3, 0, 3] += 100.0
    times = np.loadtxt(seq / "gicp_poses" / "timestamps.txt")
    valid = load_validity(seq, "gicp_poses", poses, times)
    assert np.flatnonzero(~valid).tolist() == [2, 3, 4]


def test_cli_labels_with_the_chosen_source(seq):
    label_main(["--root", str(seq), "--poses", "gicp"])
    assert not (seq / config.POSES).exists()  # the INS was not used
    assert (np.load(seq / config.TRAV_TRAJ / "000000.npy") == 1).any()
