"""TartanDrive poses from the vehicle's INS, on the lidar clock.

``gps_odom`` is the Novatel INS of the vehicle: one row per 20 ms,
``[x y z qx qy qz qw, twist]``, position in UTM (x north, y west, z up) and
the vehicle's attitude in that local level frame. The velodyne is in the
vehicle frame, so the INS pose *is* the lidar's, once looked up at the
lidar's time. That lookup is all this module does
(:func:`apairo_preprocess.reference_poses`): no estimation, no tuning.

Measured once on two recordings (turnpike_afternoon_fall, down_meadows) and
kept as constants in :mod:`.config`: the INS clock runs 0.05-0.06 s behind
the lidar's; the INS reference point and the vehicle origin give the same
relative motion to within a centimetre.

Two channels are written next to the raw ones, on the lidar clock:
``ins_poses`` (``(4, 4)`` float64, NaN where there is no pose) and
``ins_valid`` (one bool per scan).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from apairo import ChannelWriter, TartanKittiDataset
from apairo_preprocess import ReferencePoses, pose_validity

from . import config

logger = logging.getLogger(__name__)

LIDAR = "velodyne_0"


def ins() -> ReferencePoses:
    """The INS lookup with the rig's measured constants."""
    return ReferencePoses(
        clock_offset=config.INS_CLOCK_OFFSET,
        max_gap=config.INS_MAX_GAP,
        max_speed=config.MAX_SPEED,
    )


def write_ins_poses(seq_dir, *, overwrite: bool = False) -> bool:
    """Write ``ins_poses`` and ``ins_valid`` on one recording.

    Returns ``True`` when written, ``False`` when both exist and ``overwrite``
    is not set.
    """
    seq_dir = Path(seq_dir)
    out = (seq_dir / config.POSES, seq_dir / config.POSE_VALID)
    if all((d / "timestamps.txt").exists() for d in out) and not overwrite:
        return False
    for channel in (config.POSES, config.POSE_VALID):
        try:
            TartanKittiDataset.remove_channel(seq_dir, channel, data=True)
        except (KeyError, FileNotFoundError):  # not registered, or no registry yet
            pass
        shutil.rmtree(seq_dir / channel, ignore_errors=True)

    odom = seq_dir / config.STATE
    rows = np.load(odom / "odometry.npy")
    row_times = np.loadtxt(odom / "timestamps.txt")
    scan_times = np.asarray(TartanKittiDataset(seq_dir, keys=[LIDAR]).timestamps[LIDAR], dtype=np.float64)
    poses, valid = ins()(rows, row_times, scan_times)

    args = dict(loader="npys", timestamps_from=LIDAR, sources=[config.STATE, LIDAR])
    with ChannelWriter(seq_dir, config.POSES, **args) as w_pose, ChannelWriter(
        seq_dir, config.POSE_VALID, **args
    ) as w_valid:
        for i, t in enumerate(scan_times):
            stem = f"{i:06d}"
            w_pose.add(poses[i], stem=stem, timestamp=float(t))
            w_valid.add(np.asarray(valid[i]), stem=stem, timestamp=float(t))
    logger.info(
        "%s: %s written, %d of %d scans with a pose", seq_dir.name, config.POSES,
        int(valid.sum()), len(valid),
    )
    return True


def load_validity(seq_dir, poses_key: str, poses: np.ndarray, times: np.ndarray) -> np.ndarray:
    """``(N,)`` bool: the scans whose pose may be used for labels.

    ``ins_poses`` carries its own ``ins_valid``. Another source must ship a
    mask (``<poses_key>/valid_mask.npy``, as the GICP poses do) or, for an
    estimate without one, passes the same fixed physical bound. Either way a
    non-finite pose is invalid.
    """
    seq_dir = Path(seq_dir)
    n = len(poses)
    if poses_key == config.POSES:
        files = sorted((seq_dir / config.POSE_VALID).glob("*.npy"))
        valid = np.array([bool(np.load(f)) for f in files])
    elif (seq_dir / poses_key / "valid_mask.npy").exists():
        valid = np.load(seq_dir / poses_key / "valid_mask.npy").astype(bool)
    else:
        logger.warning(
            "%s: %s has no validity mask; only the top-speed bound guards it",
            seq_dir.name, poses_key,
        )
        valid = np.ones(n, dtype=bool)
    if len(valid) != n:
        raise ValueError(f"{len(valid)} validity flags for {n} poses of '{poses_key}'")
    return pose_validity(poses, times, valid, max_speed=config.MAX_SPEED)
