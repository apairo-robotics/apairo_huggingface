"""Per-point traversability labels of FoMo scans, as a labels-only companion.

``trav_traj`` holds one uint8 label per point of every raw RoboSense scan
(``1`` = under the robot's future driven footprint, ``0`` = anything else), in
the row order of FoMo's own ``robosense/<us>.bin`` file. The scans are not
redistributed: one parquet per trajectory lists, per scan, the raw file name,
its stamp, its point count, the row indices of the driven points (about 2 % of
a scan, so far smaller than the full mask) and the KISS-ICP pose. Anyone with
the raw FoMo scan rebuilds the mask with :func:`mask`.

Layout::

    <out>/robosense/<deployment>/<trajectory>.parquet

Columns: ``trajectory`` (``<deployment>/<trajectory>``), ``scan_stem``
(``NNNNNN``, the scan's rank in the trajectory), ``robosense_file``
(``<us>.bin``), ``timestamp_us`` (int64, the file's microsecond stamp),
``n_points`` (int32), ``driven_indices`` (list<int32>), ``pose`` (16 float64,
row-major ``T_world_lidar``, KISS-ICP, world = the first scan's lidar frame),
and, when ``trav_traj_past`` exists, ``driven_indices_past`` (list<int32>, the
points under the footprint swept along the trajectory *behind* the scan).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def mask(n_points: int, driven_indices) -> np.ndarray:
    """The ``(n_points,)`` uint8 per-point label from the driven row indices."""
    out = np.zeros(int(n_points), dtype=np.uint8)
    out[np.asarray(driven_indices, dtype=np.int64)] = 1
    return out


def _stamps_us(path: Path) -> np.ndarray:
    return np.round(np.loadtxt(path, dtype=np.float64, ndmin=1) * 1e6).astype(np.int64)


def trajectory_table(traj_dir):
    """The per-scan pyarrow table of one processed trajectory."""
    import pyarrow as pa

    traj_dir = Path(traj_dir)
    name = f"{traj_dir.parent.name}/{traj_dir.name}"
    labels_dir, poses_dir = traj_dir / config.TRAV_TRAJ, traj_dir / config.POSES
    stamps = _stamps_us(labels_dir / "timestamps.txt")
    if not np.array_equal(stamps, _stamps_us(poses_dir / "timestamps.txt")):
        raise ValueError(f"{name}: {config.TRAV_TRAJ} and {config.POSES} are not on the same clock")
    stems = sorted(p.stem for p in labels_dir.glob("*.npy"))
    if stems != [f"{i:06d}" for i in range(len(stamps))]:
        raise ValueError(f"{name}: {config.TRAV_TRAJ} stems do not match timestamps.txt")
    n_points, driven, poses = [], [], []
    for stem in stems:
        lab = np.load(labels_dir / f"{stem}.npy")
        n_points.append(len(lab))
        driven.append(np.flatnonzero(lab == 1).astype(np.int32))
        poses.append(np.load(poses_dir / f"{stem}.npy").astype(np.float64).reshape(16))
    columns = {
        "trajectory": pa.array([name] * len(stems), pa.string()),
        "scan_stem": pa.array(stems, pa.string()),
        "robosense_file": pa.array([f"{t}.bin" for t in stamps], pa.string()),
        "timestamp_us": pa.array(stamps, pa.int64()),
        "n_points": pa.array(n_points, pa.int32()),
        "driven_indices": pa.array(driven, pa.list_(pa.int32())),
        "pose": pa.array(poses, pa.list_(pa.float64(), 16)),
    }
    past_dir = traj_dir / config.TRAV_TRAJ_PAST
    if (past_dir / "timestamps.txt").exists():
        if not np.array_equal(stamps, _stamps_us(past_dir / "timestamps.txt")):
            raise ValueError(f"{name}: {config.TRAV_TRAJ_PAST} is not on the {config.TRAV_TRAJ} clock")
        past = []
        for stem, n in zip(stems, n_points):
            lab = np.load(past_dir / f"{stem}.npy")
            if len(lab) != n:
                raise ValueError(f"{name}: {config.TRAV_TRAJ_PAST}/{stem} has {len(lab)} labels, not {n}")
            past.append(np.flatnonzero(lab == 1).astype(np.int32))
        columns["driven_indices_past"] = pa.array(past, pa.list_(pa.int32()))
    return pa.table(columns)


def export(traj_dirs, out) -> list[Path]:
    """Write one parquet per trajectory under ``<out>/robosense/``; return the paths."""
    import pyarrow.parquet as pq

    out = Path(out)
    written = []
    for traj in map(Path, traj_dirs):
        table = trajectory_table(traj)
        path = out / config.LIDAR / traj.parent.name / f"{traj.name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")
        logger.info("%s/%s: %d scans -> %s (%.1f MB)", traj.parent.name, traj.name,
                    table.num_rows, path, path.stat().st_size / 1e6)
        written.append(path)
    return written
