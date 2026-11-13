"""``apairo-pose-diagnostic``: how far ahead do pose sources agree on a recording?

Example, on a TartanDrive recording, the INS against KISS-ICP and the
release's own SLAM trajectory (downloaded next to the recording)::

    apairo-pose-diagnostic /data/tartan/<recording> --kissicp --work /tmp/diag \\
        --against /data/tartan/<recording>/super_odom
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np

from .. import config
from ..reference_poses import LIDAR, ins
from .poses import Trajectory, compare, format_summary, load_pose_channel, load_rows

logger = logging.getLogger(__name__)


def _scan_times(seq: Path) -> np.ndarray:
    return np.loadtxt(seq / LIDAR / "timestamps.txt")


def _ins(seq: Path, scan_times: np.ndarray) -> Trajectory:
    odom = seq / config.STATE
    poses, _ = ins()(np.load(odom / "odometry.npy"), np.loadtxt(odom / "timestamps.txt"), scan_times)
    return Trajectory("ins", poses, config.INS_CLOCK_OFFSET)


def _kissicp(seq: Path, work: Path) -> Trajectory:
    """KISS-ICP over the recording's scans, cached in ``work``."""
    from apairo.core.sample import Sample
    from apairo_preprocess import KissICPOdometry

    cache = work / f"{seq.name}_kissicp_poses.npy"
    if not cache.exists():
        files = sorted(p for p in (seq / LIDAR).glob("*.npy") if not p.stem.endswith("_intensity"))
        odo = KissICPOdometry(max_range=100.0, min_range=2.0, voxel_size=1.0)
        poses = np.stack([odo(Sample(data={"lidar": np.load(f)[:, :3]})) for f in files])
        work.mkdir(parents=True, exist_ok=True)
        np.save(cache, poses)
    poses = np.load(cache)
    return Trajectory("kissicp", np.linalg.inv(poses[0]) @ poses)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path, help="a TartanDrive recording directory")
    p.add_argument("--against", type=Path, action="append", default=[],
                   help="another source: delivered rows (odometry.npy + timestamps.txt) "
                   "or a pose channel on the lidar clock; repeatable")
    p.add_argument("--kissicp", action="store_true", help="also compare KISS-ICP, run here")
    p.add_argument("--work", type=Path, help="cache directory for --kissicp")
    p.add_argument("--tolerance", type=float, default=0.30, help="agreement tolerance, m")
    p.add_argument("--json", type=Path, help="write the summaries there")
    args = p.parse_args(argv)
    if args.kissicp and args.work is None:
        p.error("--kissicp needs --work")

    seq = args.recording
    scan_times = _scan_times(seq)
    sources = [_ins(seq, scan_times)]
    if args.kissicp:
        sources.append(_kissicp(seq, args.work))
    # Delivered rows get their clock offset estimated against the lidar's own
    # motion: KISS-ICP when it runs, else a pose channel given before them.
    for path in args.against:
        path = path if path.is_absolute() or path.exists() else seq / path
        if (path / "odometry.npy").exists():
            lidar = [s for s in sources[1:] if s.clock_offset == 0.0 and s.name != "ins"]
            if lidar:
                sources.append(load_rows(path, scan_times, against=lidar[0]))
            else:
                logger.warning("%s: no lidar trajectory to estimate its clock offset; 0 s", path.name)
                sources.append(load_rows(path, scan_times, clock_offset=0.0))
        else:
            sources.append(load_pose_channel(path))
    if len(sources) < 2:
        p.error("nothing to compare: give --against and/or --kissicp")

    summaries = []
    for a, b in itertools.combinations(sources, 2):
        s = compare(a, b).summary(args.tolerance)
        summaries.append(s)
        print(format_summary(s))
    if args.json:
        args.json.write_text(json.dumps({"recording": seq.name, "pairs": summaries}, indent=1))
    return 0
