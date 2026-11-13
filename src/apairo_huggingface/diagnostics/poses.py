"""How far ahead do two trajectories agree? A diagnostic, outside any pipeline.

Trajectory labels place the robot's future path in the frame of the current
scan. What matters is therefore not the drift over a whole recording but the
offset between two sources of that future path, seen from each scan, as a
function of how far ahead it is looked at. This module measures it, to choose
a look-ahead once per dataset; no pipeline calls it.

Sources are compared on the lidar clock:

- **delivered rows** (``odometry.npy`` + ``timestamps.txt``, ``[x y z qx qy qz
  qw, ...]``): the vehicle's INS or a released trajectory, looked up at the
  scan times with their own clock offset, estimated here from the rotation
  angle over 1 s windows (which a lever arm does not change);
- **pose channels** on the lidar clock (one ``(4, 4)`` per scan, per-frame
  ``.npy`` or a stacked ``poses.npy``): an odometry computed beforehand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from apairo_preprocess import reference_poses
from scipy.spatial.transform import Rotation

DISTANCES = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)


@dataclass
class Trajectory:
    """``(N, 4, 4)`` poses on the scan clock, relative to the first, NaN where none."""

    name: str
    poses: np.ndarray
    clock_offset: float = 0.0


@dataclass
class Comparison:
    """Offsets of the future path between two sources, per look-ahead distance."""

    a: str
    b: str
    distances: tuple[float, ...]
    lateral: np.ndarray            # (scans, distances), metres
    vertical: np.ndarray           # (scans, distances), metres
    clock_offsets: dict[str, float] = field(default_factory=dict)

    def summary(self, tolerance: float = 0.30) -> dict:
        lat, ver = self.lateral, self.vertical
        agree = (lat < tolerance) & (ver < tolerance)
        return {
            "pair": f"{self.a} / {self.b}",
            "scans": int(len(lat)),
            "clock_offsets_s": self.clock_offsets,
            "tolerance_m": tolerance,
            "per_distance": {
                f"{d:g}": {
                    "lateral_median_m": float(np.median(lat[:, k])),
                    "lateral_p95_m": float(np.percentile(lat[:, k], 95)),
                    "vertical_median_m": float(np.median(ver[:, k])),
                    "vertical_p95_m": float(np.percentile(ver[:, k], 95)),
                    "agreeing_share": float(np.mean(np.all(agree[:, : k + 1], axis=1))),
                }
                for k, d in enumerate(self.distances)
            },
        }


def load_pose_channel(directory: Path, name: str | None = None) -> Trajectory:
    """A pose channel on the lidar clock: per-frame ``.npy`` or ``poses.npy``."""
    directory = Path(directory)
    stacked = directory / "poses.npy"
    if stacked.exists():
        poses = np.load(stacked).astype(np.float64)
    else:
        files = sorted(p for p in directory.glob("*.npy") if not p.stem.endswith("_mask"))
        poses = np.stack([np.load(p) for p in files]).astype(np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{directory}: poses must be (N, 4, 4), got {poses.shape}")
    finite = np.all(np.isfinite(poses.reshape(len(poses), -1)), axis=1)
    if finite.any():
        poses[finite] = np.linalg.inv(poses[int(np.flatnonzero(finite)[0])]) @ poses[finite]
    return Trajectory(name or directory.name, poses)


def _angles(poses: np.ndarray, window: int) -> np.ndarray:
    ok = np.all(np.isfinite(poses.reshape(len(poses), -1)), axis=1)
    out = np.full(len(poses) - window, np.nan)
    i = np.flatnonzero(ok[:-window] & ok[window:])
    if len(i):
        R = Rotation.from_matrix(poses[:, :3, :3][ok])
        idx = np.cumsum(ok) - 1
        out[i] = (R[idx[i]].inv() * R[idx[i + window]]).magnitude()
    return out


def load_rows(
    directory: Path,
    scan_times: np.ndarray,
    *,
    against: Trajectory | None = None,
    clock_offset: float | None = None,
    search: float = 0.3,
    name: str | None = None,
) -> Trajectory:
    """Delivered rows looked up on the scan clock.

    Without ``clock_offset``, it is estimated against ``against``: the shift,
    in 10 ms steps within ``+-search`` s, that best matches the rotation angle
    over 1 s windows.
    """
    directory = Path(directory)
    rows = np.load(directory / "odometry.npy")
    times = np.loadtxt(directory / "timestamps.txt")

    def at(offset):
        return reference_poses(rows, times, scan_times, clock_offset=offset, max_gap=0.2)[0]

    if clock_offset is None:
        if against is None:
            raise ValueError("give clock_offset, or a trajectory to estimate it against")
        window = 10
        ref = _angles(against.poses, window)
        best, best_err = 0.0, np.inf
        for off in np.arange(-search, search + 1e-9, 0.01):
            err = np.nanmedian(np.abs(_angles(at(off), window) - ref))
            if err < best_err:
                best, best_err = float(off), err
        clock_offset = round(best, 2)
    return Trajectory(name or directory.name, at(clock_offset), clock_offset)


def path_ahead(poses: np.ndarray, i: int, distances) -> np.ndarray | None:
    """``(len(distances), 3)``: the path ``d`` metres ahead of scan ``i``, in its
    frame; ``None`` when the path ends, or meets a scan without a pose, first."""
    future = poses[i:]
    ok = np.all(np.isfinite(future.reshape(len(future), -1)), axis=1)
    stop = int(np.argmin(ok)) if not ok.all() else len(future)
    future = future[:stop]
    if len(future) < 2:
        return None
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(future[:, :2, 3], axis=0), axis=1))]
    if s[-1] < max(distances):
        return None
    local = (np.linalg.inv(future[0]) @ np.c_[future[:, :3, 3], np.ones(len(future))].T).T[:, :3]
    return np.stack([[np.interp(d, s, local[:, k]) for k in range(3)] for d in distances])


def compare(a: Trajectory, b: Trajectory, *, distances=DISTANCES, stride: int = 10,
            min_motion: float = 0.5) -> Comparison:
    """Lateral and vertical offset of the future path, every ``stride`` scans
    where the robot moves (``min_motion`` metres over the next ``stride``)."""
    if len(a.poses) != len(b.poses):
        raise ValueError(f"{a.name}: {len(a.poses)} scans, {b.name}: {len(b.poses)}")
    lat, ver = [], []
    n = len(a.poses)
    for i in range(0, n - stride, stride):
        step = np.linalg.norm(a.poses[i + stride, :3, 3] - a.poses[i, :3, 3])
        if not step >= min_motion:  # NaN included
            continue
        pa, pb = path_ahead(a.poses, i, distances), path_ahead(b.poses, i, distances)
        if pa is None or pb is None:
            continue
        lat.append(np.abs(pa[:, 1] - pb[:, 1]))
        ver.append(np.abs(pa[:, 2] - pb[:, 2]))
    shape = (0, len(distances))
    return Comparison(
        a.name, b.name, tuple(distances),
        np.array(lat).reshape(-1, len(distances)) if lat else np.empty(shape),
        np.array(ver).reshape(-1, len(distances)) if ver else np.empty(shape),
        {t.name: t.clock_offset for t in (a, b)},
    )


def format_summary(s: dict) -> str:
    lines = [f"{s['pair']}: {s['scans']} scans, clock offsets {s['clock_offsets_s']}"]
    lines.append(f"  {'ahead':>6}  {'lateral med/p95':>17}  {'vertical med/p95':>18}  agree<{s['tolerance_m']*100:.0f}cm")
    for d, v in s["per_distance"].items():
        lines.append(
            f"  {d:>4} m  {v['lateral_median_m']*100:6.1f} / {v['lateral_p95_m']*100:6.1f} cm"
            f"  {v['vertical_median_m']*100:6.1f} / {v['vertical_p95_m']*100:6.1f} cm"
            f"  {v['agreeing_share']*100:5.0f} %"
        )
    return "\n".join(lines)
