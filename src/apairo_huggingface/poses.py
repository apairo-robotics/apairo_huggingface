"""LeRobot ``observation.state`` and ``action`` from a trajectory of poses.

Shared by the RELLIS image export, the RELLIS BEV export, the TartanDrive
BEV export and (``pose_state``) the TartanDrive image export. ``poses`` are
``(N, 4, 4)`` T_world_robot (+x forward, +z up), one per frame, ``stamps``
their times in seconds.

States are relative to the episode's first pose and computed in float64
before the float32 cast: a float32 world coordinate is quantised to 0.5 m at
UTM scale (~4.5e6 m), while an episode-relative one keeps sub-millimetre
resolution over tens of kilometres.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose_state(pose: np.ndarray) -> np.ndarray:
    """``(7,)`` float64 ``[x, y, z, qx, qy, qz, qw]`` of one ``(4, 4)`` pose."""
    pose = np.asarray(pose, dtype=np.float64)
    return np.concatenate([pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()])


def relative_states(poses: np.ndarray) -> np.ndarray:
    """``(N, 7)`` float32 ``[x, y, z, qx, qy, qz, qw]``: each pose relative to ``poses[0]``,
    in the robot frame of ``poses[0]``. Pass one episode's poses for an episode-relative state."""
    poses = np.asarray(poses, dtype=np.float64)
    origin = np.linalg.inv(poses[0])
    states = np.empty((len(poses), 7), dtype=np.float32)
    for k in range(len(poses)):
        states[k] = pose_state(origin @ poses[k]).astype(np.float32)
    return states


def executed_actions(poses: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """``(N, 2)`` float32 executed forward speed (m/s) and yaw rate (rad/s), from
    frame ``k`` to ``k + 1``; the last frame repeats the previous action.

    Frame ``k + 1`` is the next frame *passed in*: pass the whole recording,
    not one episode, so an episode's last frame gets the motion into the next
    episode; frames dropped in between (a gap) make the action the mean over
    the gap. An action needs two poses with increasing stamps: fewer poses, or
    a stamp that does not increase, raise ``ValueError`` instead of writing a
    NaN or an infinite speed."""
    poses = np.asarray(poses, dtype=np.float64)
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(poses) != len(stamps):
        raise ValueError(f"{len(poses)} poses for {len(stamps)} stamps")
    if len(poses) < 2:
        raise ValueError(f"an executed action needs at least two poses, got {len(poses)}")
    dts = np.diff(stamps)
    if not (dts > 0).all():
        k = int(np.flatnonzero(~(dts > 0))[0])
        raise ValueError(f"stamps must increase: frame {k + 1} at t={stamps[k + 1]:.6f} "
                         f"follows t={stamps[k]:.6f}")
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    actions = np.empty((len(poses), 2), dtype=np.float32)
    for k in range(len(poses)):
        a = min(k, len(poses) - 2)
        dt = dts[a]
        step = np.linalg.inv(poses[a]) @ poses[a + 1]
        actions[k] = np.array([step[0, 3] / dt, (yaw[a + 1] - yaw[a]) / dt], dtype=np.float32)
    return actions
