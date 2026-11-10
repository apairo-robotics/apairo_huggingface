"""LeRobot ``observation.state`` and ``action`` from a trajectory of poses.

Shared by the RELLIS image export, the RELLIS BEV export and the TartanDrive
BEV export. ``poses`` are ``(N, 4, 4)`` T_world_robot (+x forward, +z up),
one per frame, ``stamps`` their times in seconds.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def relative_states(poses: np.ndarray) -> np.ndarray:
    """``(N, 7)`` float32 ``[x, y, z, qx, qy, qz, qw]``: each pose relative to ``poses[0]``,
    in the robot frame of ``poses[0]``. Pass one episode's poses for an episode-relative state."""
    origin = np.linalg.inv(poses[0])
    states = np.empty((len(poses), 7), dtype=np.float32)
    for k in range(len(poses)):
        rel = origin @ poses[k]
        states[k] = np.concatenate([rel[:3, 3], Rotation.from_matrix(rel[:3, :3]).as_quat()]).astype(np.float32)
    return states


def executed_actions(poses: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """``(N, 2)`` float32 executed forward speed (m/s) and yaw rate (rad/s), from
    frame ``k`` to ``k + 1``; the last frame repeats the previous action."""
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    actions = np.empty((len(poses), 2), dtype=np.float32)
    for k in range(len(poses)):
        a = min(k, len(poses) - 2)
        dt = stamps[a + 1] - stamps[a]
        step = np.linalg.inv(poses[a]) @ poses[a + 1]
        actions[k] = np.array([step[0, 3] / dt, (yaw[a + 1] - yaw[a]) / dt], dtype=np.float32)
    return actions
