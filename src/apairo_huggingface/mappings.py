"""Per-frame value ops + LeRobot feature naming -- the only TartanDrive-specific code.

Everything hard (loading async sensors, resampling, episode segmentation) is done
by apairo upstream. What is left is: (1) standard ``apairo_transform`` callables
that slice/cast each channel, plus :class:`RelativeTo`, which re-expresses the
pose relative to the episode's first one, applied to the synchronized view in
:mod:`.export`, and (2) :func:`frame_of`, which only *names* the LeRobot
features -- no data logic.
"""

from __future__ import annotations

import numpy as np
from apairo import Compose
from apairo_transform import CastTo, ChannelSelect, InvertPose, PoseTo4x4

from . import config
from .poses import pose_state


class RelativeTo:
    """``[x, y, z, qx, qy, qz, qw]`` pose -> the same pose relative to ``origin``
    (same layout), in ``origin``'s body frame, float64.

    ``gps_odom`` positions are UTM-scale (x ~ 4.5e6 m): cast to float32 they
    fall on a 0.5 m grid. Subtracting the episode's first pose in float64
    first keeps the float32 state exact to well under a millimetre.
    """

    def __init__(self, origin) -> None:
        self._origin = np.asarray(origin, dtype=np.float64)
        self._inverse = InvertPose()(PoseTo4x4()(self._origin))

    def __call__(self, pose: np.ndarray) -> np.ndarray:
        return pose_state(self._inverse @ PoseTo4x4()(np.asarray(pose, dtype=np.float64)))

    def __repr__(self) -> str:
        return f"RelativeTo({self._origin.tolist()})"


def channel_transforms(origin) -> dict:
    """Per-frame value transforms of one episode, applied to its synchronized view
    (see export.py). gps_odom (pose 7 + twist 6) -> pose relative to ``origin``
    (the episode's first ``[x, y, z, qx, qy, qz, qw]``), then float32; cmd -> float32;
    image untouched."""
    return {
        config.STATE: Compose([ChannelSelect(range(config.POSE_DIM)), RelativeTo(origin),
                               CastTo(np.float32)]),
        config.ACTION: CastTo(np.float32),
    }


def frame_of(sample) -> dict:
    """Name one LeRobot frame from an aligned, already-transformed sample.

    Pure naming: ``task`` is a key of the frame dict in LeRobotDataset v3.
    """
    mask = np.asarray(sample.data[config.TRAV_MASK])
    return {
        "observation.images.front": sample.data[config.CAMERA],
        "observation.traversability_mask": np.repeat(mask[:, :, None], 3, axis=2),
        "observation.state": sample.data[config.STATE],
        "action": sample.data[config.ACTION],
        "task": config.TASK,
    }
