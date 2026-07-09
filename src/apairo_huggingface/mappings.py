"""Per-frame value ops + LeRobot feature naming -- the only TartanDrive-specific code.

Everything hard (loading async sensors, resampling, episode segmentation) is done
by apairo upstream. What is left is: (1) standard ``apairo_transform`` callables
that cast/slice each channel, applied to the synchronized view in
:mod:`.export`, and (2) :func:`frame_of`, which only *names* the LeRobot
features -- no data logic.
"""

from __future__ import annotations

import numpy as np
from apairo import Compose
from apairo_transform import CastTo, ChannelSelect

from . import config

# Per-frame value transforms, applied to the synchronized view (see export.py).
# gps_odom (pose 7 + twist 6) -> pose as float32; cmd -> float32; image untouched.
CHANNEL_TRANSFORMS = {
    config.STATE: Compose([ChannelSelect(range(config.POSE_DIM)), CastTo(np.float32)]),
    config.ACTION: CastTo(np.float32),
}


def frame_of(sample) -> dict:
    """Name one LeRobot frame from an aligned, already-transformed sample.

    Pure naming: ``task`` is a key of the frame dict in LeRobotDataset v3.
    """
    return {
        "observation.images.front": sample.data[config.CAMERA],
        "observation.state": sample.data[config.STATE],
        "action": sample.data[config.ACTION],
        "task": config.TASK,
    }
