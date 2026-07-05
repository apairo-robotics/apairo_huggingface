"""Channel -> feature mappings: the only domain code in this package.

Each function turns one aligned apairo channel into one LeRobot feature. They are
pure and tiny on purpose -- everything hard (loading, resampling, episode
segmentation) is done by apairo upstream.
"""

from __future__ import annotations

import numpy as np
from apairo.core.sample import Sample

from . import config


def image_front(sample: Sample) -> np.ndarray:
    """Left colour camera, ready as-is for a LeRobot video feature (H, W, 3 uint8)."""
    return sample.data[config.CAMERA]


def odom_to_state(sample: Sample) -> np.ndarray:
    """Robot pose (x, y, z, quaternion) from the odometry row."""
    return sample.data[config.STATE][:config.POSE_DIM].astype(np.float32)


def twist_to_action(sample: Sample) -> np.ndarray:
    """Drive command as [linear_velocity, angular_velocity]."""
    return sample.data[config.ACTION].astype(np.float32)


def frame_of(sample: Sample) -> dict:
    """Assemble one LeRobot frame from an aligned apairo sample.

    ``task`` is a key of the frame dict in LeRobotDataset v3.
    """
    return {
        "observation.images.front": image_front(sample),
        "observation.state": odom_to_state(sample),
        "action": twist_to_action(sample),
        "task": config.TASK,
    }
