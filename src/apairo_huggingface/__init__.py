"""Export apairo robotics datasets to HuggingFace LeRobotDataset format.

The TartanDrive -> LeRobot bridge, built on apairo's read side. Public API:

    from apairo_huggingface import load_tartan, to_lerobot

    dataset = to_lerobot(load_tartan("/data/tartan"), "me/tartan_offroad")
"""

from __future__ import annotations

from . import config
from .export import aligned_frames, episodes, load_tartan, to_lerobot, validate
from .features import build_features
from .mappings import CHANNEL_TRANSFORMS, frame_of

__all__ = [
    "config",
    "load_tartan",
    "to_lerobot",
    "validate",
    "episodes",
    "aligned_frames",
    "build_features",
    "frame_of",
    "CHANNEL_TRANSFORMS",
]
