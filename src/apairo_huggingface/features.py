"""The LeRobot feature schema, sized from a real frame."""

from __future__ import annotations

from apairo.core.sample import Sample

from . import config


def build_features(probe: Sample) -> dict:
    """Feature dict for ``LeRobotDataset.create``.

    The image shape is read from an actual frame so the example works at any
    camera resolution. ``dtype="video"`` matches ``use_videos=True`` (frames are
    concatenated and encoded to MP4 by LeRobot).
    """
    height, width, _ = probe.data[config.CAMERA].shape
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config.POSE_DIM,),
            "names": config.STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(config.ACTION_NAMES),),
            "names": config.ACTION_NAMES,
        },
    }
