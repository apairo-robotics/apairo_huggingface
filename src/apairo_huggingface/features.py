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
    mask_height, mask_width = probe.data[config.TRAV_MASK].shape
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        # "image" (lossless PNG), not "video": mp4 compression would corrupt
        # the {0, 1, 255} label values. Single-channel mask replicated to RGB.
        "observation.traversability_mask": {
            "dtype": "image",
            "shape": (mask_height, mask_width, 3),
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
