"""The FoMo rig: Basler intrinsics and the RoboSense -> Basler extrinsic.

Two facts were established by projecting scans onto images (2025-10-14 red
trajectory: road edges, fence, tree line and horizon land where they belong),
and both differ from a literal reading of the files shipped with each
trajectory:

- **Intrinsics.** ``calib/basler.json`` in the bucket gives f = 738 px and no
  distortion, which projects every point too far from the image centre. The
  raw PNGs match the calibration the FoMo SDK keeps in
  ``fomo_sdk/data/calib_to_ijrr/basler.json`` (norlab-ulaval/fomo-sdk at
  ``67b74ae``): f = 376 px, plumb-bob distortion, ``do_rectify: true``.
- **Extrinsic.** The ``robosense -> basler`` entry of ``calib/transforms.json``
  is the camera pose in a frame oriented like ``base_link`` (+x forward), not
  in the RoboSense sensor frame, which the same file rotates by 180 degrees
  about z (``base_link -> robosense``): the robot drives along the sensor's -x.
  So ``T_camera_from_robosense = inv(T_base_from_camera) @ T_base_from_robosense``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from apairo.core.config import CameraIntrinsics

IMAGE_HEIGHT, IMAGE_WIDTH = 1200, 1920

BASLER = CameraIntrinsics(
    K=np.array(
        [
            [376.31326302, 0.0, 936.33256844],
            [0.0, 376.31326302, 628.93459367],
            [0.0, 0.0, 1.0],
        ]
    ),
    distortion=np.array([0.00840481, -0.00601427, -0.00062722, 0.0007329, 0.00042214]),
    model="plumb_bob",
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)

LIDAR_HEIGHT = 1.114  # base_link -> footprint, m: the RoboSense sits on base_link


def _transform(entry: dict) -> np.ndarray:
    x, y, z, w = (entry["orientation"][k] for k in "xyzw")
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = [entry["position"][k] for k in "xyz"]
    return T


def camera_from_lidar(traj_dir: str | Path) -> np.ndarray:
    """``(4, 4)`` T_basler_optical_from_robosense for one trajectory."""
    entries = json.loads((Path(traj_dir) / "calib" / "transforms.json").read_text())
    tf = {(e["from"], e["to"]): _transform(e) for e in entries}
    base_from_camera = tf[("robosense", "basler")]
    base_from_lidar = tf[("base_link", "robosense")]
    return np.linalg.inv(base_from_camera) @ base_from_lidar
