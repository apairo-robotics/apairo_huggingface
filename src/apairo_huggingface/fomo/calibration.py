"""The FoMo rig: camera intrinsics and the RoboSense -> camera extrinsics.

Facts about the rig, each checked by projecting raw RoboSense scans onto the
images of ``2024-11-21/red_2024-11-21-10-34`` (and, for the driving direction,
by the KISS-ICP motion):

- **Driving direction.** ``calib/transforms.json`` gives ``base_link ->
  robosense`` as a 180 degree rotation about z: the robot drives along the
  RoboSense **-x** axis. KISS-ICP translation per scan points along -x.
- **Extrinsics are literal.** An entry ``(from=robosense, to=<camera>)`` of
  ``calib/transforms.json`` is ``T_robosense_from_camera_optical`` expressed in
  the RoboSense sensor frame, so ``T_camera_from_robosense`` is its plain
  inverse. Read that way:

  - ``zedx_left`` sits 0.08 m ahead of the lidar and its optical axis points
    along -x, tilted ~19 degrees down: it looks **forward**. Projected scans
    land on a near sign pole, the curb and the road edge.
  - ``basler`` sits 1.07 m behind the lidar and its optical axis points along
    +x: it looks **backward**. Projected scans fit tree silhouettes, a fence
    and a building; road optical flow in its images points toward the horizon
    and road signs recede. The Basler cannot see the future driven corridor.

  - ``zedx_right`` is given relative to ``zedx_left`` (``(from=zedx_left,
    to=zedx_right)``, the 0.12 m stereo baseline), so its extrinsic is the
    chain ``T_robosense_from_left @ T_left_from_right``, inverted.

  A check by eye of a ring pattern is not enough to tell a camera from its
  180-degree-flipped twin (the rings are nearly symmetric): check against
  thin vertical structures (poles, trunks) close to the robot.
- **Intrinsics are not the bucket's.** ``calib/basler.json`` and
  ``calib/zedx_left.json`` in the bucket are the same placeholder (f = 738 px,
  no distortion). The images are raw (unrectified, 1920x1200) and match the
  factory calibrations kept by the FoMo SDK (norlab-ulaval/fomo-sdk at
  ``67b74ae``): ``fomo_sdk/data/calib_to_ijrr/basler.json`` for the Basler,
  and sections ``[LEFT_CAM_FHD1200]`` / ``[RIGHT_CAM_FHD1200]`` of
  ``fomo_sdk/data/calib_to_ijrr/zedx_SN41705768.conf`` for the ZED X.
- **Footprint per camera.** A forward camera sees where the robot is going
  (the future footprint); the rear Basler sees only where it has been (the
  past footprint). :data:`FOOTPRINT` names which one each camera is labelled
  with.
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

ZEDX_LEFT = CameraIntrinsics(
    K=np.array(
        [
            [734.789, 0.0, 930.358],
            [0.0, 734.854, 606.359],
            [0.0, 0.0, 1.0],
        ]
    ),
    # k1, k2, p1, p2, k3 of [LEFT_CAM_FHD1200], zedx_SN41705768.conf
    distortion=np.array([-0.0107462, -0.0354004, -0.00023542, 0.000173096, 0.00980669]),
    model="plumb_bob",
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)

ZEDX_RIGHT = CameraIntrinsics(
    K=np.array(
        [
            [736.993, 0.0, 966.632],
            [0.0, 736.788, 586.738],
            [0.0, 0.0, 1.0],
        ]
    ),
    # k1, k2, p1, p2, k3 of [RIGHT_CAM_FHD1200], zedx_SN41705768.conf
    distortion=np.array([-0.0137929, -0.0310381, -0.000356811, -0.00015091, 0.00815849]),
    model="plumb_bob",
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)

INTRINSICS = {"zedx_left": ZEDX_LEFT, "zedx_right": ZEDX_RIGHT, "basler": BASLER}
"""Camera channel name -> intrinsics of its raw images."""

CHAIN = {"zedx_left": ("zedx_left",), "zedx_right": ("zedx_left", "zedx_right"), "basler": ("basler",)}
"""Frames walked from the RoboSense to each camera through ``transforms.json``."""

FOOTPRINT = {"zedx_left": "future", "zedx_right": "future", "basler": "past"}
"""Which driven footprint each camera can see: forward cameras the future one,
the rear-facing Basler only the past one."""

LIDAR = "robosense"
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


def transforms(traj_dir: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """``{(from, to): (4, 4) T}`` of one trajectory's ``calib/transforms.json``."""
    entries = json.loads((Path(traj_dir) / "calib" / "transforms.json").read_text())
    return {(e["from"], e["to"]): _transform(e) for e in entries}


def camera_from_lidar(traj_dir: str | Path, camera: str = "zedx_left") -> np.ndarray:
    """``(4, 4)`` T_camera_optical_from_robosense for one trajectory."""
    tf = transforms(traj_dir)
    lidar_from_camera, frame = np.eye(4), LIDAR
    for nxt in CHAIN[camera]:
        lidar_from_camera = lidar_from_camera @ tf[(frame, nxt)]
        frame = nxt
    return np.linalg.inv(lidar_from_camera)
