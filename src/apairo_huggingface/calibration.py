"""The TartanDrive v2 rig: camera intrinsics and the velodyne->camera extrinsic.

Values come from the upstream calibration release
(`tartan_drive_2.0/calibration/`):

- **Intrinsics** are the rectified projection matrix ``P`` of
  ``/multisense/left/camera_info`` — the ``image_left_color`` PNGs are the
  1024x544 *rectified* images it describes, so there is no distortion to
  model (using the raw ``K``/``D`` here would double-apply the rectification).
- **Extrinsics** compose ``vehicle -> multisense/head ->
  multisense/left_camera_optical_frame`` from ``extrinsics.yaml`` (each entry
  is ``T_parent_from_child``).  The ``velodyne_0`` channel is captured in the
  vehicle frame (``vehicle -> velodyne_1`` is identity).  The composition is
  validated by the dataset authors themselves: its translation reproduces the
  ``p: [0.17265, -0.15227, 0.05708]`` left as a comment in that same file.

Once the recordings carry a ``cameras:`` section in ``.apairo/calibration.yaml``
(apairo ``register_intrinsics``), this module reduces to
``ds.calibration.get_intrinsics(...)`` / ``get_tf(...)`` calls.
"""

from __future__ import annotations

import numpy as np
from apairo.core.config import CameraIntrinsics

IMAGE_HEIGHT, IMAGE_WIDTH = 544, 1024

CAMERA = CameraIntrinsics(
    K=np.array(
        [
            [477.6049499511719, 0.0, 499.5],
            [0.0, 477.6049499511719, 252.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
)


def _T(translation, quaternion_xyzw) -> np.ndarray:
    """4x4 rigid transform from a translation and a unit quaternion (x, y, z, w)."""
    x, y, z, w = quaternion_xyzw
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = translation
    return out


_T_VEHICLE_FROM_HEAD = _T([-0.0949, 0.0641, -0.136], [-0.0119, 0.1131, 0.0003, 0.9935])
_T_HEAD_FROM_OPTICAL = _T([0.005, 0.105, 0.0], [-0.5, 0.5, -0.5, 0.5])

T_OPTICAL_FROM_VELODYNE = np.linalg.inv(_T_VEHICLE_FROM_HEAD @ _T_HEAD_FROM_OPTICAL)
