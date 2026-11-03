"""The RELLIS-3D rig: Basler (pylon) intrinsics and the Ouster -> Basler extrinsic.

Established by projecting labelled scans onto images of all five sequences and
scoring the label agreement with the official 2D annotations
(``apairo-rellis calib-check``, :mod:`.calibcheck`):

- **Extrinsic.** ``Rellis_3D/<seq>/transforms.yaml`` (key
  ``os1_cloud_node-pylon_camera_node``) holds a quaternion ``q`` and a
  translation ``t`` that compose ``T_lidar_from_camera`` -- the camera pose in
  the Ouster frame. Projection needs its **inverse**,
  ``T_camera_from_lidar``; used as is, the points land on the wrong side of
  the image. The transform differs per sequence (00000 and 00002 share one),
  so it is read per sequence. ``calib.txt`` in each sequence is all identity
  and is not a camera calibration.
- **Intrinsics.** ``Rellis-3D/<seq>/camera_info.txt`` gives
  ``fx fy cx cy`` = 2813.64 2808.33 969.29 624.05 (the 02/2021 recalibration;
  identical in every sequence). Scores below are mean point agreement over
  the calib-check sample (40 annotated frames per sequence, 200 in all, seed
  0). The released JPEGs behave as **rectified**: the plumb-bob coefficients
  hard-coded in the official ``lidar2img`` notebook (k1 = -0.134) lower the
  agreement in every sequence (0.755 vs 0.826). A positive k1 of 0.02 / 0.04
  / 0.06 moves the mean by +0.0001 / +0.0004 / +0.0006, and no sequence by
  more than +0.0022. The older ``calibrationdata_pylon_iris5.6`` intrinsics
  (cx = 881, Feb 2020) score lower in every sequence: 0.790 with its camera
  matrix, 0.788 with its projection matrix, 0.734 with its camera matrix and
  distortion. So the projection is pinhole with ``camera_info.txt``, no
  distortion.
- **Residual.** Compact, crisply bounded classes (barrier, puddle, bush) agree
  best when the points are moved 16-24 px down (about 0.3-0.5 deg of pitch).
  It is reported, not corrected: a correction fitted on the 2D annotations
  would bias the agreement benchmark computed against them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

IMAGE_HEIGHT, IMAGE_WIDTH = 1200, 1920
TRANSFORM_KEY = "os1_cloud_node-pylon_camera_node"


def intrinsics(root: str | Path, seq: str) -> np.ndarray:
    """``(3, 3)`` pinhole K of the rectified Basler images of one sequence."""
    fx, fy, cx, cy = np.loadtxt(Path(root) / "Rellis-3D" / seq / "camera_info.txt")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def lidar_from_camera(root: str | Path, seq: str) -> np.ndarray:
    """``(4, 4)`` T_lidar_from_camera as stored in ``transforms.yaml``."""
    entry = yaml.safe_load((Path(root) / "Rellis_3D" / seq / "transforms.yaml").read_text())[TRANSFORM_KEY]
    q, t = entry["q"], entry["t"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return T


def camera_from_lidar(root: str | Path, seq: str) -> np.ndarray:
    """``(4, 4)`` T_camera_optical_from_lidar: the inverse of the stored transform."""
    return np.linalg.inv(lidar_from_camera(root, seq))
