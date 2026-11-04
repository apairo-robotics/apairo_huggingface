"""The FoMo rig: which way each camera looks, and where the ground ahead lands.

The transforms are the ``calib/transforms.json`` entries shipped with every
red trajectory (identical across the 12 deployments). A camera whose
extrinsic is read with the wrong convention looks the opposite way; these
tests pin the convention to the driving direction of the robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apairo_huggingface.fomo import calibration as C
from apairo_huggingface.fomo.labeling import _project_unclipped

TRANSFORMS = [
    {"from": "base_link", "to": "robosense",
     "position": {"x": 0.0, "y": 0.0, "z": 0.0},
     "orientation": {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}},
    {"from": "robosense", "to": "basler",
     "position": {"x": 1.066401777414892, "y": -0.04479714911878229, "z": -0.2229260982579288},
     "orientation": {"x": 0.5643909887493772, "y": -0.5554999657574516,
                     "z": 0.42771568621203176, "w": -0.43582323438538856}},
    {"from": "robosense", "to": "zedx_left",
     "position": {"x": -0.07952352627148711, "y": -0.04440832511705086, "z": -0.21333361167112735},
     "orientation": {"x": 0.5729177995838587, "y": 0.5754445229626066,
                     "z": -0.4169693162625371, "w": -0.4083691775928738}},
    {"from": "zedx_left", "to": "zedx_right",
     "position": {"x": 0.119702, "y": -0.000191556, "z": 5.27709e-05},
     "orientation": {"x": 0.0013524997200016092, "y": 0.004252502083427449,
                     "z": 0.0005769131339936735, "w": 0.9999898770196494}},
]
EVIDENCE = Path("/mnt/vault-fellowship/fomo/2024-11-21/red_2024-11-21-10-34")


@pytest.fixture
def rig(tmp_path):
    (tmp_path / "calib").mkdir()
    (tmp_path / "calib" / "transforms.json").write_text(json.dumps(TRANSFORMS))
    return tmp_path


def _driving_direction(traj_dir) -> np.ndarray:
    """base_link +x (forward) in the RoboSense frame."""
    R = C.transforms(traj_dir)[("base_link", "robosense")][:3, :3]  # base <- lidar
    return R.T @ np.array([1.0, 0.0, 0.0])


def _optical_axis(traj_dir, camera) -> np.ndarray:
    return np.linalg.inv(C.camera_from_lidar(traj_dir, camera))[:3, 2]


def test_robot_drives_along_minus_x_of_the_lidar(rig):
    np.testing.assert_allclose(_driving_direction(rig), [-1.0, 0.0, 0.0], atol=1e-9)


def test_zed_looks_forward_and_basler_looks_backward(rig):
    forward = _driving_direction(rig)
    assert _optical_axis(rig, "zedx_left") @ forward > 0.9
    assert _optical_axis(rig, "zedx_right") @ forward > 0.9
    assert _optical_axis(rig, "basler") @ forward < -0.9
    assert {c: f for c, f in C.FOOTPRINT.items()} == {"zedx_left": "future", "zedx_right": "future",
                                                       "basler": "past"}


def test_zed_right_is_chained_through_the_left_camera(rig):
    centre = {c: np.linalg.inv(C.camera_from_lidar(rig, c))[:3, 3] for c in ("zedx_left", "zedx_right")}
    forward = _driving_direction(rig)
    right = np.cross(forward, [0.0, 0.0, 1.0])               # robot right, lidar frame
    baseline = centre["zedx_right"] - centre["zedx_left"]
    assert abs(np.linalg.norm(baseline) - 0.1197) < 1e-3
    assert baseline @ right > 0.119                           # the right camera is on the right
    assert C.camera_from_lidar(rig).tolist() == C.camera_from_lidar(rig, "zedx_left").tolist()


def _ground_ahead(traj_dir, distance, lateral=0.0):
    """Lidar-frame ground point ``distance`` m ahead, ``lateral`` m to the left."""
    forward = _driving_direction(traj_dir)
    left = np.cross([0.0, 0.0, 1.0], forward)
    return distance * forward + lateral * left + [0.0, 0.0, -C.LIDAR_HEIGHT]


def _pixel(traj_dir, camera, p):
    E = C.camera_from_lidar(traj_dir, camera)
    u, v, z = _project_unclipped((E[:3, :3] @ p + E[:3, 3])[None], C.INTRINSICS[camera])
    return float(u[0]), float(v[0]), float(z[0])


def test_ground_ahead_projects_below_the_horizon_centred_and_widens_downward(rig):
    horizon = _pixel(rig, "zedx_left", _ground_ahead(rig, 1e4))[1]
    assert 0 < horizon < C.IMAGE_HEIGHT / 2           # tilted down: the horizon is high
    width = {}
    for d in (4.0, 8.0, 15.0):
        ul, vl, zl = _pixel(rig, "zedx_left", _ground_ahead(rig, d, +0.7))
        ur, vr, zr = _pixel(rig, "zedx_left", _ground_ahead(rig, d, -0.7))
        assert zl > 0 and zr > 0
        assert ul < ur                                  # robot left is image left
        assert horizon < vl < C.IMAGE_HEIGHT           # below the horizon, in the image
        assert abs((ul + ur) / 2 - C.IMAGE_WIDTH / 2) < 0.1 * C.IMAGE_WIDTH
        width[d] = ur - ul
    assert width[4.0] > width[8.0] > width[15.0]       # the corridor widens toward the bottom
    # ... and the near end is lower in the image than the far end
    assert _pixel(rig, "zedx_left", _ground_ahead(rig, 4.0))[1] > _pixel(rig, "zedx_left", _ground_ahead(rig, 15.0))[1]


def test_ground_ahead_is_behind_the_basler(rig):
    assert _pixel(rig, "basler", _ground_ahead(rig, 8.0))[2] < 0


def test_zed_intrinsics_are_the_factory_calibration_not_the_bucket_placeholder():
    K = C.ZEDX_LEFT.K
    assert (K[0, 0], K[1, 1], K[0, 2], K[1, 2]) == (734.789, 734.854, 930.358, 606.359)
    assert C.ZEDX_LEFT.distortion.tolist() == [-0.0107462, -0.0354004, -0.00023542, 0.000173096, 0.00980669]
    assert (C.ZEDX_LEFT.width, C.ZEDX_LEFT.height) == (1920, 1200)


def test_ground_behind_projects_into_the_basler(rig):
    u, v, z = _pixel(rig, "basler", _ground_ahead(rig, -8.0))
    assert z > 0 and 0 <= u < C.IMAGE_WIDTH and 0 <= v < C.IMAGE_HEIGHT


@pytest.mark.skipif(not (EVIDENCE / "kissicp_poses").is_dir(), reason="FoMo data not on this machine")
def test_zed_axis_follows_the_measured_motion():
    """KISS-ICP motion of a real trajectory, not the calibration file, sets 'forward'."""
    files = sorted((EVIDENCE / "kissicp_poses").glob("*.npy"))
    P = np.stack([np.load(f) for f in files])
    step = np.einsum("nij,nj->ni", np.linalg.inv(P[:-1])[:, :3, :3], P[1:, :3, 3] - P[:-1, :3, 3])
    step = step[np.linalg.norm(step, axis=1) > 0.05]
    motion = (step / np.linalg.norm(step, axis=1, keepdims=True)).mean(0)
    motion /= np.linalg.norm(motion)
    assert _optical_axis(EVIDENCE, "zedx_left") @ motion > 0.9
    assert _optical_axis(EVIDENCE, "zedx_right") @ motion > 0.9
    assert _optical_axis(EVIDENCE, "basler") @ motion < -0.9
