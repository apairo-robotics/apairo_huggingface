"""FoMo end to end on a synthetic trajectory: label_scans -> label_images ->
frames -> to_lerobot -> reload. Image ``k`` has intensity ``6 k``, the robot
accelerates along the lidar -x and the command's linear part is its own stamp,
so each exported row can be checked against the instant it claims to be."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from apairo import ChannelWriter
from apairo_huggingface.fomo import config, export
from apairo_huggingface.fomo.format import RECORD
from apairo_huggingface.fomo.labeling import label_images, label_scans

T0 = 1732203312_000000      # us
N_SCANS = 24
CAM_OFFSET_US = 37_000      # the camera ticks 37 ms after each scan
MIN_FUTURE_PATH = 4.0       # m; the synthetic drive is ~10 m long
# The real rig (calib/transforms.json of a 2024-11-21 trajectory).
TRANSFORMS = [
    {"from": "base_link", "to": "robosense", "position": {"x": 0.0, "y": 0.0, "z": 0.0},
     "orientation": {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}},
    {"from": "robosense", "to": "basler",
     "position": {"x": 1.066401777414892, "y": -0.04479714911878229, "z": -0.2229260982579288},
     "orientation": {"x": 0.5643909887493772, "y": -0.5554999657574516, "z": 0.42771568621203176, "w": -0.43582323438538856}},
    {"from": "robosense", "to": "zedx_left",
     "position": {"x": -0.07952352627148711, "y": -0.04440832511705086, "z": -0.21333361167112735},
     "orientation": {"x": 0.5729177995838587, "y": 0.5754445229626066, "z": -0.4169693162625371, "w": -0.4083691775928738}},
    {"from": "zedx_left", "to": "zedx_right", "position": {"x": 0.119702, "y": -0.000191556, "z": 5.27709e-05},
     "orientation": {"x": 0.0013524997200016092, "y": 0.004252502083427449, "z": 0.0005769131339936735, "w": 0.9999898770196494}},
]


def distance(t):
    """Metres driven ``t`` seconds after T0: accelerating."""
    return 2.0 * t * t


def _trajectory(traj, camera=config.CAMERA):
    """A FoMo trajectory directory; returns the camera stamps (us)."""
    for name in ("robosense", "calib", "metadata", camera):
        (traj / name).mkdir(parents=True)
    (traj / "calib" / "transforms.json").write_text(json.dumps(TRANSFORMS))
    # Ground 1.114 m below the lidar, ahead of the robot (its -x).
    gx, gy = np.meshgrid(np.arange(-30.0, -2.0, 0.25), np.arange(-6, 6, 0.25))
    rec = np.zeros(gx.size, dtype=RECORD)
    rec["x"], rec["y"], rec["z"] = gx.ravel(), gy.ravel(), -1.114
    scan_us = [T0 + 100_000 * k for k in range(N_SCANS)]
    for us in scan_us:
        rec["t"] = us + np.arange(len(rec)) % 1000
        rec.tofile(traj / "robosense" / f"{us}.bin")
    with ChannelWriter(traj, config.POSES, loader="npys", timestamps_from="robosense", sources=["robosense"]) as w:
        for k, us in enumerate(scan_us):
            T = np.eye(4)
            T[0, 3] = -distance((us - T0) * 1e-6)              # forward = lidar -x
            w.add(T, stem=f"{k:06d}", timestamp=us * 1e-6)
    cam_us = [us + CAM_OFFSET_US for us in scan_us]
    for k, us in enumerate(cam_us):
        rgba = np.full((1200, 1920, 4), 255, np.uint8)   # the real ZED X resolution, RGBA like its PNGs
        rgba[..., :3] = 6 * k
        Image.fromarray(rgba).save(traj / camera / f"{us}.png", compress_level=1)
    # ~50 Hz commands, linear part = seconds since T0 (so "previous" is checkable).
    t = np.arange(T0 - 500_000, scan_us[-1] + 500_000, 20_000)
    lines = ["timestamp,lx,az\n"] + [f"{us},{(us - T0) * 1e-6:.6f},{0.001 * i:.6f}\n" for i, us in enumerate(t)]
    (traj / "metadata" / "cmd_velocity.csv").write_text("".join(lines))
    return cam_us


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    traj = tmp_path_factory.mktemp("fomo_e2e") / "2024-11-21" / "red_synth"
    cam_us = _trajectory(traj)
    label_scans(traj)
    written = label_images(traj)
    return traj, cam_us, written


def test_every_row_is_the_instant_of_its_image(labelled, monkeypatch):
    monkeypatch.setattr(export, "MIN_FUTURE_PATH", MIN_FUTURE_PATH)
    traj, cam_us, written = labelled
    assert written == N_SCANS - 1          # the last image is after the last pose: no pose to interpolate
    rows = list(export.frames(traj))
    t_cam = (np.asarray(cam_us[:written]) - T0) * 1e-6
    ahead = distance((N_SCANS - 1) * 0.1) - distance(t_cam)   # path left after each image
    assert len(rows) == int((ahead >= MIN_FUTURE_PATH).sum()) > 5
    for r, f in enumerate(rows):
        k = int(round(np.median(f["observation.images.front"]) / 6))
        assert k == r
        # base_link (+x forward), relative to the first exported frame, interpolated at the image.
        np.testing.assert_allclose(f["observation.state"][:3], [distance(t_cam[k]) - distance(t_cam[0]), 0, 0],
                                   atol=0.05)
        np.testing.assert_allclose(f["observation.state"][3:], [0, 0, 0, 1], atol=1e-6)
        # The command in force: the last one at or before the image (50 Hz).
        assert -0.0201 <= f["action"][0] - t_cam[k] <= 1e-4
    assert {0, 1} <= set(np.unique(rows[0]["observation.traversability_mask"]).tolist())


def test_lerobot_round_trip(labelled, monkeypatch, tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(export, "MIN_FUTURE_PATH", MIN_FUTURE_PATH)
    traj, _, _ = labelled
    expected = list(export.frames(traj))
    export.to_lerobot([traj], "test/fomo", root=tmp_path / "lr")
    data = LeRobotDataset("test/fomo", root=tmp_path / "lr")
    assert data.num_episodes == 1 and data.num_frames == len(expected)
    for i in (0, len(expected) - 1):
        np.testing.assert_allclose(data[i]["observation.state"].numpy(), expected[i]["observation.state"])
        np.testing.assert_allclose(data[i]["action"].numpy(), expected[i]["action"])
    assert data[0]["observation.images.front"].shape == (3, 600, 960)
