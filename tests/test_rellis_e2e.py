"""RELLIS-3D end to end on a synthetic sequence: label_sequence -> frames ->
to_lerobot -> reload. Every frame carries content that identifies it (image
intensity, scan class, 2D label, split), so a frame matched to the wrong image,
scan, pose or split shows as a mismatch, not only as a wrong count."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from PIL import Image

from apairo_huggingface.rellis import config
from apairo_huggingface.rellis.export import check, frames, lidar_splits, to_lerobot
from apairo_huggingface.rellis.labeling import label_sequence

SEQ = "00000"
N = 10
IDS = [1, 3, 4, 5, 6, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34]  # RELLIS class ids


def distance(k):
    """Metres driven at frame ``k`` (0.1 s apart): accelerating, so that the first
    frames have the 10 m of path ahead the trav mask needs (``MIN_FUTURE_PATH``)."""
    return 0.2 * k * k


def _rellis_tree(root, n=N, t0=1581624652):
    """A RELLIS-3D source tree, real 00000 calibration, at the real camera resolution
    (the intrinsics are the real ones); scan / label / image ``k`` identify ``k``."""
    src = root / "Rellis-3D" / SEQ
    for name in config.SOURCE_DIRS:
        (src / name).mkdir(parents=True)
    (root / "Rellis_3D" / SEQ).mkdir(parents=True)
    entry = {"os1_cloud_node-pylon_camera_node": {
        "q": {"w": -0.50507811, "x": 0.51206185, "y": 0.49024953, "z": -0.49228464},
        "t": {"x": -0.13165462, "y": 0.03870398, "z": -0.17253834}}}
    (root / "Rellis_3D" / SEQ / "transforms.yaml").write_text(yaml.safe_dump(entry))
    (src / "camera_info.txt").write_text("2813.643275 2808.326079 969.285772 624.049972\n")
    # Ground 1 m below the lidar, behind it (the Ouster's -x is the robot's forward).
    gx, gy = np.meshgrid(np.arange(-30.0, -3.0, 0.25), np.arange(-6, 6, 0.25))
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.0), np.zeros(gx.size)]).astype(np.float32)
    rows, lists = [], {"train": [], "val": [], "test": []}
    for k in range(n):
        ms = 100 * k
        stem = f"frame{k:06d}-{t0 + ms // 1000}_{ms % 1000:03d}"
        Image.fromarray(np.full((1200, 1920, 3), 8 * k, np.uint8)).save(src / config.CAMERA / f"{stem}.jpg", quality=95)
        ground.tofile(src / "os1_cloud_node_kitti_bin" / f"{k:06d}.bin")
        np.full(len(ground), IDS[k % len(IDS)], "<u4").tofile(src / "os1_cloud_node_semantickitti_label_id" / f"{k:06d}.label")
        if k % 2 == 0:
            Image.fromarray(np.full((1200, 1920), IDS[(k + 5) % len(IDS)], np.uint8)).save(src / config.LABEL_2D / f"{stem}.png")
        T = np.eye(4)[:3].copy()
        T[0, 3] = -distance(k)                        # forward = lidar -x
        rows.append(T.reshape(12))
        lists[("train", "val", "test")[k % 3]].append(f"{SEQ}/os1_cloud_node_kitti_bin/{k:06d}.bin {SEQ}/x/{k:06d}.label\n")
    np.savetxt(src / "poses.txt", np.array(rows))
    for name, lines in lists.items():
        (root / f"pt_{name}.lst").write_text("".join(lines))
    return root


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rellis_e2e")
    root = _rellis_tree(tmp / "rellis")
    report = label_sequence(root, SEQ, tmp / "out")
    return root, tmp / "out" / SEQ, report


def test_every_frame_is_its_own_image_scan_pose_and_split(labelled):
    root, view, report = labelled
    assert report["images"] == N
    rows = list(frames(root, view, lidar_splits(root)))
    assert len(rows) == N
    for r, f in enumerate(rows):
        k = int(f["observation.frame_info"][1])
        assert k == r
        assert abs(int(np.median(f["observation.images.front"])) - 8 * k) <= 1
        assert set(np.unique(f["observation.semantic_mask"]).tolist()) - {255} == {IDS[k % len(IDS)]}
        expected_2d = {IDS[(k + 5) % len(IDS)]} if k % 2 == 0 else {255}
        assert set(np.unique(f["observation.label_2d"]).tolist()) == expected_2d
        assert f["observation.frame_info"].tolist() == [k % 3, k, int(k % 2 == 0)]
        # Robot frame (+x forward), relative to the first frame; speed to the next frame.
        np.testing.assert_allclose(f["observation.state"], [distance(k), 0, 0, 0, 0, 0, 1], atol=1e-4)
        a = min(k, N - 2)
        np.testing.assert_allclose(f["action"], [(distance(a + 1) - distance(a)) / 0.1, 0.0], atol=1e-3)
    assert set(np.unique(rows[0]["observation.traversability_mask"]).tolist()) == {0, 1, 255}


def test_lerobot_round_trip(labelled, tmp_path):
    pytest.importorskip("lerobot")
    root, view, _ = labelled
    to_lerobot(root, [view], "test/rellis", lerobot_root=tmp_path / "lr")
    report = check("test/rellis", tmp_path / "lr")
    assert report["episodes"] == 1 and report["frames"] == N
    assert report["split_counts"] == {"train": 4, "val": 3, "test": 3}
    assert report["frames_with_label_2d"] == N // 2
    assert report["values"]["observation.traversability_mask"] == [0, 1, 255]
    assert report["item0_shapes"]["observation.images.front"] == (3, 600, 960)
