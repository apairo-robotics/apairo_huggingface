import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from apairo.testing import check_format
from apairo_huggingface.rellis import config
from apairo_huggingface.rellis.agreement import scores, sequence_confusion
from apairo_huggingface.rellis.calibration import camera_from_lidar, intrinsics, lidar_from_camera
from apairo_huggingface.rellis.export import LIDAR_FROM_ROBOT, lidar_splits
from apairo_huggingface.rellis.format import SemKittiLabelFormat, SemKittiLabelLoader
from apairo_huggingface.rellis.labeling import corridor_world, make_view

RELLIS_ROOT = Path(os.environ.get("RELLIS_ROOT", "/mnt/vault-fellowship/rellis"))


# ------------------------------------------------------------------ format

def _labels(tmp_path, frames=3, n=6):
    for k in range(frames):
        raw = (np.arange(n, dtype=np.uint32) % 5) | (np.uint32(7) << 16)  # instance id 7 in the high bits
        raw.astype("<u4").tofile(tmp_path / f"{k:06d}.label")
    return tmp_path


def test_label_format_follows_the_contract(tmp_path):
    check_format(SemKittiLabelFormat(), _labels(tmp_path))


def test_label_keeps_the_semantic_low_16_bits(tmp_path):
    frame = SemKittiLabelLoader(str(_labels(tmp_path)))[0]
    assert frame.dtype == np.int64
    np.testing.assert_array_equal(frame, np.arange(6) % 5)


def test_truncated_label_is_refused(tmp_path):
    loader = SemKittiLabelLoader(str(_labels(tmp_path)))
    path = tmp_path / loader.files[0]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="multiple of 4"):
        loader[0]


def test_entry_point_makes_the_format_available():
    from apairo.core.formats import get_format

    assert get_format("semkitti_label").name == "semkitti_label"


# ------------------------------------------------------------- calibration

def _calib_tree(tmp_path):
    seq = "00000"
    (tmp_path / "Rellis_3D" / seq).mkdir(parents=True)
    (tmp_path / "Rellis-3D" / seq).mkdir(parents=True)
    # The real 00000 entry: camera pose in the Ouster frame.
    entry = {"os1_cloud_node-pylon_camera_node": {
        "q": {"w": -0.50507811, "x": 0.51206185, "y": 0.49024953, "z": -0.49228464},
        "t": {"x": -0.13165462, "y": 0.03870398, "z": -0.17253834}}}
    (tmp_path / "Rellis_3D" / seq / "transforms.yaml").write_text(yaml.safe_dump(entry))
    (tmp_path / "Rellis-3D" / seq / "camera_info.txt").write_text("2813.643275 2808.326079 969.285772 624.049972\n")
    return tmp_path, seq


def test_extrinsic_is_the_inverse_of_the_stored_transform(tmp_path):
    root, seq = _calib_tree(tmp_path)
    np.testing.assert_allclose(camera_from_lidar(root, seq) @ lidar_from_camera(root, seq), np.eye(4), atol=1e-12)


def test_camera_looks_along_the_robot_forward_axis(tmp_path):
    # The robot drives along the Ouster's -x: a point 10 m ahead of the robot
    # must be in front of the camera, near the image centre.
    root, seq = _calib_tree(tmp_path)
    K = intrinsics(root, seq)
    p = camera_from_lidar(root, seq) @ np.array([-10.0, 0.0, 0.0, 1.0])
    assert p[2] > 9.0
    u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
    assert abs(u - K[0, 2]) < 150 and 0 < v < 1200


# ---------------------------------------------------------------- labeling

def test_corridor_of_a_straight_path_is_the_robot_width():
    future = np.column_stack([np.linspace(0, 20, 50), np.zeros(50), np.full(50, 1.0)])
    left, right = corridor_world(future)
    np.testing.assert_allclose(left[:, 1], config.ROBOT_HALF_WIDTH)
    np.testing.assert_allclose(right[:, 1], -config.ROBOT_HALF_WIDTH)
    np.testing.assert_allclose(left[:, 2], 1.0 + config.GROUND_OFFSET)


def test_view_links_the_source_and_writes_nothing_there(tmp_path):
    src = tmp_path / "rellis" / "Rellis-3D" / "00000"
    for name in config.SOURCE_DIRS:
        (src / name).mkdir(parents=True)
    before = sorted(p.relative_to(tmp_path) for p in (tmp_path / "rellis").rglob("*"))
    view = make_view(tmp_path / "rellis", "00000", tmp_path / "out")
    make_view(tmp_path / "rellis", "00000", tmp_path / "out")  # idempotent
    assert all((view / name).is_symlink() for name in config.SOURCE_DIRS)
    assert sorted(p.relative_to(tmp_path) for p in (tmp_path / "rellis").rglob("*")) == before


# --------------------------------------------------------------- agreement

def test_scores_on_a_hand_made_confusion():
    cm = np.zeros((256, 256), dtype=np.int64)
    cm[3, 3], cm[3, 19], cm[19, 19] = 80, 20, 50   # [projected, 2D]
    r = scores(cm)
    assert r["pixel_accuracy"] == pytest.approx(130 / 150)
    assert r["per_class"]["grass"]["iou"] == pytest.approx(80 / 100)
    assert r["per_class"]["bush"]["agreement"] == pytest.approx(50 / 70)
    assert r["per_class"]["bush"]["precision"] == pytest.approx(1.0)


def test_confusion_ignores_no_data_and_void(tmp_path):
    view = tmp_path / "00000"
    (view / config.LABEL_2D).mkdir(parents=True)
    (view / config.SEM_MASK).mkdir()
    sem = np.array([[3, 3, 255, 0]], dtype=np.uint8)
    ref = np.array([[3, 19, 3, 3]], dtype=np.uint8)
    Image.fromarray(sem).save(view / config.SEM_MASK / "f.png")
    Image.fromarray(ref).save(view / config.LABEL_2D / "f.png")
    cm = sequence_confusion(view, {"00000/f": "test"})
    assert cm["all"].sum() == 2 and cm["test"][3, 3] == 1 and cm["test"][3, 19] == 1


# ------------------------------------------------------------------ export

def test_lidar_splits_parse_the_official_lists(tmp_path):
    for name, idx in (("train", 0), ("val", 1), ("test", 2)):
        (tmp_path / f"pt_{name}.lst").write_text(
            f"00000/os1_cloud_node_kitti_bin/00000{idx}.bin 00000/os1_cloud_node_semantickitti_label_id/00000{idx}.label\n")
    assert lidar_splits(tmp_path) == {("00000", 0): 0, ("00000", 1): 1, ("00000", 2): 2}


def test_robot_frame_turns_the_lidar_x_forward():
    # Moving along the lidar's -x is moving forward in the robot frame.
    lidar_step = np.eye(4)
    lidar_step[0, 3] = -1.0
    robot_step = np.linalg.inv(LIDAR_FROM_ROBOT) @ lidar_step @ LIDAR_FROM_ROBOT
    assert robot_step[0, 3] == pytest.approx(1.0)


# --------------------------------------------------- real data, if present

@pytest.mark.skipif(not (RELLIS_ROOT / "Rellis-3D" / "00003").is_dir(), reason="RELLIS-3D not available")
def test_masks_on_real_frames(tmp_path):
    from apairo_huggingface.rellis.labeling import label_sequence

    report = label_sequence(RELLIS_ROOT, "00003", tmp_path, limit=4)
    assert report["images"] == 4
    for name, allowed in ((config.SEM_MASK, set(config.ONTOLOGY) | {255}), (config.TRAV_MASK, {0, 1, 255})):
        files = sorted((tmp_path / "00003" / name).glob("*.png"))
        assert len(files) == 4
        for f in files:
            m = np.asarray(Image.open(f))
            assert m.shape == (1200, 1920) and m.dtype == np.uint8
            assert set(np.unique(m)) <= allowed and 255 in set(np.unique(m))


# ------------------------------------------------- review regressions

def _image_tree(tmp_path, n=4, size=(8, 6)):
    """A tiny RELLIS-3D sequence: n images, scans, labels, poses; returns (root, view)."""
    root, seq = tmp_path / "rellis", "00000"
    src = root / "Rellis-3D" / seq
    for name in config.SOURCE_DIRS:
        (src / name).mkdir(parents=True)
    for k in range(n):
        stem = f"frame{k:06d}-1581624652_{100 * k:03d}"
        Image.fromarray(np.full((size[1], size[0], 3), 10 * k, np.uint8)).save(src / config.CAMERA / f"{stem}.jpg")
        np.zeros((5, 4), np.float32).tofile(src / "os1_cloud_node_kitti_bin" / f"{k:06d}.bin")
        np.zeros(5, "<u4").tofile(src / "os1_cloud_node_semantickitti_label_id" / f"{k:06d}.label")
    poses = np.tile(np.eye(4)[:3].reshape(1, 12), (n, 1))
    poses[:, 3] = -np.arange(n)                     # 1 m per frame along the lidar -x
    np.savetxt(src / "poses.txt", poses)
    (root / "pt_train.lst").write_text("".join(f"{seq}/os1_cloud_node_kitti_bin/{k:06d}.bin x\n" for k in range(n)))
    (root / "pt_val.lst").write_text("")
    (root / "pt_test.lst").write_text("")
    return root, make_view(root, seq, tmp_path / "out")


def _write_masks(view, indices):
    from apairo import RawDataset

    files = sorted((view / config.CAMERA).glob("*.jpg"))
    stamps = []
    for name in (config.SEM_MASK, config.TRAV_MASK):
        (view / name).mkdir()
    for i in indices:
        stem = files[i].stem
        Image.fromarray(np.full((6, 8), 3, np.uint8)).save(view / config.SEM_MASK / f"{stem}.png")
        Image.fromarray(np.full((6, 8), 1, np.uint8)).save(view / config.TRAV_MASK / f"{stem}.png")
        s, ms = stem.split("-")[1].split("_")
        stamps.append(int(s) + int(ms) / 1000)
    for name in (config.SEM_MASK, config.TRAV_MASK):
        np.savetxt(view / name / "timestamps.txt", stamps, fmt="%.3f")
        RawDataset.register_channel(view, name, "img", sources=[config.LIDAR, config.LABELS, config.CAMERA])


def test_export_refuses_images_without_masks(tmp_path):
    # process --limit 2 then export: frames 2 and 3 must not vanish silently.
    from apairo_huggingface.rellis.export import frames

    root, view = _image_tree(tmp_path)
    _write_masks(view, [0, 1])
    with pytest.raises(ValueError, match="2 of the first 4 images"):
        list(frames(root, view, lidar_splits(root)))
    assert [f["observation.frame_info"][1] for f in frames(root, view, lidar_splits(root), limit=2)] == [0, 1]


def test_export_keys_frames_by_image_not_by_view_position(tmp_path):
    # A hole in the masks used to shift every later frame onto the wrong image index.
    from apairo_huggingface.rellis.export import frames

    root, view = _image_tree(tmp_path)
    _write_masks(view, [0, 2, 3])
    with pytest.raises(ValueError, match="first: frame000001"):
        list(frames(root, view, lidar_splits(root)))
    out = list(frames(root, view, lidar_splits(root), limit=1))
    assert len(out) == 1 and out[0]["observation.frame_info"][1] == 0


def test_export_full_masks_give_every_frame_its_own_index(tmp_path):
    from apairo_huggingface.rellis.export import frames

    root, view = _image_tree(tmp_path)
    _write_masks(view, range(4))
    out = list(frames(root, view, lidar_splits(root)))
    assert [int(f["observation.frame_info"][1]) for f in out] == [0, 1, 2, 3]
    assert [int(f["observation.images.front"][0, 0, 0]) for f in out] == [0, 10, 20, 30]
    assert out[1]["action"][0] == pytest.approx(10.0, rel=1e-3)   # 1 m in 0.1 s, forward


def test_coverage_columns_share_the_same_frames(tmp_path):
    # A frame without a trav mask (all 255) must not dilute "with data" only.
    from apairo_huggingface.rellis.labeling import coverage

    view = tmp_path / "00000"
    for name in (config.SEM_MASK, config.TRAV_MASK):
        (view / name).mkdir(parents=True)
    no_trav = np.full((2, 4), 255, np.uint8)
    trav = np.array([[1, 1, 0, 255], [1, 1, 0, 255]], np.uint8)    # 3/4 with data, 1/2 driven
    sem = np.array([[3, 255, 255, 255], [3, 255, 255, 255]], np.uint8)
    for stem, tm in (("a", no_trav), ("b", trav)):
        Image.fromarray(tm).save(view / config.TRAV_MASK / f"{stem}.png")
        Image.fromarray(sem).save(view / config.SEM_MASK / f"{stem}.png")
    c = coverage(view)
    assert c["images"] == 2 and c["trav_mask_frames_with_data"] == 1
    assert c["sem_mask_pixel_coverage_mean"] == pytest.approx(0.25)
    assert c["trav_mask_pixel_coverage_mean"] == pytest.approx(0.75)
    assert c["trav_mask_driven_fraction_mean"] == pytest.approx(0.5)
    assert c["trav_mask_driven_fraction_mean"] <= c["trav_mask_pixel_coverage_mean"]


def test_calib_check_reads_png_annotations_only(tmp_path):
    from apairo_huggingface.rellis.calibcheck import annotations

    labels = tmp_path / "Rellis-3D" / "00000" / config.LABEL_2D
    labels.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(labels / "frame001493-1581624802_049.png")
    Image.fromarray(np.zeros((2, 2, 4), np.uint8)).save(labels / "frame001493-1581624802_049.jpg", format="PNG")
    assert [p.name for p in annotations(tmp_path, "00000")] == ["frame001493-1581624802_049.png"]


@pytest.mark.skipif(not (RELLIS_ROOT / "Rellis-3D" / "00000").is_dir(), reason="RELLIS-3D not available")
def test_calib_check_on_real_frames_scores_scan_offsets():
    from apairo_huggingface.rellis.calibcheck import SCAN_OFFSETS, annotations, calibration_check

    assert not any(p.suffix != ".png" for p in annotations(RELLIS_ROOT, "00000"))
    r = calibration_check(RELLIS_ROOT, ["00000"], frames_per_seq=2)["00000"]
    offsets = r["chosen, scan i+d on image i: d -> agreement"]
    assert sorted(offsets) == SCAN_OFFSETS
    assert offsets[0] == pytest.approx(r["chosen: inv(transforms.yaml), no distortion"], abs=1e-4)


@pytest.mark.skipif(not (RELLIS_ROOT / "Rellis-3D" / "00000").is_dir(), reason="RELLIS-3D not available")
def test_calib_check_on_real_frames_scores_k1_and_iris_variants():
    from apairo_huggingface.rellis.calibcheck import K1_SWEEP, calibration_check

    r = calibration_check(RELLIS_ROOT, ["00000"], frames_per_seq=2)
    chosen = r["00000"]["chosen: inv(transforms.yaml), no distortion"]
    assert all(f"inv(transforms.yaml), k1 = {k1:.2f}" in r["00000"] for k1 in K1_SWEEP)
    assert r["00000"]["inv(transforms.yaml), k1 = 0.00"] == pytest.approx(chosen)
    if (RELLIS_ROOT / "raw_data" / "calibrationdata_pylon_iris5.6.zip").exists():
        assert "iris5.6 camera matrix, no distortion" in r["00000"]
    summary = r["all"]["chosen: inv(transforms.yaml), no distortion"]
    assert summary["frames"] == 2 and summary["min"] <= summary["mean"] <= summary["max"]
    assert summary["mean"] == pytest.approx(chosen, abs=1e-4)
