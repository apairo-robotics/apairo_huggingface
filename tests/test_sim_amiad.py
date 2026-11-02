import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from apairo_huggingface.sim_amiad import config
from apairo_huggingface.sim_amiad.agreement import scores
from apairo_huggingface.sim_amiad.extract import (
    check_classes_json,
    decode_points_semantic,
    pose7_to_matrix,
    runtime_lut,
)
from apairo_huggingface.sim_amiad.labeling import self_cells, self_mask, visible_labels

SEGMENT = Path(os.environ.get("SIM_AMIAD_SEGMENT", Path.home() / "data/sim_amiad/seg_t035_t065"))
needs_segment = pytest.mark.skipif(not (SEGMENT / "point_sem").is_dir(), reason="no labelled sim segment")


# ----------------------------------------------------------------- extract

def test_runtime_ids_are_mapped_by_name_not_by_number():
    labels = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}, "2": {"class": "tree"},
              "3": {"class": "vegetation"}, "4": {"class": "ground"}, "time_stamp": {"sec": 1, "nanosec": 0}}
    unknown = {}
    lut = runtime_lut(labels, unknown)
    np.testing.assert_array_equal(lut, [0, 0, 3, 4, 1])     # runtime 4 = ground = training id 1
    assert unknown == {"BACKGROUND": 1, "UNLABELLED": 1}


def test_points_semantic_colours_decode_to_class_ids():
    xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    colours = [config.CLASSES[1][1], config.CLASSES[3][1], config.POINTS_SEMANTIC_NO_CAMERA_RGB]
    packed = np.array([(r << 16) | (g << 8) | b for r, g, b in colours], dtype=np.uint32).view(np.float32)
    msg = SimpleNamespace(data=np.column_stack([xyz, packed]).astype(np.float32).tobytes(), point_step=16)
    out = decode_points_semantic(msg)
    np.testing.assert_array_equal(out[:, :3], xyz)
    np.testing.assert_array_equal(out[:, 3], [1, 3, config.NO_DATA])


def test_pose7_quaternion_order_is_xyzw():
    M = pose7_to_matrix([1, 2, 3, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])   # 90 deg about z
    np.testing.assert_allclose(M[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(M[:3, 3], [1, 2, 3])


def test_classes_json_matches_the_packaged_table():
    path = Path(config.CLASSES_JSON)
    if not path.exists():
        pytest.skip("share not mounted")
    check_classes_json(path)


def test_classes_json_drift_is_refused(tmp_path):
    data = {"name_to_id": {**config.NAME_TO_ID, "ground": 9}, "classes": []}
    (tmp_path / "c.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="name_to_id"):
        check_classes_json(tmp_path / "c.json")


# ---------------------------------------------------------------- labeling

def test_only_points_matching_the_depth_image_get_a_label():
    semantic = np.full((4, 4), 3, dtype=np.uint8)
    depth_mm = np.full((4, 4), 5000, dtype=np.uint16)
    depth_mm[0, 0] = 0                                      # sky
    uv = np.array([[1.5, 1.5, 5.05],                        # visible
                   [2.5, 2.5, 9.0],                         # hidden behind the 5 m surface
                   [0.2, 0.2, 5.0],                         # on a sky pixel
                   [np.nan, np.nan, np.nan]], dtype=np.float32)  # outside the image
    labels, visible = visible_labels(uv, semantic, depth_mm)
    np.testing.assert_array_equal(labels, [3, 255, 255, 255])
    np.testing.assert_array_equal(visible, [True, False, False, False])


def test_self_filter_keeps_the_fixed_ground_pattern():
    # The same two sensor-frame points in every scan: one on the robot body,
    # one on flat ground just ahead. Both persist; only the body is removed.
    base_from_lidar = np.eye(4)
    fixed = np.array([[0.5, 0.0, 0.2], [2.0, 0.0, -0.42]])
    rng = np.random.default_rng(0)
    scans = [np.vstack([fixed, rng.uniform(-20, 20, (50, 3))]) for _ in range(20)]
    mask = self_mask(scans[0], self_cells(scans), base_from_lidar)
    assert mask[0] and not mask[1]
    assert not mask[2:].any()


# --------------------------------------------------------------- agreement

def test_pu_scores_from_a_confusion():
    cm = np.zeros((2, 256), dtype=np.int64)
    ground, veg, tree = config.NAME_TO_ID["ground"], config.NAME_TO_ID["vegetation"], config.NAME_TO_ID["tree"]
    cm[1, ground], cm[1, veg] = 80, 20       # positives
    cm[0, ground], cm[0, tree] = 30, 70      # unlabeled
    cm[0, 0] = 1000                          # class 0 is never scored
    s = scores(cm, [ground])
    assert s["points"] == 200 and s["positives"] == 100
    assert s["positive_precision"] == pytest.approx(0.8)
    assert s["positive_recall"] == pytest.approx(80 / 110)
    assert s["unlabeled_positive_rate"] == pytest.approx(0.3)
    assert s["prior"] == pytest.approx(110 / 200)


# ------------------------------------------------- real segment (optional)

@needs_segment
def test_lidar_depth_matches_the_depth_image_in_the_optical_frame():
    """The camera header frame 'ZED2' is not optical; 'ZED2_pcl' is."""
    from apairo import RawDataset
    from apairo.core.sample import Sample
    from apairo_preprocess import LidarCameraProjection

    cal = RawDataset(SEGMENT, keys=[config.LIDAR]).calibration
    k = int(np.load(SEGMENT / "scored_frames.npy")[0])
    pts = np.load(SEGMENT / config.LIDAR / f"{k:06d}.npy")
    depth = np.asarray(Image.open(SEGMENT / config.DEPTH / f"{k:06d}.png")) / 1000.0
    errors = {}
    for frame in (config.CAMERA_FRAME, config.CAMERA_HEADER_FRAME):
        proj = LidarCameraProjection(intrinsics=cal.get_intrinsics(config.CAMERA_FRAME),
                                     extrinsics=cal.get_tf(config.LIDAR_FRAME, frame),
                                     image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
        uv = proj(Sample(data={"lidar": pts}))
        ok = np.isfinite(uv[:, 0])
        d = depth[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        good = d > 0
        errors[frame] = float(np.median(np.abs(d[good] - uv[ok, 2][good])))
    assert errors[config.CAMERA_FRAME] < 0.1
    assert errors[config.CAMERA_HEADER_FRAME] > 1.0


@needs_segment
def test_derived_files_are_row_aligned_with_their_scan():
    for k in np.load(SEGMENT / "scored_frames.npy")[:5]:
        n = len(np.load(SEGMENT / config.LIDAR / f"{k:06d}.npy"))
        assert len(np.load(SEGMENT / "point_sem" / f"{k:06d}.npy")) == n
        assert len(np.load(SEGMENT / "trav_traj" / f"{k:06d}.npy")) == n


@needs_segment
def test_export_frames_have_the_declared_shapes():
    from apairo_huggingface.sim_amiad.export import features, frames

    frame = next(frames(SEGMENT, limit=1))
    feats = features(*frame["observation.images.front"].shape[:2])
    for key, spec in feats.items():
        assert tuple(np.asarray(frame[key]).shape) == tuple(spec["shape"]), key
    assert set(np.unique(frame["observation.semantic_mask"])) <= set(config.CLASSES)
