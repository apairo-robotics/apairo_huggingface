import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from apairo_huggingface.sim_amiad import config
from apairo_huggingface.sim_amiad.agreement import cloud_agreement_scores, scores
from apairo_huggingface.sim_amiad.extract import (
    check_classes_json,
    decode_points_semantic,
    depth_to_mm,
    pose7_to_matrix,
    runtime_lut,
)
from apairo_huggingface.sim_amiad.export import export_fps, nearest_within
from apairo_huggingface.sim_amiad.labeling import camera_project, corridor_mask, self_cells, self_mask, visible_labels

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


def test_missing_rosbags_names_the_sim_extra(monkeypatch):
    """rosbags is the optional [sim] extra: without it, extract says how to get it."""
    import sys

    from apairo_huggingface.sim_amiad import extract

    monkeypatch.setitem(sys.modules, "rosbags.highlevel", None)    # import now raises ImportError
    with pytest.raises(ImportError, match=r"apairo-huggingface\[sim\]"):
        extract.extract("missing.mcap", "/nonexistent/out")


def test_rosbags_is_not_a_core_dependency():
    import tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    assert not any(d.startswith("rosbags") for d in project["dependencies"])
    assert any(d.startswith("rosbags") for d in project["optional-dependencies"]["sim"])


def test_classes_json_drift_is_refused(tmp_path):
    data = {"name_to_id": {**config.NAME_TO_ID, "ground": 9}, "classes": []}
    (tmp_path / "c.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="name_to_id"):
        check_classes_json(tmp_path / "c.json")


# ---------------------------------------------------------------- labeling

def test_depth_beyond_the_uint16_range_is_no_return_not_saturated():
    """A depth past 65.534 m does not fit in uint16 mm: it is stored as 0 = no
    return, never clipped to 65535, which the occlusion test would read as a
    surface at 65.535 m."""
    d = np.array([[1.2345, 65.534, 65.5355, 80.0], [np.inf, np.nan, -0.5, 0.0]], np.float32)
    mm, out_of_range = depth_to_mm(d)
    assert mm.dtype == np.uint16
    np.testing.assert_array_equal(mm, [[1234, 65534, 0, 0], [0, 0, 0, 0]])
    assert out_of_range == 2
    # a lidar point on the far surface is not labelled through a saturated pixel
    uv = np.array([[3.0, 0.0, 80.0], [2.0, 0.0, 65.6]])
    labels, visible = visible_labels(uv, np.full((2, 4), 3, np.uint8), mm)
    assert not visible.any()


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


def _distorted_camera():
    from apairo.core.config import CameraIntrinsics

    K = np.array([[500.0, 1.5, 640.0], [0.0, 480.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraIntrinsics(K=K, distortion=np.array([-0.3, 0.1, 0.002, -0.001, 0.0]), width=1280, height=720)


def test_corridor_and_lidar_share_one_camera_model():
    """The corridor quads and the lidar points go through the same K (with
    skew) and plumb-bob distortion: the corridor projection was pinhole only."""
    from apairo.core.sample import Sample
    from apairo_preprocess import LidarCameraProjection

    cam = _distorted_camera()
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-8, 8, 500), rng.uniform(-4, 4, 500), rng.uniform(1, 30, 500)])
    uv = LidarCameraProjection(intrinsics=cam, extrinsics=np.eye(4))(Sample(data={"lidar": pts}))
    ok = np.isfinite(uv[:, 0])
    assert ok.sum() > 300
    u, v, z = camera_project(pts, cam)
    np.testing.assert_allclose(np.column_stack([u, v, z])[ok], uv[ok], atol=1e-3)
    pinhole_u = cam.K[0, 0] * pts[:, 0] / pts[:, 2] + cam.K[0, 2]
    assert np.abs(pinhole_u[ok] - u[ok]).max() > 50        # the distortion matters at this strength


def test_corridor_mask_lands_where_the_lidar_projection_does():
    """A small ground quad 6 m ahead and 3 m to the side: the mask covers the
    pixel the lidar projection gives its centre, not the pinhole pixel."""
    from apairo.core.sample import Sample
    from apairo_preprocess import LidarCameraProjection

    cam = _distorted_camera()
    left = np.array([[2.8, 1.0, 5.8], [2.8, 1.0, 6.2]])     # camera optical frame = world here
    right = np.array([[3.2, 1.0, 5.8], [3.2, 1.0, 6.2]])
    mask = corridor_mask(left, right, np.eye(4), cam, np.zeros((720, 1280), np.uint16))
    centre = np.array([[3.0, 1.0, 6.0]])
    uv = LidarCameraProjection(intrinsics=cam, extrinsics=np.eye(4))(Sample(data={"lidar": centre}))
    assert mask[int(uv[0, 1]), int(uv[0, 0])]
    pin_u = int(cam.K[0, 0] * 3.0 / 6.0 + cam.K[0, 1] * 1.0 / 6.0 + cam.K[0, 2])
    pin_v = int(cam.K[1, 1] * 1.0 / 6.0 + cam.K[1, 2])
    assert not mask[pin_v, pin_u]


# ------------------------------------------------------------------ export

def test_export_fps_is_measured_from_the_stamps():
    t = 35.0 + np.arange(400) / 60.0
    assert export_fps(t, stride=4) == 15
    assert export_fps(t, stride=6) == 10


def test_export_fps_refuses_a_missing_scan():
    t = np.delete(35.0 + np.arange(400) / 60.0, 101)          # one scan dropped: a 2-tick interval
    with pytest.raises(ValueError, match="irregular"):
        export_fps(t, stride=4)


def test_export_fps_refuses_a_fractional_rate():
    with pytest.raises(ValueError, match="whole number"):
        export_fps(np.arange(100) * 0.07, stride=1)


def test_export_matches_images_with_the_labelling_tolerance():
    """The export matched images by exact microsecond equality while the
    labelling accepts config.SYNC_TOLERANCE: stamps one rounding apart, or a
    few ms apart, were refused by the export only."""
    ts = np.array([10.0, 10.0 + 1 / 60, 10.0 + 2 / 60])
    assert nearest_within(ts, 10.0000005001, "rgb") == 0     # rounds to a different microsecond
    assert nearest_within(ts, ts[1] + 0.9 * config.SYNC_TOLERANCE, "rgb") == 1
    with pytest.raises(ValueError, match="no rgb"):
        nearest_within(ts, ts[1] + 1.1 * config.SYNC_TOLERANCE, "rgb")


def test_one_sync_tolerance_for_labelling_and_export():
    from apairo_huggingface.sim_amiad import export, labeling

    assert not hasattr(labeling, "SYNC_TOLERANCE") and not hasattr(export, "SYNC_TOLERANCE")
    assert 0 < config.SYNC_TOLERANCE < 0.5 / 60                # unambiguous on the 60 Hz tick


# --------------------------------------------------------------- agreement

def test_cloud_agreement_leaves_unlabeled_out_of_the_headline():
    ground, tree, veg = config.NAME_TO_ID["ground"], config.NAME_TO_ID["tree"], config.NAME_TO_ID["vegetation"]
    cm = np.zeros((256, 256), dtype=np.int64)               # [ours, sim]
    cm[ground, ground], cm[veg, veg], cm[ground, veg] = 60, 30, 10
    cm[tree, 0] = 20                                        # sim says unlabeled
    cm[0, 0] = 50                                           # both unlabeled
    cm[veg, config.NO_DATA] = 7
    s = cloud_agreement_scores(cm)
    assert s["both_labeled"] == 100 and s["agreement"] == pytest.approx(0.9)
    assert s["both_labeled_incl_unlabeled"] == 170
    assert s["agreement_incl_unlabeled"] == pytest.approx(140 / 170)
    assert s["ours_class_sim_unlabeled"] == 20 and s["ours_unlabeled_sim_class"] == 0
    assert s["ours_labeled_sim_not"] == 7


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
