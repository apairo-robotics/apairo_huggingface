import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apairo.core.sample import Sample
from apairo_huggingface import bev_grid
from apairo_huggingface.tartan_bev import bev, config

TARTAN_ROOT = Path(os.environ.get("TARTAN_ROOT", "/mnt/vault-fellowship/tartan"))
H, W = bev_grid.grid_shape()


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------------ geometry

def test_level_frame_is_up_with_the_heading_levelled():
    pitch = np.array([[np.cos(0.17), 0, np.sin(0.17)], [0, 1, 0], [-np.sin(0.17), 0, np.cos(0.17)]])
    T = np.eye(4)
    T[:3, :3] = _rot_z(30) @ pitch
    T[:3, 3] = [5.0, -2.0, 3.0]
    B = bev_grid.level_frame(T, T[:3, 0], np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(B[:3, :3] @ B[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(B[:3, 2], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(B[:3, 0], [np.cos(np.radians(30)), np.sin(np.radians(30)), 0], atol=1e-12)
    np.testing.assert_allclose(B[:3, 3], T[:3, 3])
    # With up = world z it is the RELLIS frame.
    np.testing.assert_allclose(B, bev_grid.world_from_heading(T, T[:3, 0]), atol=1e-12)
    # Up need not be world z: the frame follows it.
    up = pitch @ [0.0, 0.0, 1.0]
    B = bev_grid.level_frame(T, np.array([1.0, 0.0, 0.0]), up)
    np.testing.assert_allclose(B[:3, 2], up, atol=1e-12)
    assert abs(B[:3, 0] @ up) < 1e-12 and np.linalg.det(B[:3, :3]) > 0


def test_geometry_roughness_step_and_nan_unknown():
    a, b, c = bev_grid.cell_index(np.array([[10.1, 0.1, 0], [10.1, 0.3, 0], [20.1, 0.1, 0]]))
    assert b == a - 1                               # neighbours on one row
    cells = np.array([a, a, b, b, c])
    heights = np.array([0.0, 0.2, 0.5, 0.5, 1.0])
    mean = np.full(H * W, np.nan)
    mean[[a, b, c]] = [0.1, 0.5, 1.0]
    g = bev.geometry(cells, heights, mean.reshape(H, W).astype(np.float16))
    assert g.shape == (2, H, W) and g.dtype == np.float16
    rough, step = g[0].ravel(), g[1].ravel()
    np.testing.assert_allclose([rough[a], rough[b]], [0.1, 0.0], atol=1e-3)
    assert np.isnan(rough[c])                       # a single point: no roughness
    np.testing.assert_allclose([step[a], step[b]], [0.4, 0.4], atol=1e-3)
    assert np.isnan(step[c])                        # no known neighbour
    assert np.isfinite(step).sum() == 2


# ------------------------------------------------------ synthetic recording

N_SCANS = 10
DUP = 3          # scan DUP repeats scan DUP - 1 under the same stamp, as TartanDrive does
T0 = 1694536776.0


def _ground_scan(rng, n=4000):
    # Flat ground LIDAR_HEIGHT below the lidar, vehicle frame (no points within 2 m).
    r = rng.uniform(2.5, 25, n)
    a = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, -config.LIDAR_HEIGHT)]).astype(np.float32)


def _synthetic_root(tmp_path, tilt_deg=0.0, invalid=(), attitude_gap=None):
    # The vehicle drives along world +y (its +x) at 2 m per scan on level
    # ground. ``tilt_deg`` tilts the GICP world about its x (as GICP's drifting
    # z does: the poses tilt, the scans and the INS attitude do not);
    # ``invalid`` scans are False in valid_mask.npy; ``attitude_gap`` = scan
    # index whose stamp gps_odom has no row within 0.15 s of.
    seq = tmp_path / "tartan" / "rec"
    for name in config.SOURCE_DIRS:
        (seq / name).mkdir(parents=True)
    rng = np.random.default_rng(1)
    c, s = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
    tilt = np.eye(4)
    tilt[1:3, 1:3] = [[c, -s], [s, c]]
    poses, stamps = [], []
    for k in range(N_SCANS):
        j = k - 1 if k == DUP else k
        T = np.eye(4)
        T[:3, :3] = _rot_z(90.0)
        T[:3, 3] = [0.0, 2.0 * j + (0.001 if k == DUP else 0.0), 0.0]
        poses.append(tilt @ T)
        stamps.append(T0 + 0.1 * j)
        scan = np.load(seq / config.LIDAR / f"{j:06d}.npy") if k == DUP else _ground_scan(rng)
        np.save(seq / config.LIDAR / f"{k:06d}.npy", scan)
        np.save(seq / config.LIDAR / f"{k:06d}_intensity.npy", np.zeros(len(scan), np.float32))
        Image.fromarray(np.full((68, 128, 3), 20 * k, np.uint8)).save(seq / config.CAMERA / f"{k:06d}.png")
    for name in (config.LIDAR, config.POSES, config.CAMERA):
        np.savetxt(seq / name / "timestamps.txt", np.array(stamps) + (0.003 if name == config.CAMERA else 0.0))
    np.save(seq / config.POSES / "poses.npy", np.stack(poses))
    valid = np.ones(N_SCANS, bool)
    valid[list(invalid)] = False
    np.save(seq / config.POSES / "valid_mask.npy", valid)
    # INS at 50 Hz: heading 90 deg, level (qx qy qz qw at columns 3:7).
    t = np.arange(stamps[0] - 0.5, stamps[-1] + 0.5, 0.02)
    if attitude_gap is not None:
        t = t[np.abs(t - stamps[attitude_gap]) > 0.15]
    rows = np.zeros((len(t), 13))
    rows[:, 3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    np.save(seq / config.ATTITUDE / "odometry.npy", rows)
    np.savetxt(seq / config.ATTITUDE / "timestamps.txt", t)
    np.save(seq / config.ACTION / "twist.npy", np.column_stack([np.zeros(N_SCANS), np.linspace(-1, 1, N_SCANS)]))
    np.savetxt(seq / config.ACTION / "timestamps.txt", np.array(stamps) + 0.01)
    return tmp_path / "tartan", seq


def test_view_writes_nothing_in_the_source(tmp_path):
    root, seq = _synthetic_root(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in seq.rglob("*"))
    view = bev.make_view(root, "rec", tmp_path / "out")
    bev.make_view(root, "rec", tmp_path / "out")            # idempotent
    bev.process_sequence(root, "rec", tmp_path / "out")
    assert sorted(p.relative_to(tmp_path) for p in seq.rglob("*")) == before
    ds = bev.open_view(view, [config.LIDAR, config.POSES, config.CAMERA, config.ACTION])
    assert all(len(ds.timestamps[k]) == N_SCANS for k in ds.timestamps)
    assert np.asarray(ds.loaders[config.LIDAR][0]).shape == (4000, 3)   # not the intensity file
    assert np.asarray(ds.loaders[config.POSES][0]).shape == (4, 4)       # poses.npy, not valid_mask.npy
    ds = bev.open_view(view)
    assert np.asarray(ds.loaders[config.POSE_VALID][0]).shape == ()      # valid_mask.npy, one bool per scan
    assert len(ds.timestamps[config.POSE_VALID]) == N_SCANS
    assert bev.valid_scans(ds).all()


def test_process_writes_three_registered_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_FUTURE_PATH", 4.0)
    root, _ = _synthetic_root(tmp_path)
    bev.process_sequence(root, "rec", tmp_path / "out")
    ds = bev.open_view(tmp_path / "out" / "rec", bev.OUTPUT_KEYS)
    assert all(len(ds.timestamps[k]) == N_SCANS for k in bev.OUTPUT_KEYS)
    elev, geom, trav = (ds.loaders[k][2] for k in bev.OUTPUT_KEYS)
    assert elev.shape == (3, H, W) and geom.shape == (2, H, W) and trav.dtype == np.uint8
    assert set(np.unique(trav)) == {0, 1, 255}
    # Flat ground LIDAR_HEIGHT below the lidar: height 0, no roughness, no step.
    assert abs(np.nanmedian(elev[2].astype(float))) < 0.01
    assert np.nanmax(geom.astype(float)) < 0.01
    # The driven corridor runs straight up the grid through the vehicle.
    assert trav[100, 100] == 1 and trav[100, 60] != 1 and trav[150, 100] == 1
    assert (ds.loaders[config.BEV_TRAV][N_SCANS - 1] == 255).all()   # < MIN_FUTURE_PATH ahead
    # The repeated scan shares its stamp with the original: the pose tells them apart.
    grids = bev.BevGrids(bev.open_view(tmp_path / "out" / "rec"))
    stamp = float(ds.timestamps[config.BEV_TRAV][DUP])
    assert stamp == float(ds.timestamps[config.BEV_TRAV][DUP - 1])
    assert grids.index_of(stamp, grids._poses[DUP]) == DUP and grids.index_of(stamp, grids._poses[DUP - 1]) == DUP - 1
    # Recipe reuse: a second run is a no-op.
    mtime = (tmp_path / "out" / "rec" / config.BEV_TRAV / "000002.npy").stat().st_mtime
    bev.process_sequence(root, "rec", tmp_path / "out")
    assert (tmp_path / "out" / "rec" / config.BEV_TRAV / "000002.npy").stat().st_mtime == mtime


def test_accumulation_is_past_only(tmp_path):
    root, seq = _synthetic_root(tmp_path)
    view = bev.make_view(root, "rec", tmp_path / "out")
    j = 6

    def grids():
        ds = bev.open_view(view)
        g = bev.BevGrids(ds, accumulate=3, stride=1)
        assert g.scans_used(j) == [6, 5, 4]
        s = Sample(data={k: ds.loaders[k][j] for k in bev.INPUT_KEYS}, timestamp=float(ds.timestamps[config.LIDAR][j]))
        return g(s)

    before = grids()
    for k in range(j + 1, N_SCANS):          # rewrite every future scan: the inputs must not move
        np.save(seq / config.LIDAR / f"{k:06d}.npy", (np.random.default_rng(k).normal(size=(4000, 3)) * 5).astype(np.float32))
    after = grids()
    np.testing.assert_array_equal(before[config.BEV_ELEVATION], after[config.BEV_ELEVATION])
    np.testing.assert_array_equal(before[config.BEV_GEOMETRY], after[config.BEV_GEOMETRY])
    shifted = _ground_scan(np.random.default_rng(7))
    shifted[:, 2] += 0.5
    np.save(seq / config.LIDAR / f"{j - 1:06d}.npy", shifted)   # and a past scan does move them
    moved = grids()
    assert not np.array_equal(np.nan_to_num(before[config.BEV_ELEVATION]), np.nan_to_num(moved[config.BEV_ELEVATION]))


def _grid_of(view, j, **kwargs):
    ds = bev.open_view(view)
    g = bev.BevGrids(ds, **kwargs)
    return g, g(Sample(data={k: ds.loaders[k][j] for k in bev.INPUT_KEYS},
                       timestamp=float(ds.timestamps[config.LIDAR][j])))


def test_grids_are_levelled_by_the_ins_not_the_gicp_world(tmp_path):
    # GICP's world z drifts from gravity (up to 26 deg on the real recordings).
    # A world tilted by 10 deg must not tilt the grid: level ground reads 0 at
    # 25 m ahead, where the GICP z would put it 25 * tan(10 deg) = 4.4 m off.
    root, _ = _synthetic_root(tmp_path, tilt_deg=10.0)
    _, out = _grid_of(bev.make_view(root, "rec", tmp_path / "out"), 6)
    mean = out[config.BEV_ELEVATION][2].astype(float)
    known = np.isfinite(mean)
    assert known.sum() > 5000
    assert np.abs(mean[known]).max() < 0.02
    assert np.nanmax(out[config.BEV_GEOMETRY].astype(float)) < 0.02


def test_invalid_poses_enter_no_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_FUTURE_PATH", 4.0)
    bad = 5
    root, _ = _synthetic_root(tmp_path, invalid=[bad])
    view = bev.make_view(root, "rec", tmp_path / "out")
    g, out = _grid_of(view, bad)
    assert np.isnan(out[config.BEV_ELEVATION].astype(float)).all()
    assert np.isnan(out[config.BEV_GEOMETRY].astype(float)).all()
    assert (out[config.BEV_TRAV] == 255).all()
    assert g.scans_used(6) == [6, 4, 2, 0] and bad not in g.scans_used(7)
    # The path skips the invalid pose: move it far away, nothing changes.
    poses = np.load(root / "rec" / config.POSES / "poses.npy")
    _, before = _grid_of(view, 6, stride=1)
    poses[bad, :3, 3] = [500.0, -300.0, 40.0]
    np.save(root / "rec" / config.POSES / "poses.npy", poses)
    g, after = _grid_of(view, 6, stride=1)
    for key in bev.OUTPUT_KEYS:
        np.testing.assert_array_equal(np.nan_to_num(before[key]), np.nan_to_num(after[key]))
    assert abs(g._path[-1] - 18.0) < 1e-2               # 0 -> 18 m along y, not via (500, -300)


def test_scans_without_ins_attitude_are_invalid(tmp_path):
    root, _ = _synthetic_root(tmp_path, attitude_gap=6)
    ds = bev.open_view(bev.make_view(root, "rec", tmp_path / "out"))
    up, has = bev.scan_up(ds)
    assert has.tolist() == [k != 6 for k in range(N_SCANS)]
    np.testing.assert_allclose(up, np.tile([0.0, 0.0, 1.0], (N_SCANS, 1)), atol=1e-12)
    assert bev.valid_scans(ds).tolist() == has.tolist()


def test_grid_resolution_and_bounds_are_in_the_recipe(tmp_path, monkeypatch):
    from apairo.preprocess.runner import _recipe_key

    root, _ = _synthetic_root(tmp_path)
    ds = bev.open_view(bev.make_view(root, "rec", tmp_path / "out"))
    key = _recipe_key(bev.BevGrids(ds), "npys")
    assert _recipe_key(bev.BevGrids(ds), "npys") == key
    monkeypatch.setattr(bev_grid, "RESOLUTION", 0.25)
    assert _recipe_key(bev.BevGrids(ds), "npys") != key
    monkeypatch.setattr(bev_grid, "RESOLUTION", 0.2)
    monkeypatch.setattr(bev_grid, "X_MAX", 40.0)
    assert _recipe_key(bev.BevGrids(ds), "npys") != key


def test_export_reloads_the_grids_unchanged(tmp_path, monkeypatch):
    import json

    from apairo_huggingface.tartan_bev import export
    from apairo_huggingface.tartan_bev.export import check, to_lerobot

    monkeypatch.setattr(config, "MIN_FUTURE_PATH", 4.0)
    monkeypatch.setattr(export, "MAX_EPISODE_FRAMES", 4)
    root, _ = _synthetic_root(tmp_path)
    bev.process_sequence(root, "rec", tmp_path / "out")
    to_lerobot([tmp_path / "out" / "rec"], "test/tartan_bev", lerobot_root=tmp_path / "lr")
    report = check("test/tartan_bev", tmp_path / "lr")
    assert report["frames"] == N_SCANS - 1                     # the repeated scan is dropped
    assert report["frames_per_episode"] == [3, 3, 3]           # one recording, cut in equal pieces <= 4
    recordings = json.loads((tmp_path / "lr" / "meta" / "apairo_recordings.json").read_text())
    assert recordings == [{"recording": "rec", "episodes": [0, 1, 2]}]
    assert report["trav_values"] == [0, 1, 255]
    assert report["item_mid_shapes"]["observation.bev.geometry"] == (2, H, W)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    data = LeRobotDataset("test/tartan_bev", root=tmp_path / "lr")
    ds = bev.open_view(tmp_path / "out" / "rec", bev.OUTPUT_KEYS)
    item = data[DUP]                                  # frame DUP is scan DUP + 1
    assert item["observation.frame_info"].tolist() == [0, DUP + 1, DUP + 1]
    np.testing.assert_array_equal(item["observation.bev.elevation"].numpy(),
                                  ds.loaders[config.BEV_ELEVATION][DUP + 1].astype(np.float32))
    np.testing.assert_array_equal(np.round(item["observation.bev.trav"][0].numpy() * 255).astype(np.uint8),
                                  ds.loaders[config.BEV_TRAV][DUP + 1])
    np.testing.assert_allclose(item["action"].numpy(), [20.0, 0.0], atol=1e-3)   # 2 m per 0.1 s, straight
    # The state is relative to the episode's first frame: frames 0, 3 and 6
    # start episodes 0, 1 and 2 at the identity, 2 m per frame after that.
    for f, x in [(0, 0.0), (1, 2.0), (2, 4.0), (DUP, 0.0), (DUP + 1, 2.0), (6, 0.0), (8, 4.0)]:
        np.testing.assert_allclose(data[f]["observation.state"].numpy(), [x, 0, 0, 0, 0, 0, 1], atol=1e-3)


# --------------------------------------------------- real data, if present

REAL = "turnpike_2023-09-12-12-39-19"


@pytest.mark.skipif(not (TARTAN_ROOT / REAL / config.LIDAR).is_dir(), reason="TartanDrive not available")
def test_bev_left_is_image_left_on_real_frames(tmp_path):
    # Independent of the BEV conventions: the calibrated camera. Points the BEV
    # puts on the left (y > 1 m) must project left of the image centre.
    from apairo_huggingface.calibration import CAMERA, T_OPTICAL_FROM_VELODYNE

    view = bev.make_view(TARTAN_ROOT, REAL, tmp_path)
    ds = bev.open_view(view)
    k = 800
    p = np.asarray(ds.loaders[config.LIDAR][k]).astype(float)
    pose = np.asarray(ds.loaders[config.POSES][k])
    g = bev.BevGrids(ds)
    b = bev_grid.transform(np.linalg.inv(g.world_from_bev(k)), bev_grid.transform(pose, p))
    c = bev_grid.transform(T_OPTICAL_FROM_VELODYNE, p)
    front = (c[:, 2] > 1) & (b[:, 0] > 3)
    u = CAMERA.K[0, 0] * c[front, 0] / c[front, 2] + CAMERA.K[0, 2]
    assert (u[b[front, 1] > 1] < CAMERA.K[0, 2]).mean() > 0.99
    assert (u[b[front, 1] < -1] > CAMERA.K[0, 2]).mean() > 0.99
    out = g(Sample(data={key: ds.loaders[key][k] for key in bev.INPUT_KEYS},
                   timestamp=float(ds.timestamps[config.LIDAR][k])))
    assert set(np.unique(out[config.BEV_TRAV])) == {0, 1, 255}
    # The ground the vehicle drives on: flat in the near corridor.
    near = out[config.BEV_TRAV] == 1
    assert np.nanmedian(out[config.BEV_GEOMETRY][0][near].astype(float)) < config.FLAT_ROUGHNESS


def test_export_drops_invalid_scans(tmp_path, monkeypatch):
    from apairo_huggingface.tartan_bev.export import frames

    monkeypatch.setattr(config, "MIN_FUTURE_PATH", 4.0)
    root, _ = _synthetic_root(tmp_path, invalid=[5])
    bev.process_sequence(root, "rec", tmp_path / "out")
    scans = [int(f["observation.frame_info"][1]) for f in frames(tmp_path / "out" / "rec")]
    assert scans == [0, 1, 3, 4, 6, 7, 8, 9]               # the last of the repeat 2 / 3, and 5 is invalid
