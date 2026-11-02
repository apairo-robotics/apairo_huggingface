import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from apairo.core.sample import Sample
from apairo_huggingface.rellis import bev, config
from apairo_preprocess import BEVRasterisation

RELLIS_ROOT = Path(os.environ.get("RELLIS_ROOT", "/mnt/vault-fellowship/rellis"))
H, W = bev.grid_shape()


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------------ geometry

def test_grid_is_200_by_200_with_the_lidar_at_the_corner_of_row_150_col_100():
    assert (H, W) == (200, 200)
    cells = bev.cell_index(np.array([[0.01, -0.01, 0.0], [29.99, 19.99, 0.0], [-9.99, -19.99, 0.0], [31.0, 0.0, 0.0]]))
    assert cells[0] == 149 * W + 100            # just ahead-right of the lidar: row 149 ends at x = 0
    assert cells[1] == 0                        # far front, far left
    assert cells[2] == H * W - 1                # far back, far right
    assert cells[3] == -1                       # outside


def test_cell_index_bins_like_bev_rasterisation():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-12, 32, 500), rng.uniform(-22, 22, 500), np.zeros(500)])
    raster = BEVRasterisation(bounds=bev.PLANE_BOUNDS, resolution=config.BEV_RESOLUTION,
                              points_key="uv", values_key="v", reduce="last", background=-1)
    grid = raster(Sample(data={"uv": bev.to_plane(pts), "v": np.arange(500, dtype=float)}))
    cells = bev.cell_index(pts)
    inside = cells >= 0
    hit = np.flatnonzero(grid.ravel() >= 0)
    assert set(hit) == set(cells[inside])
    centres = bev.cell_centres()
    np.testing.assert_array_equal(bev.cell_index(np.column_stack([centres, np.zeros(len(centres))])),
                                  np.arange(H * W))


def test_bev_frame_is_the_heading_on_the_ground_plane():
    # Lidar yawed 30 deg and pitched 10 deg: the robot heading (lidar -x) on the
    # ground plane becomes BEV +x, and BEV z stays world up.
    pitch = np.array([[np.cos(0.17), 0, np.sin(0.17)], [0, 1, 0], [-np.sin(0.17), 0, np.cos(0.17)]])
    T = np.eye(4)
    T[:3, :3] = _rot_z(30) @ pitch
    T[:3, 3] = [5.0, -2.0, 3.0]
    B = bev.world_from_bev(T)
    np.testing.assert_allclose(B[:3, 2], [0, 0, 1], atol=1e-12)
    heading = -T[:3, 0]
    heading[2] = 0
    np.testing.assert_allclose(B[:3, 0], heading / np.linalg.norm(heading), atol=1e-12)
    np.testing.assert_allclose(B[:3, 3], T[:3, 3])


def test_self_box_drops_returns_behind_the_lidar_only():
    pts = np.array([[2.3, 0.0, -0.1], [-2.3, 0.0, -0.1], [2.3, 0.0, -1.0], [3.5, 0.0, 0.0]])
    np.testing.assert_array_equal(bev.in_self_box(pts), [True, False, False, False])


# -------------------------------------------------------------------- layers

def test_elevation_min_max_mean_and_nan_unknown():
    pts = np.array([[0.05, 0.05, 0.0]] * 3 + [[10.05, 0.05, 0.0]])
    e = bev.elevation(pts, np.array([0.0, 1.0, 2.0, -0.5]))
    assert e.shape == (3, H, W) and e.dtype == np.float16
    c = bev.cell_index(pts[:1])[0]
    np.testing.assert_allclose(e.reshape(3, -1)[:, c], [0.0, 2.0, 1.0])
    assert np.isnan(e).sum() == 3 * (H * W - 2)


def test_semantic_majority_void_only_when_alone_and_255_unknown():
    a, b, c = bev.cell_index(np.array([[0.05, 0.05, 0], [5.05, 0.05, 0], [9.05, 0.05, 0]]))
    cells = np.array([a, a, a, a, b, b, c])
    labels = np.array([3, 3, 4, 0, 0, 0, 19])   # a: grass x2 vs tree, void ignored; b: void only
    sem = bev.semantic(cells, labels).ravel()
    assert sem[a] == 3 and sem[b] == 0 and sem[c] == 19
    assert (sem == config.NO_DATA).sum() == H * W - 3


def test_footprint_of_a_straight_path_is_the_robot_width():
    path = np.column_stack([np.linspace(-5, 25, 50), np.zeros(50)])
    fp = bev.footprint(path, bev.cell_centres(), config.ROBOT_HALF_WIDTH)
    row = fp[50]   # 20 m ahead
    width = row.sum() * config.BEV_RESOLUTION
    assert abs(width - 2 * config.ROBOT_HALF_WIDTH) <= 2 * config.BEV_RESOLUTION
    assert row[100] and row[99]                 # centred on y = 0
    assert not fp[0].any()                      # path ends at 25 m


# ------------------------------------------------------ synthetic sequence

N_SCANS = 8


def _ground_scan(rng):
    # A flat ground 1 m below the lidar, in the lidar frame (no points within 2 m).
    r = rng.uniform(2.5, 25, 4000)
    a = rng.uniform(-np.pi, np.pi, 4000)
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(4000, -1.0), np.zeros(4000)]).astype(np.float32)


def _synthetic_root(tmp_path):
    # Robot drives along world +y at 2 m per scan; the lidar's -x is the heading.
    seq = tmp_path / "rellis" / "Rellis-3D" / "00000"
    for name in bev.BEV_SOURCE_DIRS:
        (seq / name).mkdir(parents=True)
    rng = np.random.default_rng(1)
    rows = []
    for k in range(N_SCANS):
        R = _rot_z(-90.0)                       # lidar -x -> world +y
        rows.append(np.hstack([R, [[0.0], [2.0 * k], [1.0]]]).ravel())
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(
            seq / config.CAMERA / f"frame{k:06d}-1581624000_{100 * k:03d}.jpg")
        _ground_scan(rng).tofile(seq / "os1_cloud_node_kitti_bin" / f"{k:06d}.bin")
        np.full(4000, 3, dtype="<u4").tofile(seq / "os1_cloud_node_semantickitti_label_id" / f"{k:06d}.label")
    np.savetxt(seq / "poses.txt", np.array(rows))
    return tmp_path / "rellis", seq


def test_view_writes_nothing_in_the_source(tmp_path):
    root, seq = _synthetic_root(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in seq.rglob("*"))
    view = bev.make_view(root, "00000", tmp_path / "out")
    bev.make_view(root, "00000", tmp_path / "out")          # idempotent
    assert sorted(p.relative_to(tmp_path) for p in seq.rglob("*")) == before
    stamps = np.loadtxt(view / config.BEV_POSES / "timestamps.txt")
    np.testing.assert_allclose(stamps, 1581624000 + 0.1 * np.arange(N_SCANS))


def test_process_writes_three_registered_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MIN_FUTURE_PATH", 4.0)
    root, _ = _synthetic_root(tmp_path)
    bev.process_sequence(root, "00000", tmp_path / "out")
    ds = bev.open_view(tmp_path / "out" / "00000", bev.OUTPUT_KEYS)
    assert all(len(ds.timestamps[k]) == N_SCANS for k in bev.OUTPUT_KEYS)
    elev = ds.loaders[config.BEV_ELEVATION][2]
    sem = ds.loaders[config.BEV_SEMANTIC][2]
    trav = ds.loaders[config.BEV_TRAV][2]
    assert elev.shape == (3, H, W) and sem.dtype == np.uint8 and trav.dtype == np.uint8
    assert set(np.unique(sem)) == {3, 255} and set(np.unique(trav)) == {0, 1, 255}
    # Ground 1 m below the lidar = height 0 above the ground under the robot.
    assert abs(np.nanmedian(elev[2].astype(float))) < 0.01
    # The driven corridor runs straight up the grid through the robot.
    assert trav[100, 100] == 1 and trav[100, 60] != 1
    assert trav[150, 100] == 1                  # under the robot
    # The last scans have < MIN_FUTURE_PATH ahead: all unknown.
    assert (ds.loaders[config.BEV_TRAV][N_SCANS - 1] == 255).all()
    # Recipe reuse: a second run is a no-op.
    mtime = (tmp_path / "out" / "00000" / config.BEV_TRAV / "000002.npy").stat().st_mtime
    bev.process_sequence(root, "00000", tmp_path / "out")
    assert (tmp_path / "out" / "00000" / config.BEV_TRAV / "000002.npy").stat().st_mtime == mtime


def test_accumulation_is_past_only(tmp_path):
    root, seq = _synthetic_root(tmp_path)
    view = bev.make_view(root, "00000", tmp_path / "out")
    j = 4

    def grids():
        ds = bev.open_view(view)
        g = bev.BevGrids(ds, accumulate=3, stride=1)
        assert g.scans_used(j) == [4, 3, 2]
        s = Sample(data={k: ds.loaders[k][j] for k in bev.INPUT_KEYS}, timestamp=float(ds.timestamps[config.LIDAR][j]))
        return g(s)

    before = grids()
    # Rewrite every future scan: the inputs (elevation, semantic) must not move.
    for k in range(j + 1, N_SCANS):
        (np.random.default_rng(k).normal(size=(4000, 4)) * 5).astype(np.float32).tofile(
            seq / "os1_cloud_node_kitti_bin" / f"{k:06d}.bin")
    after = grids()
    np.testing.assert_array_equal(before[config.BEV_ELEVATION], after[config.BEV_ELEVATION])
    np.testing.assert_array_equal(before[config.BEV_SEMANTIC], after[config.BEV_SEMANTIC])
    # And a past scan does move them.
    shifted = _ground_scan(np.random.default_rng(7))
    shifted[:, 2] += 0.5
    shifted.tofile(seq / "os1_cloud_node_kitti_bin" / f"{j - 1:06d}.bin")
    moved = grids()
    assert not np.array_equal(np.nan_to_num(before[config.BEV_ELEVATION]), np.nan_to_num(moved[config.BEV_ELEVATION]))


# --------------------------------------------------- real data, if present

@pytest.mark.skipif(not (RELLIS_ROOT / "Rellis-3D" / "00003").is_dir(), reason="RELLIS-3D not available")
def test_bev_left_is_image_left_on_real_frames(tmp_path):
    # Independent of the BEV code's conventions: the calibrated camera. Points
    # the BEV puts on the left (y > 1 m) must project left of the image centre.
    from apairo_huggingface.rellis.calibration import camera_from_lidar, intrinsics

    view = bev.make_view(RELLIS_ROOT, "00003", tmp_path)
    ds = bev.open_view(view)
    E, K = camera_from_lidar(RELLIS_ROOT, "00003"), intrinsics(RELLIS_ROOT, "00003")
    poses = bev.poses_from_rows(np.loadtxt(view / config.BEV_POSES / "poses.txt"))
    k = 1000
    p = np.asarray(ds.loaders[config.LIDAR][k])[:, :3].astype(float)
    b = bev.transform(np.linalg.inv(bev.world_from_bev(poses[k])), bev.transform(poses[k], p))
    c = bev.transform(E, p)
    front = (c[:, 2] > 1) & (b[:, 0] > 3)
    u = K[0, 0] * c[front, 0] / c[front, 2] + K[0, 2]
    assert (u[b[front, 1] > 1] < K[0, 2]).mean() > 0.99
    assert (u[b[front, 1] < -1] > K[0, 2]).mean() > 0.99
    # And the grid itself on that frame.
    g = bev.BevGrids(ds)
    out = g(Sample(data={key: ds.loaders[key][k] for key in bev.INPUT_KEYS},
                   timestamp=float(ds.timestamps[config.LIDAR][k])))
    assert set(np.unique(out[config.BEV_TRAV])) == {0, 1, 255}
    assert set(np.unique(out[config.BEV_SEMANTIC])) <= set(config.ONTOLOGY) | {255}
