"""End-to-end tests for the traversability labeling pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from apairo_huggingface import config, label_sequence
from apairo_huggingface.calibration import IMAGE_HEIGHT, IMAGE_WIDTH
from conftest import GROUND_Z, N


@pytest.fixture
def seq(tartan_root_raw):
    return Path(tartan_root_raw) / "forest_00"


def _mask(seq, i=0):
    return np.asarray(Image.open(seq / config.TRAV_MASK / f"{i:06d}.png"))


def test_label_sequence_writes_registered_channels(seq):
    assert label_sequence(seq) is True

    assert len(sorted((seq / config.TRAV_TRAJ).glob("[0-9]*.npy"))) == N
    assert len(sorted((seq / config.TRAV_MASK).glob("[0-9]*.png"))) == N
    assert (seq / config.TRAV_MASK / "timestamps.txt").exists()
    assert not (seq / config.LIDAR_UV).exists()  # opt-in, ~2 GB/recording

    registered = yaml.safe_load((seq / ".apairo" / "channels.yaml").read_text())["channels"]
    assert registered[config.TRAV_TRAJ]["loader"] == "npys"
    assert registered[config.TRAV_MASK]["loader"] == "img"


def test_labels_are_row_aligned_and_semantically_right(seq):
    label_sequence(seq, keep_uv=True)
    scan = np.load(seq / "velodyne_0" / "000000.npy")
    labels = np.load(seq / config.TRAV_TRAJ / "000000.npy")
    uv = np.load(seq / config.LIDAR_UV / "000000.npy")

    assert labels.shape == (len(scan),) and labels.dtype == np.uint8
    assert uv.shape == (len(scan), 3) and uv.dtype == np.float32

    # Ground inside the future corridor is traversable, the obstacle is not,
    # ground far to the side is not.
    ground = scan[:, 2] == np.float32(GROUND_Z)
    corridor = ground & (np.abs(scan[:, 1]) < 0.5) & (scan[:, 0] > 1.0) & (scan[:, 0] < 4.5)
    assert labels[corridor].all()
    assert not labels[~ground].any()
    assert not labels[ground & (np.abs(scan[:, 1]) > 2.0)].any()


def test_mask_contains_dense_corridor(seq):
    label_sequence(seq)
    mask = _mask(seq)
    assert mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1, 255}

    # The rasterized corridor is orders of magnitude denser than the splatted
    # rings alone: a 0.8 m half-width over ~5 m of drive fills a solid band.
    assert (mask == 1).sum() > 5000
    assert (mask == 0).any()      # the obstacle points
    assert (mask == 255).any()    # sky / unobserved


def test_corridor_handles_near_stationary_trajectory():
    # Remaining path shorter than one interpolation step (robot stopping):
    # no corridor, no crash (np.gradient needs at least two samples).
    from apairo_huggingface.labeling import corridor_edges, corridor_mask

    traj = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    left, right = corridor_edges(traj, 0, max_path=40.0, half_width=0.8, ground_offset=-1.77)
    assert len(left) == 0 and len(right) == 0

    mask = corridor_mask(
        traj, 0, np.eye(4), np.full((1, 3), np.nan, dtype=np.float32),
        max_path=40.0, half_width=0.8, ground_offset=-1.77,
    )
    assert not mask.any()


def test_relabel_skips_unless_overwrite(seq):
    assert label_sequence(seq) is True
    before = _mask(seq)

    assert label_sequence(seq) is False

    assert label_sequence(seq, overwrite=True) is True
    np.testing.assert_array_equal(before, _mask(seq))  # deterministic pipeline
