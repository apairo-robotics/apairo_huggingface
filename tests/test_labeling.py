"""End-to-end tests for the traversability labeling pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from apairo_huggingface import config, label_sequence
from apairo_huggingface.calibration import IMAGE_HEIGHT, IMAGE_WIDTH
from conftest import GROUND_Z, N


@pytest.fixture
def seq(tartan_root_raw):
    return Path(tartan_root_raw) / "forest_00"


def test_label_sequence_writes_registered_channels(seq):
    assert label_sequence(seq) is True

    for channel in (config.TRAV_TRAJ, config.LIDAR_UV, config.TRAV_MASK):
        files = sorted((seq / channel).glob("[0-9]*.npy"))
        assert len(files) == N, channel
        assert (seq / channel / "timestamps.txt").exists()

    registered = yaml.safe_load((seq / ".apairo" / "channels.yaml").read_text())["channels"]
    for channel in (config.TRAV_TRAJ, config.LIDAR_UV, config.TRAV_MASK):
        assert registered[channel]["loader"] == "npys"


def test_labels_are_row_aligned_and_semantically_right(seq):
    label_sequence(seq)
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
    mask = np.load(seq / config.TRAV_MASK / "000000.npy")
    assert mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1, 255}

    # The rasterized corridor is orders of magnitude denser than the splatted
    # rings alone: a 0.8 m half-width over ~5 m of drive fills a solid band.
    assert (mask == 1).sum() > 5000
    assert (mask == 0).any()      # the obstacle points
    assert (mask == 255).any()    # sky / unobserved


def test_relabel_skips_unless_overwrite(seq):
    assert label_sequence(seq) is True
    before = np.load(seq / config.TRAV_MASK / "000000.npy")

    assert label_sequence(seq) is False

    assert label_sequence(seq, overwrite=True) is True
    after = np.load(seq / config.TRAV_MASK / "000000.npy")
    np.testing.assert_array_equal(before, after)  # deterministic pipeline
