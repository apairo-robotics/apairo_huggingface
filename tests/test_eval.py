"""End-to-end tests for the hand-labeled ground-truth pipeline (toaster -> HF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from apairo_huggingface import config
from apairo_huggingface.calibration import IMAGE_HEIGHT, IMAGE_WIDTH
from apairo_huggingface.eval_export import load_eval_sequences, to_eval_dataset, validate_eval
from apairo_huggingface.eval_labeling import label_ground_truth_root, label_ground_truth_sequence
from conftest import GT_INDICES, H, W


@pytest.fixture
def seq(tartan_root_gt):
    return Path(tartan_root_gt) / "forest_00"


def test_label_ground_truth_sequence_writes_registered_channel(seq):
    assert label_ground_truth_sequence(seq) is True

    masks = sorted((seq / config.TRAV_MASK_GT).glob("[0-9]*.png"))
    assert len(masks) == len(GT_INDICES)
    assert (seq / config.TRAV_MASK_GT / "timestamps.txt").exists()

    registered = yaml.safe_load((seq / ".apairo" / "channels.yaml").read_text())["channels"]
    assert registered[config.TRAV_MASK_GT]["kind"] == "preprocess"
    assert registered[config.TRAV_MASK_GT]["loader"] == "img"


def test_ground_truth_mask_is_semantically_right(seq):
    label_ground_truth_sequence(seq)
    mask = np.asarray(Image.open(seq / config.TRAV_MASK_GT / "000000.png"))

    assert mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1, 2}
    assert (mask == 1).any()  # ground, hand-labeled traversable
    assert (mask == 2).any()  # obstacle, hand-labeled not traversable


def test_relabel_skips_unless_overwrite(seq):
    assert label_ground_truth_sequence(seq) is True
    before = np.asarray(Image.open(seq / config.TRAV_MASK_GT / "000000.png"))

    assert label_ground_truth_sequence(seq) is False

    assert label_ground_truth_sequence(seq, overwrite=True) is True
    after = np.asarray(Image.open(seq / config.TRAV_MASK_GT / "000000.png"))
    np.testing.assert_array_equal(before, after)  # deterministic pipeline


def test_label_ground_truth_root_skips_unlabeled_recordings(tartan_root_gt):
    # forest_01 carries no ground_truth at all -- must not raise.
    label_ground_truth_root(tartan_root_gt)
    assert (Path(tartan_root_gt) / "forest_00" / config.TRAV_MASK_GT).exists()
    assert not (Path(tartan_root_gt) / "forest_01" / config.TRAV_MASK_GT).exists()


def test_load_eval_sequences_finds_only_labeled_recordings(tartan_root_gt):
    label_ground_truth_root(tartan_root_gt)
    found = load_eval_sequences(tartan_root_gt)
    assert [p.name for p in found] == ["forest_00"]


def test_eval_dataset_rows_are_row_aligned(tartan_root_gt):
    label_ground_truth_root(tartan_root_gt)
    dataset = to_eval_dataset(tartan_root_gt, "apairo-demo/tartandrive_offroad_eval", push=False)

    assert len(dataset) == len(GT_INDICES)
    row = dataset[0]
    assert row["sequence"] == "forest_00"
    assert len(row["points"]) == len(row["point_labels"])
    assert set(row["point_labels"]) <= {0, 1, 2}
    assert np.asarray(row["image"]).shape == (H, W, 3)
    assert np.asarray(row["trav_mask_gt"]).shape == (IMAGE_HEIGHT, IMAGE_WIDTH)


def test_validate_eval_matches_dataset(tartan_root_gt):
    label_ground_truth_root(tartan_root_gt)
    counts, total_points = validate_eval(tartan_root_gt)

    assert counts == {"forest_00": len(GT_INDICES)}
    assert total_points > 0
