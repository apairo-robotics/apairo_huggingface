"""The hand-labeled eval benchmark: ``ground_truth`` -> a plain HF dataset.

Unlike :mod:`.export` (a LeRobotDataset of continuous trajectories, every
feature with one fixed shape declared up front), this is a curated subset of
hand-labeled frames carrying a *variable-size* point cloud per row -- a
LeRobotDataset feature has nowhere to put that (``feature_utils.py`` maps
each feature to a fixed-shape ``datasets.ArrayND``). A plain
``datasets.Dataset`` has no such restriction: points and their per-point
labels are ragged ``Sequence`` columns, image and mask stay ``datasets.Image``
so the Hub viewer renders them like any other dataset -- it just won't render
the point cloud itself; nothing on the Hub does. See docs/eval_benchmark.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from apairo import TartanKittiDataset

from . import config
from .labeling import LIDAR


def load_eval_sequences(root) -> list[Path]:
    """Recordings under ``root`` that carry a projected ``trav_mask_gt``."""
    root = Path(root)
    if (root / config.TRAV_MASK_GT).is_dir():
        return [root]
    probe = TartanKittiDataset(root, keys=[config.CAMERA])
    if not getattr(probe, "_is_root", False):
        raise FileNotFoundError(f"no '{config.TRAV_MASK_GT}' channel under {root}")
    seqs = [Path(s.root_dir) for s in probe.sequences if (Path(s.root_dir) / config.TRAV_MASK_GT).is_dir()]
    if not seqs:
        raise FileNotFoundError(
            f"no recording under {root} has a '{config.TRAV_MASK_GT}' channel -- "
            f"run apairo-tartan-gt-label first"
        )
    return seqs


def eval_rows(seq_dir):
    """One row per hand-labeled frame: image, projected mask, raw points + labels.

    ``ground_truth`` is the reference clock (sparse -- only the hand-labeled
    frames), so ``velodyne_0`` resolves to the exact scan it was labeled from
    and ``image_left_color`` to its nearest camera frame, same tolerance as
    the main export.
    """
    seq_dir = Path(seq_dir)
    ds = TartanKittiDataset(
        seq_dir, keys=[config.CAMERA, config.TRAV_MASK_GT, config.GROUND_TRUTH, LIDAR]
    )
    frames = ds.synchronize(reference=config.GROUND_TRUTH, method="nearest", tolerance=config.TOLERANCE)
    for sample in frames:
        points = np.asarray(sample.data[LIDAR], dtype=np.float32)[:, :3]
        labels = np.asarray(sample.data[config.GROUND_TRUTH]).astype(np.int8).ravel()
        if len(points) != len(labels):
            raise ValueError(
                f"{seq_dir.name} @ {sample.timestamp}: {len(points)} point(s) vs "
                f"{len(labels)} label(s) -- ground_truth was not projected from this scan"
            )
        yield {
            "sequence": seq_dir.name,
            "timestamp": float(sample.timestamp),
            "image": np.asarray(sample.data[config.CAMERA]),
            "trav_mask_gt": np.asarray(sample.data[config.TRAV_MASK_GT]),
            "points": points,
            "point_labels": labels,
        }


def _eval_features():
    import datasets

    return datasets.Features({
        "sequence": datasets.Value("string"),
        "timestamp": datasets.Value("float64"),
        "image": datasets.Image(),
        "trav_mask_gt": datasets.Image(),
        "points": datasets.Sequence(datasets.Sequence(datasets.Value("float32"), length=3)),
        "point_labels": datasets.Sequence(datasets.Value("int8")),
    })


def to_eval_dataset(root, repo_id: str, *, push: bool = False):
    """Build the hand-labeled eval benchmark and optionally push it to the Hub."""
    import datasets

    def generator():
        for seq_dir in load_eval_sequences(root):
            yield from eval_rows(seq_dir)

    dataset = datasets.Dataset.from_generator(generator, features=_eval_features())
    if push:
        dataset.push_to_hub(repo_id)
    return dataset


def validate_eval(root) -> tuple[dict[str, int], int]:
    """Exercise the read path without building the HF dataset.

    Returns per-recording frame counts and the total point count.
    """
    counts, total_points = {}, 0
    for seq_dir in load_eval_sequences(root):
        n = 0
        for row in eval_rows(seq_dir):
            n += 1
            total_points += len(row["points"])
        counts[seq_dir.name] = n
    return counts, total_points
