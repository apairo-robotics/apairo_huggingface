"""Project hand-labeled point clouds into the camera -- the eval counterpart
of :mod:`.labeling`.

``ground_truth`` (see :mod:`.config`) is written per hand-labeled scan by
apairo_experiments/scripts/import_toaster_gt.py from toaster exports: one
``<idx>.npy`` of per-point labels (0 unlabeled / 1 traversable / 2 obstacle),
row-aligned with the raw ``velodyne_0`` scan of the same index. This module
splats those labels through the same camera projection as ``trav_mask``
(:class:`LidarCameraProjection` + :class:`ImageMaskFromPointLabels`) to get a
2D ``trav_mask_gt`` -- no trajectory, no corridor, the labels are already
ground truth.

Most recordings carry no ``ground_truth`` at all (only the hand-labeled eval
subset does); :func:`label_ground_truth_root` skips those silently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from apairo import TartanKittiDataset
from apairo.core.sample import Sample
from apairo_preprocess import ImageMaskFromPointLabels, LidarCameraProjection

from . import config
from .calibration import CAMERA, IMAGE_HEIGHT, IMAGE_WIDTH, T_OPTICAL_FROM_VELODYNE
from .labeling import LIDAR, _remove_channel

logger = logging.getLogger(__name__)


def label_ground_truth_sequence(seq_dir, *, overwrite: bool = False) -> bool:
    """Materialize ``trav_mask_gt`` on one recording's hand-labeled frames.

    Returns ``True`` when the channel was (re)written, ``False`` when it
    already existed and ``overwrite`` was not set. Raises if ``seq_dir`` has
    no ``ground_truth`` channel at all -- call :func:`label_ground_truth_root`
    to skip recordings without hand labels instead of failing on them.
    """
    seq_dir = Path(seq_dir)
    gt_dir = seq_dir / config.GROUND_TRUTH
    if not gt_dir.is_dir():
        raise FileNotFoundError(
            f"'{config.GROUND_TRUTH}' not found in {seq_dir} -- import hand labels "
            f"first (apairo_experiments/scripts/import_toaster_gt.py)."
        )

    mask_dir = seq_dir / config.TRAV_MASK_GT
    complete = (mask_dir / "timestamps.txt").exists()
    if complete and not overwrite:
        logger.info("%s: ground truth already projected, skipping (overwrite=True to redo)",
                    seq_dir.name)
        return False
    if mask_dir.exists():
        _remove_channel(seq_dir, config.TRAV_MASK_GT)

    stems = sorted(p.stem for p in gt_dir.glob("[0-9]*.npy"))
    if not stems:
        raise ValueError(f"{gt_dir} has no labeled scans")
    gt_ts = np.loadtxt(gt_dir / "timestamps.txt").reshape(-1)
    if len(stems) != len(gt_ts):
        raise ValueError(f"{seq_dir.name}: {len(stems)} labeled scan(s) but {len(gt_ts)} timestamp(s)")

    ds = TartanKittiDataset(seq_dir, keys=[LIDAR])
    projector = LidarCameraProjection(
        intrinsics=CAMERA, extrinsics=T_OPTICAL_FROM_VELODYNE, lidar_key=LIDAR,
    )
    masker = ImageMaskFromPointLabels(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        labels_key=config.GROUND_TRUTH,
        radius=config.SPLAT_RADIUS,
        occlusion_bin_px=config.OCCLUSION_BIN_PX,
        unknown_value=0,  # toaster's own "unlabeled" value -- no remap needed
    )

    mask_dir.mkdir()
    for stem in stems:
        idx = int(stem)
        labels = np.load(gt_dir / f"{stem}.npy").astype(np.uint8)
        scan = np.asarray(ds.load(LIDAR, idx))
        if len(labels) != len(scan):
            raise ValueError(
                f"{seq_dir.name}/{stem}: {len(labels)} label(s) vs {len(scan)} point(s) "
                f"-- ground_truth was not labeled from this scan"
            )
        uv = projector(Sample(data={LIDAR: scan}))
        mask = masker(Sample(data={config.GROUND_TRUTH: labels, "lidar_uv": uv}))
        Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")

    np.savetxt(mask_dir / "timestamps.txt", gt_ts)
    TartanKittiDataset.register_channel(
        seq_dir, config.TRAV_MASK_GT, "img",
        timestamps_from=config.GROUND_TRUTH, sources=[config.GROUND_TRUTH],
    )
    logger.info("%s: ground truth projected (%d frame(s))", seq_dir.name, len(stems))
    return True


def label_ground_truth_root(root, *, overwrite: bool = False) -> None:
    """Project ``ground_truth`` on every recording under ``root`` that has one.

    Recordings without hand labels (the overwhelming majority) are skipped
    silently -- only the curated eval subset carries ``ground_truth``.
    """
    probe = TartanKittiDataset(root, keys=[LIDAR])
    seq_dirs = (
        [Path(s.root_dir) for s in probe.sequences]
        if getattr(probe, "_is_root", False)
        else [Path(root)]
    )
    labeled = [s for s in seq_dirs if (s / config.GROUND_TRUTH).is_dir()]
    if not labeled:
        logger.warning("no recording under %s has a '%s' channel", root, config.GROUND_TRUTH)
        return

    failed = []
    for seq_dir in labeled:
        try:
            label_ground_truth_sequence(seq_dir, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001 -- survey run, report at the end
            failed.append(seq_dir.name)
            logger.warning("%s: failed -- %s", seq_dir.name, exc)
    if failed:
        logger.warning("%d/%d recording(s) failed: %s", len(failed), len(labeled), ", ".join(failed))
