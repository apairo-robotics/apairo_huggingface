"""Rebuild the FoMo label channels, one trajectory at a time, raw scans transient.

For each red trajectory, in date order:

1. check the disk can hold its RoboSense scans, download them (calib,
   metadata, ``trav_traj`` and ``kissicp_poses`` are already on disk; no
   image is downloaded in bulk);
2. check that the scans are the ones ``trav_traj`` labelled (stamps, counts);
3. per-point labels of the *past* footprint (``trav_traj_past``), for the
   rear-facing Basler;
4. one mask channel per camera (``config.TRAV_MASKS``), at the camera stamps
   of the bucket listing: future footprint for ``zedx_left`` / ``zedx_right``,
   past footprint for ``basler``;
5. verify the masks (one PNG per labelled stamp, values in {0, 1, 255}, every
   stamp a listed raw image) and measure them (pixel shares, where the
   driven pixels lie, bytes per mask);
6. write the label tables under ``--labels``: ``robosense/`` (point labels,
   future and past), ``<camera>/`` (masks) and ``preview/`` (one frame every
   ~10 s per camera: only these raw images are downloaded), and measure the
   preview images (mean intensity, saturated share);
7. delete the downloaded scans of that trajectory, and only those (the file
   names listed in the bucket under its ``robosense/``).

Resumable: a per-trajectory state file records the last finished step; the
fetch skips complete files; complete channels are skipped.

    setsid nohup python scripts/fomo_labels_pipeline.py > /mnt/vault-fellowship/fomo_logs/labels_pipeline.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from apairo_huggingface.fomo import camera_labels, config, point_labels
from apairo_huggingface.fomo.calibration import FOOTPRINT, IMAGE_HEIGHT, IMAGE_WIDTH
from apairo_huggingface.fomo.export import SATURATED_LEVEL
from apairo_huggingface.fomo.fetch import catalog, fetch_trajectory, list_keys
from apairo_huggingface.fomo.labeling import camera_stamps, check_scans, label_images, label_scans

MARGIN = 10e9                 # bytes kept free on the disk after a download

log = logging.getLogger("labels_pipeline")


def state_path(state_dir: Path, traj: str) -> Path:
    return state_dir / (traj.replace("/", "__") + ".json")


def load_state(state_dir: Path, traj: str) -> dict:
    p = state_path(state_dir, traj)
    return json.loads(p.read_text()) if p.exists() else {"trajectory": traj, "step": "todo"}


def save_state(state_dir: Path, state: dict) -> None:
    state_path(state_dir, state["trajectory"]).write_text(json.dumps(state, indent=2))


def present_bytes(traj_dir: Path) -> int:
    d = traj_dir / config.LIDAR
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file()) if d.is_dir() else 0


def verify_masks(traj_dir: Path, camera: str, listed: np.ndarray) -> dict:
    """Completeness and geometry of the mask channel of ``camera``."""
    files = camera_labels.mask_files(traj_dir, camera)
    stamps = np.round(np.loadtxt(traj_dir / config.TRAV_MASKS[camera] / "timestamps.txt", ndmin=1) * 1e6)
    stems = np.array([int(p.stem) for p in files])
    if len(files) == 0 or not np.array_equal(stems, stamps.astype(np.int64)):
        raise RuntimeError(f"{traj_dir.name} {camera}: {len(files)} masks, {len(stamps)} stamps")
    if not np.isin(stems, listed).all():
        raise RuntimeError(f"{traj_dir.name} {camera}: masks at stamps that are not raw images")
    counts = np.zeros(256, dtype=np.int64)
    rows = np.zeros(IMAGE_HEIGHT, dtype=np.int64)
    cols = np.zeros(IMAGE_WIDTH, dtype=np.int64)
    with_driven = 0
    for f in files:
        m = np.asarray(Image.open(f))
        if m.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RuntimeError(f"{f}: shape {m.shape}")
        counts += np.bincount(m.ravel(), minlength=256)
        driven = m == 1
        rows += driven.sum(1)
        cols += driven.sum(0)
        with_driven += bool(driven.any())
    if counts[[v for v in range(256) if v not in (0, 1, 255)]].any():
        raise RuntimeError(f"{traj_dir.name} {camera}: mask values outside {{0, 1, 255}}")
    n, total = len(files), counts.sum()
    return {
        "footprint": FOOTPRINT[camera],
        "masks": n,
        "listed_frames": int(len(listed)),
        "frames_without_mask": int(len(listed) - n),
        "frames_with_driven": with_driven,
        "share": {str(k): float(counts[k] / total) for k in (1, 0, 255)},
        "driven_mean_row": float((rows * np.arange(IMAGE_HEIGHT)).sum() / max(rows.sum(), 1)),
        "driven_mean_col": float((cols * np.arange(IMAGE_WIDTH)).sum() / max(cols.sum(), 1)),
        # mean driven width (px per row, per mask) in bands of rows
        "driven_width_rows_400_500": float(rows[400:500].mean() / n),
        "driven_width_rows_1000_1200": float(rows[1000:1200].mean() / n),
        "mean_png_bytes": float(np.mean([f.stat().st_size for f in files])),
    }


def preview_quality(cache: Path, traj: str, camera: str) -> dict:
    """Mean intensity and saturated share of the preview raw images."""
    means, sat = [], []
    for f in sorted((cache / traj / camera).glob("*.png")):
        rgb = np.asarray(Image.open(f).convert("RGB"))
        means.append(rgb.mean())
        sat.append((rgb >= SATURATED_LEVEL).all(axis=2).mean())
    return {"frames": len(means), "mean_intensity": float(np.mean(means)),
            "saturated_share": float(np.mean(sat))}


def delete_downloads(traj: str, traj_dir: Path) -> int:
    """Delete the bucket's ``robosense/`` files of ``traj``, nothing else."""
    names = {k.rsplit("/", 1)[1] for k, _ in list_keys(f"data/{traj}/{config.LIDAR}/")}
    d = traj_dir / config.LIDAR
    removed = 0
    if d.is_dir():
        for f in d.iterdir():
            if f.is_file() and (f.name in names or f.name.removesuffix(".part") in names):
                f.unlink()
                removed += 1
        if not any(d.iterdir()):
            d.rmdir()
    return removed


def run_one(traj: str, sizes: dict, args) -> dict:
    state_dir = args.logs / "labels_pipeline"
    state = load_state(state_dir, traj)
    if state["step"] == "done":
        log.info("%s: done", traj)
        return state
    traj_dir = args.root / traj
    if not (traj_dir / config.POSES / "timestamps.txt").exists():
        raise RuntimeError(f"{traj}: no {config.POSES} on disk")
    t0 = time.time()

    if state["step"] != "labelled":
        need = sizes[config.LIDAR] - present_bytes(traj_dir)
        free = shutil.disk_usage(args.root).free
        log.info("%s: %.1f GB to download, %.1f GB free", traj, need / 1e9, free / 1e9)
        if free - need < MARGIN:
            raise SystemExit(f"{traj}: not enough space ({free / 1e9:.1f} GB free, "
                             f"{need / 1e9:.1f} GB needed + {MARGIN / 1e9:.0f} GB margin)")
        fetch_trajectory(traj, args.root, subdirs=(config.LIDAR,))
        state["fetch_s"] = round(time.time() - t0)

        t1 = time.time()
        label_scans(traj_dir, footprint="future")
        state["scans_checked"] = check_scans(traj_dir)
        save_state(state_dir, state)
        label_scans(traj_dir, footprint="past")
        state["masks"] = {}
        for camera in config.CAMERAS:
            listed = camera_stamps(traj, camera)
            label_images(traj_dir, camera=camera, stamps_us=listed)
            state["masks"][camera] = verify_masks(traj_dir, camera, listed)
            log.info("%s %s: %s", traj, camera, json.dumps(state["masks"][camera]))
        state["label_s"] = round(time.time() - t1)

        t2 = time.time()
        point_labels.export([traj_dir], args.labels)
        camera_labels.export([traj_dir], args.labels)
        preview = camera_labels.export_preview(traj, traj_dir, args.labels, args.logs / "preview_cache")
        state["preview_bytes"] = preview.stat().st_size
        state["preview_quality"] = {c: preview_quality(args.logs / "preview_cache", traj, c)
                                    for c in config.CAMERAS}
        dep, name = traj.split("/")
        state["table_bytes"] = {c: (args.labels / c / dep / f"{name}.parquet").stat().st_size
                                for c in (config.LIDAR, *config.CAMERAS)}
        state["export_s"] = round(time.time() - t2)
        state["step"] = "labelled"
        save_state(state_dir, state)

    state["deleted_files"] = delete_downloads(traj, traj_dir)
    state["step"] = "done"
    state["total_s"] = round(time.time() - t0)
    save_state(state_dir, state)
    log.info("%s: done, deleted %d downloaded scans", traj, state["deleted_files"])
    return state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/mnt/vault-fellowship/fomo"))
    p.add_argument("--logs", type=Path, default=Path("/mnt/vault-fellowship/fomo_logs"))
    p.add_argument("--labels", type=Path, default=Path("/mnt/vault-fellowship/fomo_labels"))
    p.add_argument("--color", default="red")
    p.add_argument("--only", action="append", help="<deployment>/<trajectory>; repeatable")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    (args.logs / "labels_pipeline").mkdir(parents=True, exist_ok=True)

    cat = catalog()
    trajs = sorted(f"{d}/{n}" for d, ts in cat.items() for n in ts if n.startswith(args.color + "_"))
    todo = [t for t in trajs if not args.only or t in args.only]
    for attempt in range(2):  # a second pass retries what failed (network, mostly)
        for traj in todo:
            dep, name = traj.split("/")
            try:
                run_one(traj, cat[dep][name]["sensors"], args)
            except Exception:  # noqa: BLE001 -- the disk check still stops the run
                log.exception("%s: failed (pass %d)", traj, attempt + 1)
    done = [t for t in todo if load_state(args.logs / "labels_pipeline", t)["step"] == "done"]
    log.info("%d / %d trajectories done", len(done), len(todo))


if __name__ == "__main__":
    main()
