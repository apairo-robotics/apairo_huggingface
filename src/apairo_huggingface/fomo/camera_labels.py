"""Per-camera traversability masks of FoMo, as labels-only tables.

One parquet per camera and trajectory, one row per labelled camera frame::

    <out>/<camera>/<deployment>/<trajectory>.parquet

Columns: ``trajectory``, ``camera``, ``footprint`` (``future`` for the
forward ZED X cameras, ``past`` for the rear-facing Basler), ``image_file``
(``<us>.png``, the raw FoMo image under ``<trajectory>/<camera>/``),
``timestamp_us`` (int64, the image's stamp), ``mask_png`` (the (1200, 1920)
uint8 mask as a lossless PNG: 1 driven / 0 lidar-seen, not driven / 255 no
data).

A small ``preview`` table makes the labels checkable without downloading
FoMo: one frame every ~10 s per camera, a half-size JPEG of the raw image
with the mask drawn over it, and the mask itself::

    <out>/preview/<deployment>/<trajectory>.parquet

Columns: ``trajectory``, ``camera``, ``footprint``, ``image_file``,
``timestamp_us``, ``overlay_jpeg``, ``mask_png``.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .calibration import FOOTPRINT

logger = logging.getLogger(__name__)

PREVIEW_PERIOD_S = 10.0
PREVIEW_DOWNSCALE = 2
GREEN, RED = np.array([0, 255, 0]), np.array([255, 40, 40])


def _name(traj_dir: Path) -> str:
    return f"{traj_dir.parent.name}/{traj_dir.name}"


def mask_files(traj_dir, camera: str) -> list[Path]:
    """The mask PNGs of a complete mask channel, in time order."""
    mask_dir = Path(traj_dir) / config.TRAV_MASKS[camera]
    if not (mask_dir / "timestamps.txt").exists():
        raise FileNotFoundError(f"{mask_dir} is not complete")
    return sorted(mask_dir.glob("*.png"), key=lambda p: int(p.stem))


def trajectory_table(traj_dir, camera: str):
    """The per-frame pyarrow table of one camera of one trajectory."""
    import pyarrow as pa

    traj_dir = Path(traj_dir)
    files = mask_files(traj_dir, camera)
    n = len(files)
    return pa.table({
        "trajectory": pa.array([_name(traj_dir)] * n, pa.string()),
        "camera": pa.array([camera] * n, pa.string()),
        "footprint": pa.array([FOOTPRINT[camera]] * n, pa.string()),
        "image_file": pa.array([f"{p.stem}.png" for p in files], pa.string()),
        "timestamp_us": pa.array([int(p.stem) for p in files], pa.int64()),
        "mask_png": pa.array([p.read_bytes() for p in files], pa.binary()),
    })


def export(traj_dirs, out, cameras=config.CAMERAS) -> list[Path]:
    """Write ``<out>/<camera>/<deployment>/<trajectory>.parquet``; return the paths."""
    import pyarrow.parquet as pq

    out = Path(out)
    written = []
    for traj in map(Path, traj_dirs):
        for camera in cameras:
            table = trajectory_table(traj, camera)
            path = out / camera / traj.parent.name / f"{traj.name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path, compression="none")  # PNG bytes do not compress further
            logger.info("%s %s: %d masks -> %s (%.1f MB)", _name(traj), camera,
                        table.num_rows, path, path.stat().st_size / 1e6)
            written.append(path)
    return written


def decode_mask(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)))


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGB image with 1 tinted green and 0 painted red (255 left as is)."""
    rgb = image[..., :3].astype(np.float32)
    out = rgb.copy()
    out[mask == 1] = 0.55 * rgb[mask == 1] + 0.45 * GREEN
    out[mask == 0] = RED
    return out.astype(np.uint8)


def preview_stamps(stamps_us, period_s: float = PREVIEW_PERIOD_S) -> list[int]:
    """One stamp every ``period_s`` seconds, starting with the first."""
    picked, nxt = [], None
    for us in stamps_us:
        if nxt is None or us >= nxt:
            picked.append(int(us))
            nxt = us + int(period_s * 1e6)
    return picked


def preview_table(traj: str, traj_dir, cache, cameras=config.CAMERAS, period_s: float = PREVIEW_PERIOD_S):
    """The preview rows of one trajectory; fetches only the picked raw images."""
    import pyarrow as pa

    from .fetch import BUCKET, get

    traj_dir, cache = Path(traj_dir), Path(cache)
    rows = {k: [] for k in ("trajectory", "camera", "footprint", "image_file", "timestamp_us",
                            "overlay_jpeg", "mask_png")}
    for camera in cameras:
        files = {int(p.stem): p for p in mask_files(traj_dir, camera)}
        for us in preview_stamps(sorted(files), period_s):
            raw = cache / traj / camera / f"{us}.png"
            if not raw.exists():
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(get(f"{BUCKET}/data/{traj}/{camera}/{us}.png"))
            image = np.asarray(Image.open(raw).convert("RGB"))
            mask_png = files[us].read_bytes()
            ov = Image.fromarray(overlay(image, decode_mask(mask_png)))
            ov = ov.resize((ov.width // PREVIEW_DOWNSCALE, ov.height // PREVIEW_DOWNSCALE), Image.BILINEAR)
            buf = io.BytesIO()
            ov.save(buf, "JPEG", quality=85)
            for k, v in (("trajectory", traj), ("camera", camera), ("footprint", FOOTPRINT[camera]),
                         ("image_file", f"{us}.png"), ("timestamp_us", us),
                         ("overlay_jpeg", buf.getvalue()), ("mask_png", mask_png)):
                rows[k].append(v)
    types = {"timestamp_us": pa.int64(), "overlay_jpeg": pa.binary(), "mask_png": pa.binary()}
    return pa.table({k: pa.array(v, types.get(k, pa.string())) for k, v in rows.items()})


def export_preview(traj: str, traj_dir, out, cache, cameras=config.CAMERAS) -> Path:
    import pyarrow.parquet as pq

    table = preview_table(traj, traj_dir, cache, cameras)
    dep, name = traj.split("/")
    path = Path(out) / "preview" / dep / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="none")
    logger.info("%s preview: %d frames -> %s (%.1f MB)", traj, table.num_rows, path,
                path.stat().st_size / 1e6)
    return path
