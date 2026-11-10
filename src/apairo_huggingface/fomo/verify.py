"""Check the published FoMo labels against FoMo's own raw files.

Reads the label tables (``<labels>/robosense/...``, ``<labels>/<camera>/...``),
fetches only what each check needs from the public FoMo bucket, and reports:

1. **stamps** - the RoboSense files listed in the bucket are exactly the
   labelled scans; every labelled camera frame is a listed raw image.
2. **n_points** - every listed scan size is 26 x ``n_points``; ``n`` scans
   downloaded read back with exactly ``n_points`` rows.
3. **geometry** - the optical axis of each camera against the driving
   direction measured from the label poses (+1: looks forward, -1: backward).
4. **agreement** - on ``n`` frames per camera, the scan is moved to the image
   instant and projected: the share of driven lidar points (per-point labels
   of the camera's footprint) that land on driven mask pixels, and the share
   of lidar points landing on driven pixels that are labelled driven.

It also writes one overlay JPEG per checked frame, and can log them to Rerun.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .calibration import FOOTPRINT, IMAGE_HEIGHT, IMAGE_WIDTH, INTRINSICS, camera_from_lidar
from .camera_labels import decode_mask, overlay
from .fetch import BUCKET, get, list_keys
from .format import RECORD

logger = logging.getLogger(__name__)


def _fetch(cache: Path, key: str) -> Path:
    out = cache / key
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(get(f"{BUCKET}/data/{key}"))
        tmp.rename(out)
    return out


def driving_direction(poses: np.ndarray, min_step: float = 0.05) -> np.ndarray:
    """Mean unit translation between consecutive poses, in the lidar frame."""
    step = np.einsum("nij,nj->ni", np.linalg.inv(poses[:-1])[:, :3, :3], poses[1:, :3, 3] - poses[:-1, :3, 3])
    step = step[np.linalg.norm(step, axis=1) > min_step]
    d = (step / np.linalg.norm(step, axis=1, keepdims=True)).mean(0)
    return d / np.linalg.norm(d)


def pose_at(t: float, stamps: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """SE(3) pose at ``t``: slerp on rotation, linear on translation."""
    from scipy.spatial.transform import Rotation, Slerp

    i = int(np.clip(np.searchsorted(stamps, t), 1, len(stamps) - 1))
    a = (t - stamps[i - 1]) / (stamps[i] - stamps[i - 1])
    T = np.eye(4)
    T[:3, :3] = Slerp(stamps[i - 1:i + 1], Rotation.from_matrix(poses[i - 1:i + 1, :3, :3]))(t).as_matrix()
    T[:3, 3] = (1 - a) * poses[i - 1, :3, 3] + a * poses[i, :3, 3]
    return T


def _project(points_cam: np.ndarray, camera: str):
    from .labeling import _project_unclipped

    u, v, z = _project_unclipped(points_cam, INTRINSICS[camera])
    ok = (z > 0.5) & (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    return np.where(ok, u, 0).astype(int), np.where(ok, v, 0).astype(int), ok


def verify(traj: str, labels, cache, out, *, cameras=config.CAMERAS, n: int = 20,
           rerun: bool = False) -> dict:
    import pyarrow.parquet as pq

    labels, cache, out = Path(labels), Path(cache), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dep, name = traj.split("/")
    scans = pq.read_table(labels / config.LIDAR / dep / f"{name}.parquet").to_pandas()
    stamps_us = scans.timestamp_us.to_numpy()
    poses = np.stack(scans.pose.to_numpy()).reshape(-1, 4, 4)
    report: dict = {"trajectory": traj}

    # 1-2. stamps and point counts against the bucket listing
    listed = {int(Path(k).stem): s for k, s in list_keys(f"data/{traj}/{config.LIDAR}/") if k.endswith(".bin")}
    report["scans"] = {
        "labelled": len(stamps_us), "listed": len(listed),
        "stamps_equal": sorted(listed) == stamps_us.tolist(),
        "sizes_equal_26_n_points": all(listed.get(int(t)) == RECORD.itemsize * int(c)
                                       for t, c in zip(stamps_us, scans.n_points)),
    }
    picks = np.linspace(0, len(scans) - 1, n).astype(int)
    rows_ok = []
    for i in picks:
        f = _fetch(cache, f"{traj}/{config.LIDAR}/{stamps_us[i]}.bin")
        rows_ok.append(len(np.fromfile(f, dtype=RECORD)) == int(scans.n_points[i]))
    report["scans"]["downloaded"] = len(picks)
    report["scans"]["downloaded_rows_equal_n_points"] = int(sum(rows_ok))

    # 3. geometry
    calib = cache / traj / "calib"
    _fetch(cache, f"{traj}/calib/transforms.json")
    forward = driving_direction(poses)
    report["driving_direction_lidar"] = forward.round(3).tolist()

    if rerun:
        import rerun as rr

        rr.init("fomo_verify", spawn=False)
        rr.connect_grpc() if hasattr(rr, "connect_grpc") else rr.connect()

    t_scan = stamps_us * 1e-6
    for camera in cameras:
        E = camera_from_lidar(calib.parent, camera)
        axis = np.linalg.inv(E)[:3, 2]
        frames = pq.read_table(labels / camera / dep / f"{name}.parquet").to_pandas()
        cam_listed = {int(Path(k).stem) for k, _ in list_keys(f"data/{traj}/{camera}/") if k.endswith(".png")}
        driven_col = "driven_indices" if FOOTPRINT[camera] == "future" else "driven_indices_past"
        res = {
            "footprint": FOOTPRINT[camera],
            "optical_axis_dot_driving_direction": round(float(axis @ forward), 3),
            "labelled": len(frames), "listed": len(cam_listed),
            "labelled_stamps_listed": int(frames.timestamp_us.isin(cam_listed).sum()),
        }
        on_driven, driven_label, per_frame = [], [], []
        for k in np.linspace(0, len(frames) - 1, n).astype(int):
            us = int(frames.timestamp_us[k])
            t = us * 1e-6
            j = int(np.argmin(np.abs(t_scan - t)))
            xyz = np.fromfile(_fetch(cache, f"{traj}/{config.LIDAR}/{stamps_us[j]}.bin"), dtype=RECORD)
            xyz = np.column_stack([xyz["x"], xyz["y"], xyz["z"]]).astype(np.float64)
            lab = np.zeros(len(xyz), dtype=bool)
            lab[np.asarray(scans[driven_col][j], dtype=np.int64)] = True
            motion = np.linalg.inv(pose_at(t, t_scan, poses)) @ poses[j]
            pc = xyz @ (E @ motion)[:3, :3].T + (E @ motion)[:3, 3]
            u, v, ok = _project(pc, camera)
            mask = decode_mask(frames.mask_png[k])
            hit = np.zeros(len(xyz), dtype=bool)
            hit[ok] = mask[v[ok], u[ok]] == 1
            if (lab & ok).sum() >= 20:
                on_driven.append(hit[lab & ok].mean())
            if hit.sum() >= 20:
                driven_label.append(lab[hit].mean())
            image = np.asarray(Image.open(_fetch(cache, f"{traj}/{camera}/{us}.png")).convert("RGB"))
            ov = Image.fromarray(overlay(image, mask)).resize((IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2))
            path = out / f"{dep}_{name}_{camera}_{us}.jpg"
            ov.save(path, quality=88)
            per_frame.append(str(path))
            if rerun:
                rr.set_time("camera_time", timestamp=t) if hasattr(rr, "set_time") else rr.set_time_seconds("camera_time", t)
                rr.log(f"{camera}/image", rr.Image(image))
                rr.log(f"{camera}/mask", rr.SegmentationImage(np.where(mask == 255, 0, 2 - mask)))
        res["frames_checked"] = len(per_frame)
        res["driven_points_on_driven_pixels_median"] = round(float(np.median(on_driven)), 3) if on_driven else None
        res["points_on_driven_pixels_labelled_driven_median"] = (
            round(float(np.median(driven_label)), 3) if driven_label else None)
        res["overlays"] = per_frame
        report[camera] = res
    (out / f"{dep}_{name}_verify.json").write_text(json.dumps(report, indent=2))
    return report
