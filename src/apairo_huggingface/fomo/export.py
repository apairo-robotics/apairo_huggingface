"""FoMo trajectories -> one LeRobotDataset, one episode per trajectory.

Three clocks meet here and apairo aligns them lazily: each frame is a real
camera image (default: the forward ZED X left) with its camera-clocked mask
(``config.TRAV_MASKS[camera]``), the twist command
in force at that instant (``previous`` on the ~50 Hz ``cmd_velocity``), and
the KISS-ICP pose *interpolated* at the image timestamp.

Poses are re-expressed in ``base_link`` (+x forward) and relative to the first
pose of the trajectory, so every episode starts at the identity.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from apairo_transform.interp import Se3Interp

from . import config
from .labeling import load_poses, open_trajectory

FPS = 10
ROBOT_TYPE = "clearpath_warthog"
TASK = "drive the robot along the forest trail"
DOWNSCALE = 2                 # 1920x1200 -> 960x600
MIN_FUTURE_PATH = 10.0        # m; frames closer to the end have no corridor ahead
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["linear_velocity", "angular_velocity"]
ENV_NAMES = ["snow_depth_m", "air_temperature_c"]

# A trajectory whose camera is saturated carries no image information (the
# Basler of 2025-01-29, -24 degC, is pure white). The decision is measured on
# the images of the exported camera, never taken from a list of names.
SATURATED_LEVEL = 250         # a pixel is saturated when all of R, G, B reach it
MAX_SATURATED_SHARE = 0.5     # exclude above this share of saturated pixels
QUALITY_SAMPLES = 50          # frames sampled evenly along the trajectory


def image_quality(traj_dir, camera: str = config.CAMERA, samples: int = QUALITY_SAMPLES) -> dict:
    """Mean intensity and saturated-pixel share over frames sampled evenly."""
    files = sorted((Path(traj_dir) / camera).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no {camera} image under {traj_dir}")
    picks = [files[i] for i in np.linspace(0, len(files) - 1, min(samples, len(files))).astype(int)]
    means, saturated = [], []
    for f in picks:
        rgb = np.asarray(Image.open(f).convert("RGB"))
        means.append(rgb.mean())
        saturated.append((rgb >= SATURATED_LEVEL).all(axis=2).mean())
    return {"camera": camera, "frames": len(picks), "mean_intensity": float(np.mean(means)),
            "saturated_share": float(np.mean(saturated))}


def excluded(traj_dir, camera: str = config.CAMERA) -> bool:
    """True if ``camera`` is saturated on ``traj_dir`` (see :func:`image_quality`)."""
    return image_quality(traj_dir, camera)["saturated_share"] > MAX_SATURATED_SHARE


# ------------------------------------------------------------------ helpers

def _base_from_lidar(traj_dir: Path) -> np.ndarray:
    from .calibration import transforms

    return transforms(traj_dir)[("base_link", config.LIDAR)]


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


def _campbell_mean(path: Path, column: str) -> float:
    """Mean of one column of a Campbell logger table (header + 2 unit rows)."""
    if not path.exists():
        return float("nan")
    with path.open(newline="", encoding="latin-1") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4 or column not in rows[0]:
        return float("nan")
    j = rows[0].index(column)
    values = []
    for row in rows[3:]:
        try:
            values.append(float(row[j]))
        except (ValueError, IndexError):
            pass
    return float(np.nanmean(values)) if values else float("nan")


def environment(traj_dir: Path) -> np.ndarray:
    """Snow depth (m) and air temperature (degC) over the trajectory."""
    meta = traj_dir / "metadata"
    snow_mm = _campbell_mean(meta / "snow_data.csv", "SDMS40_Depth_Avg")
    temp_c = _campbell_mean(meta / "meteo_data.csv", "T_probe_Avg")
    return np.array([snow_mm / 1000.0, temp_c], dtype=np.float32)


# ------------------------------------------------------------------- frames

def aligned(traj_dir, camera: str = config.CAMERA):
    """The synchronized view of one trajectory on the mask (camera) clock."""
    mask = config.TRAV_MASKS[camera]
    ds = open_trajectory(traj_dir, [mask, camera, "cmd_velocity", config.POSES])
    return ds.synchronize(
        reference=mask,
        method={
            camera: "nearest",                  # the mask's own image: same stamp
            "cmd_velocity": "previous",         # the command in force
            config.POSES: Se3Interp(),          # the pose at the image instant
        },
        tolerance=config.POSE_TOLERANCE,
    )


def frames(traj_dir, camera: str = config.CAMERA):
    """Yield LeRobot frame dicts for one trajectory."""
    traj_dir = Path(traj_dir)
    sync = aligned(traj_dir, camera)
    base_from_lidar = _base_from_lidar(traj_dir)
    lidar_from_base = np.linalg.inv(base_from_lidar)
    ds_poses, poses_ts = load_poses(open_trajectory(traj_dir, [config.POSES]))
    path = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ds_poses[:, :3, 3], axis=0), axis=1))])
    origin = None
    env = environment(traj_dir)
    stale = np.abs(sync.time_offsets(camera)) > 1e-3
    for k in range(len(sync)):
        if stale[k]:
            continue
        sample = sync[k]
        t = float(sample.timestamp)
        here = np.interp(t, poses_ts, path)
        if path[-1] - here < MIN_FUTURE_PATH:
            continue
        world_base = np.asarray(sample.data[config.POSES]) @ lidar_from_base
        if origin is None:
            origin = np.linalg.inv(world_base)
        pose = origin @ world_base
        image = Image.fromarray(np.asarray(sample.data[camera])[..., :3])  # ZED PNGs are RGBA
        mask = np.asarray(sample.data[config.TRAV_MASKS[camera]])
        if mask.ndim == 3:
            mask = mask[..., 0]
        size = (image.width // DOWNSCALE, image.height // DOWNSCALE)
        mask_small = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
        yield {
            "observation.images.front": np.asarray(image.resize(size, Image.BILINEAR)),
            "observation.traversability_mask": np.repeat(mask_small[:, :, None], 3, axis=2),
            "observation.state": np.concatenate([pose[:3, 3], _quat_xyzw(pose[:3, :3])]).astype(np.float32),
            "observation.environment": env,
            "action": np.asarray(sample.data["cmd_velocity"], dtype=np.float32),
            "task": TASK,
        }


def features(height: int, width: int) -> dict:
    return {
        "observation.images.front": {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        # Lossless PNG: mp4 would corrupt the {0, 1, 255} label values.
        "observation.traversability_mask": {
            "dtype": "image", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.environment": {"dtype": "float32", "shape": (2,), "names": ENV_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
    }


def to_lerobot(traj_dirs, repo_id: str, *, camera: str = config.CAMERA, root=None,
               push: bool = False):
    """Write the trajectories to a LeRobotDataset (one episode each)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    traj_dirs = [Path(t) for t in traj_dirs]
    probe = next(frames(traj_dirs[0], camera))
    h, w, _ = probe["observation.images.front"].shape
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=features(h, w),
        robot_type=ROBOT_TYPE, root=root,
    )
    for traj in traj_dirs:
        for frame in frames(traj, camera):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    if push:
        dataset.push_to_hub()
    return dataset
