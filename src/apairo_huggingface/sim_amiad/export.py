"""Labelled sim sequences -> one LeRobotDataset, one episode per sequence.

The sensors tick at 60 Hz of sim time; every ``STRIDE``-th labelled scan
becomes a LeRobot frame (15 Hz). Per frame:

- ``observation.images.front``          the RGB image (video), downscaled by 2;
- ``observation.semantic_mask``         the perfect sim segmentation, training
  class ids (``semantic_classes.json``), lossless PNG;
- ``observation.traversability_mask``   ``trav_mask``: 1 driven, 0 seen not
  driven, 255 no data (all 255 when less than 10 m of path is ahead);
- ``observation.state``                 odom -> base_link pose relative to the
  first frame, ``[x, y, z, qx, qy, qz, qw]``;
- ``action``                            the last ``/cmd_vel`` sent
  ``[linear_x, angular_z]`` (zeros before the first command);
- ``observation.frame_info``            ``[scan_index, scored, sim_time_ms]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from . import config
from .extract import pose7_to_matrix
from .labeling import TRAV_MASK

FPS = 15
STRIDE = 4                    # 60 Hz sim ticks -> 15 Hz
ROBOT_TYPE = "barakuda_isaac_sim"
TASK = "drive the robot along the off-road trail"
DOWNSCALE = 2                 # 1280x720 -> 640x360
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["linear_x", "angular_z"]
FRAME_INFO_NAMES = ["scan_index", "scored", "sim_time_ms"]


def _stamp_index(seq_dir: Path, name: str) -> dict[int, int]:
    """Stamp (integer microseconds) -> file index of an image channel."""
    ts = np.loadtxt(seq_dir / name / "timestamps.txt", ndmin=1)
    return {int(round(t * 1e6)): i for i, t in enumerate(ts)}


def _mask3(mask: np.ndarray, size) -> np.ndarray:
    small = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
    return np.repeat(small[:, :, None], 3, axis=2)


def frames(seq_dir, limit: int | None = None):
    """Yield LeRobot frame dicts for one labelled sequence directory."""
    seq_dir = Path(seq_dir)
    label_files = sorted((seq_dir / TRAV_MASK).glob("*.png"))
    label_ts = np.loadtxt(seq_dir / TRAV_MASK / "timestamps.txt", ndmin=1)
    if len(label_files) != len(label_ts):
        raise ValueError(f"{seq_dir}: {len(label_files)} trav masks for {len(label_ts)} stamps")
    rgb_of = _stamp_index(seq_dir, config.RGB)
    sem_of = _stamp_index(seq_dir, config.SEMANTIC)
    scored = set(np.load(seq_dir / "scored_frames.npy").tolist())

    pose_ts = np.loadtxt(seq_dir / config.POSE / "timestamps.txt", ndmin=1)
    poses = np.load(seq_dir / config.POSE / f"{config.POSE}.npy")
    cmd_path = seq_dir / config.CMD_VEL / f"{config.CMD_VEL}.npy"
    cmd_ts = np.loadtxt(seq_dir / config.CMD_VEL / "timestamps.txt", ndmin=1) if cmd_path.exists() else np.zeros(0)
    cmds = np.load(cmd_path) if cmd_path.exists() else np.zeros((0, 6))

    picks = list(range(0, len(label_files), STRIDE))
    if limit is not None:
        picks = picks[:limit]
    origin = None
    for i in picks:
        t = float(label_ts[i])
        key = int(round(t * 1e6))
        if key not in rgb_of or key not in sem_of:
            raise ValueError(f"{seq_dir}: no image on the stamp of scan {label_files[i].stem}")
        rgb = Image.open(seq_dir / config.RGB / f"{rgb_of[key]:06d}.png").convert("RGB")
        size = (rgb.width // DOWNSCALE, rgb.height // DOWNSCALE)
        sem = np.asarray(Image.open(seq_dir / config.SEMANTIC / f"{sem_of[key]:06d}.png"))
        trav = np.asarray(Image.open(label_files[i]))
        j = int(np.argmin(np.abs(pose_ts - t)))
        T = pose7_to_matrix(poses[j])
        if origin is None:
            origin = np.linalg.inv(T)
        pose = origin @ T
        c = int(np.searchsorted(cmd_ts, t, side="right")) - 1
        action = [cmds[c, 0], cmds[c, 5]] if c >= 0 else [0.0, 0.0]
        scan = int(label_files[i].stem)
        yield {
            "observation.images.front": np.asarray(rgb.resize(size, Image.BILINEAR)),
            "observation.semantic_mask": _mask3(sem, size),
            "observation.traversability_mask": _mask3(trav, size),
            "observation.state": np.concatenate(
                [pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]).astype(np.float32),
            "observation.frame_info": np.array([scan, int(scan in scored), int(round(t * 1000))], dtype=np.int64),
            "action": np.asarray(action, dtype=np.float32),
            "task": TASK,
        }


def features(height: int, width: int) -> dict:
    hwc = {"shape": (height, width, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.front": {"dtype": "video", **hwc},
        # Lossless PNG: mp4 would corrupt the label values.
        "observation.semantic_mask": {"dtype": "image", **hwc},
        "observation.traversability_mask": {"dtype": "image", **hwc},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.frame_info": {"dtype": "int64", "shape": (3,), "names": FRAME_INFO_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
    }


def to_lerobot(seq_dirs, repo_id: str, *, lerobot_root, limit: int | None = None):
    """Write the sequences to a local LeRobotDataset (one episode each). Never pushes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    seq_dirs = [Path(s) for s in seq_dirs]
    probe = next(frames(seq_dirs[0], limit=1))
    h, w, _ = probe["observation.images.front"].shape
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features(h, w),
                                    robot_type=ROBOT_TYPE, root=lerobot_root)
    for seq_dir in seq_dirs:
        for frame in frames(seq_dir, limit=limit):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return dataset


def check(repo_id: str, lerobot_root) -> dict:
    """Reload a written dataset and report its features, episodes and value sets."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=lerobot_root)
    hf = dataset.hf_dataset.with_format(None)
    values = {}
    for key in ("observation.semantic_mask", "observation.traversability_mask"):
        seen = set()
        for image in hf[key]:
            seen |= set(np.unique(np.asarray(image)[..., 0]).tolist())
        values[key] = sorted(seen)
    info = np.asarray(hf["observation.frame_info"])
    action = np.asarray(hf["action"])
    item = dataset[0]  # decodes the video frame too
    return {
        "features": sorted(dataset.meta.features),
        "episodes": dataset.meta.total_episodes,
        "frames": dataset.meta.total_frames,
        "fps": dataset.meta.fps,
        "values": values,
        "scored_frames": int(info[:, 1].sum()),
        "sim_time_s": [float(info[:, 2].min()) / 1000, float(info[:, 2].max()) / 1000],
        "linear_x_median": float(np.median(action[:, 0])),
        "item0_shapes": {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")},
    }
