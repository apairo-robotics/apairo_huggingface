"""Processed RELLIS-3D sequences -> one LeRobotDataset, one episode per sequence.

Every camera frame is a LeRobot frame (10 Hz): the image, its projected lidar
label mask, its traversability mask, the official 2D annotation where one
exists, the pose and the official lidar split of its co-captured scan.

- **Poses** come from ``poses.txt`` (one per scan, Ouster frame). The Ouster's
  +x points to the robot's *rear*, so poses are re-expressed in a robot frame
  turned 180 deg about z (+x forward, +z up, origin at the lidar) and made
  relative to the first frame of the sequence.
- **Action**: RELLIS-3D's KITTI-format release has no command topic, so the
  action is the motion actually executed, from the poses: forward speed and
  yaw rate between this frame and the next (the last frame repeats the
  previous one).
- **frame_info** = ``[split, scan_index, has_label_2d]``: the official lidar
  split of the frame's scan (``pt_{train,val,test}.lst``, which covers every
  scan; 0 train / 1 val / 2 test), the RELLIS scan / image index, and whether
  an official 2D annotation exists.
- **label_2d**: the official annotation, 255 on frames without one.

Frames are keyed by their image: the masks are matched to each image by stamp,
checked to carry the image's file stem, and an image of the requested range
without its masks is an error, never a silently shorter episode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from . import config
from .labeling import load_poses, open_view

FPS = 10
ROBOT_TYPE = "clearpath_warthog"
TASK = "drive the robot along the off-road trail"
DOWNSCALE = 2                 # 1920x1200 -> 960x600
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["linear_velocity", "angular_velocity"]
FRAME_INFO_NAMES = ["split", "scan_index", "has_label_2d"]
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
LIDAR_FROM_ROBOT = np.diag([-1.0, -1.0, 1.0, 1.0])  # 180 deg about z


def lidar_splits(root) -> dict[tuple[str, int], int]:
    """``(seq, scan index) -> split id`` from ``pt_{train,val,test}.lst``."""
    out = {}
    for name, sid in SPLIT_IDS.items():
        for line in (Path(root) / f"pt_{name}.lst").read_text().split("\n"):
            if line.strip():
                scan = line.split()[0]                   # 00000/os1_cloud_node_kitti_bin/000307.bin
                seq, _, fname = scan.split("/")
                out[(seq, int(Path(fname).stem))] = sid
    return out


def _mask3(mask: np.ndarray, size) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    small = np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST))
    return np.repeat(small[:, :, None], 3, axis=2)


def frames(root, view, splits, limit: int | None = None):
    """Yield LeRobot frame dicts for one processed sequence view."""
    view = Path(view)
    seq = view.name
    ds = open_view(view, [config.CAMERA, config.SEM_MASK, config.TRAV_MASK])
    sync = ds.synchronize(
        reference=config.CAMERA,
        method={config.SEM_MASK: "nearest", config.TRAV_MASK: "nearest"},
        tolerance=config.TOLERANCE,
    )
    for key in (config.SEM_MASK, config.TRAV_MASK):
        if len(sync) and np.abs(sync.time_offsets(key)).max() > 1e-3:
            raise ValueError(f"{seq}: {key} is not on the camera clock")
    camera_files = ds.loaders[config.CAMERA].files
    stamps = np.asarray(ds.timestamps[config.CAMERA], dtype=np.float64)
    n = len(camera_files) if limit is None else min(limit, len(camera_files))
    # synchronize() drops the images without masks: index by the image itself.
    image_of = np.asarray(sync.frame_indices[config.CAMERA])
    missing = sorted(set(range(n)) - set(image_of.tolist()))
    if missing:
        raise ValueError(f"{seq}: {len(missing)} of the first {n} images have no sem_mask / trav_mask "
                         f"(first: {Path(camera_files[missing[0]]).stem}); run process on the "
                         f"whole sequence or export with a smaller --limit")
    poses = load_poses(root, seq) @ LIDAR_FROM_ROBOT       # T_world_robot, one per scan = per image
    if len(poses) != len(camera_files):
        raise ValueError(f"{seq}: {len(poses)} poses for {len(camera_files)} images")
    origin = np.linalg.inv(poses[0])
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    mask_files = {key: ds.loaders[key].files for key in (config.SEM_MASK, config.TRAV_MASK)}
    for k in np.flatnonzero(image_of < n):
        k, i = int(k), int(image_of[k])                    # view position, image (= scan) index
        stem = Path(camera_files[i]).stem
        for key, files in mask_files.items():
            got = Path(files[int(sync.frame_indices[key][k])]).stem
            if got != stem:
                raise ValueError(f"{seq}: image {stem} was matched to {key} {got}")
        sample = sync[k]
        image = Image.fromarray(np.asarray(sample.data[config.CAMERA]))
        size = (image.width // DOWNSCALE, image.height // DOWNSCALE)
        ref_path = view / config.LABEL_2D / f"{stem}.png"
        if ref_path.exists():
            label_2d, has_2d = np.asarray(Image.open(ref_path)), 1
        else:
            label_2d, has_2d = np.full((image.height, image.width), config.NO_DATA, np.uint8), 0
        a = min(i, len(poses) - 2)                          # forward difference i -> i+1
        dt = stamps[a + 1] - stamps[a]
        step = np.linalg.inv(poses[a]) @ poses[a + 1]
        pose = origin @ poses[i]
        yield {
            "observation.images.front": np.asarray(image.resize(size, Image.BILINEAR)),
            "observation.semantic_mask": _mask3(np.asarray(sample.data[config.SEM_MASK]), size),
            "observation.traversability_mask": _mask3(np.asarray(sample.data[config.TRAV_MASK]), size),
            "observation.label_2d": _mask3(label_2d, size),
            "observation.state": np.concatenate(
                [pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]).astype(np.float32),
            "observation.frame_info": np.array([splits[(seq, i)], i, has_2d], dtype=np.int64),
            "action": np.array([step[0, 3] / dt, (yaw[a + 1] - yaw[a]) / dt], dtype=np.float32),
            "task": TASK,
        }


def features(height: int, width: int) -> dict:
    hwc = {"shape": (height, width, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.front": {"dtype": "video", **hwc},
        # Lossless PNG: mp4 would corrupt the label values.
        "observation.semantic_mask": {"dtype": "image", **hwc},
        "observation.traversability_mask": {"dtype": "image", **hwc},
        "observation.label_2d": {"dtype": "image", **hwc},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.frame_info": {"dtype": "int64", "shape": (3,), "names": FRAME_INFO_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
    }


def to_lerobot(root, views, repo_id: str, *, lerobot_root=None, push: bool = False,
               limit: int | None = None):
    """Write the sequences to a LeRobotDataset (one episode each).

    ``root`` is the RELLIS-3D source tree, ``lerobot_root`` the output directory."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    views = [Path(v) for v in views]
    splits = lidar_splits(root)
    probe = next(frames(root, views[0], splits, limit=1))
    h, w, _ = probe["observation.images.front"].shape
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=features(h, w),
        robot_type=ROBOT_TYPE, root=lerobot_root,
    )
    for view in views:
        for frame in frames(root, view, splits, limit=limit):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    if push:
        dataset.push_to_hub()
    return dataset


def check(repo_id: str, lerobot_root) -> dict:
    """Reload a written dataset and report its features, episodes and value sets."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=lerobot_root)
    hf = dataset.hf_dataset.with_format(None)
    values = {}
    for key in ("observation.semantic_mask", "observation.traversability_mask", "observation.label_2d"):
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
        "values": values,
        "split_counts": {name: int((info[:, 0] == sid).sum()) for name, sid in SPLIT_IDS.items()},
        "frames_with_label_2d": int(info[:, 2].sum()),
        "linear_velocity_median": float(np.median(action[:, 0])),
        "item0_shapes": {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")},
    }
