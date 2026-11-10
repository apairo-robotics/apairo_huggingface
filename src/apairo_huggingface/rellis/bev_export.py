"""BEV-processed RELLIS-3D sequences -> one LeRobotDataset, one episode per sequence.

Every scan is a LeRobot frame (10 Hz), with the camera image co-captured with
it and its three BEV grids:

- ``observation.bev.elevation``  ``(3, 200, 200)`` float32: min / max / mean
  height above the ground under the robot, NaN unknown (stored as float32: the
  apairo channel is float16, LeRobot's statistics want float32);
- ``observation.bev.semantic``   ``(200, 200, 3)`` PNG: majority ontology id in
  every channel, 255 unknown;
- ``observation.bev.trav``       ``(200, 200, 3)`` PNG: 1 driven / 0 seen, not
  driven / 255 unknown.

Grid layout (``config.BEV_*``): 0.2 m cells, x in [-10, 30) m ahead, y in
[-20, 20) m left, row 0 = 30 m ahead, column 0 = 20 m left -- the robot drives
up the image. ``observation.state``, ``action`` and ``observation.frame_info``
are those of :mod:`.export` (pose relative to the first frame in a robot
frame, executed forward speed and yaw rate, ``[split, scan_index,
has_label_2d]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..poses import executed_actions, relative_states
from . import config
from .bev import declaration, grid_shape, poses_from_rows
from .export import (ACTION_NAMES, FPS, FRAME_INFO_NAMES, LIDAR_FROM_ROBOT, ROBOT_TYPE, SPLIT_IDS,
                     STATE_NAMES, TASK, lidar_splits)

DOWNSCALE = 4                 # 1920x1200 -> 480x300: context for the grids, not the product
ELEVATION_NAMES = ["layer", "row", "col"]
BEV_KEYS = [config.CAMERA, config.BEV_POSES, config.BEV_ELEVATION, config.BEV_SEMANTIC, config.BEV_TRAV]


def _grid3(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(grid, dtype=np.uint8)[:, :, None], 3, axis=2)


def frames(root, view, splits, limit: int | None = None):
    """Yield LeRobot frame dicts for one BEV-processed sequence view."""
    from apairo import RawDataset

    view = Path(view)
    seq = view.name
    ds = RawDataset(view, declare=declaration(), keys=BEV_KEYS)
    n_scans = len(ds.timestamps[config.BEV_POSES])
    for key in BEV_KEYS:
        if len(ds.timestamps[key]) != n_scans:
            raise ValueError(f"{seq}: {key} has {len(ds.timestamps[key])} frames for {n_scans} scans")
        if np.abs(np.asarray(ds.timestamps[key]) - ds.timestamps[config.BEV_POSES]).max() > 1e-3:
            raise ValueError(f"{seq}: {key} is not on the scan clock")
    stamps = np.asarray(ds.timestamps[config.BEV_POSES], dtype=np.float64)
    poses_loader = ds.loaders[config.BEV_POSES]
    poses = poses_from_rows(np.stack([poses_loader[i] for i in range(n_scans)])) @ LIDAR_FROM_ROBOT
    states, actions = relative_states(poses), executed_actions(poses, stamps)
    camera_files = ds.loaders[config.CAMERA].files
    label_2d_dir = Path(root) / "Rellis-3D" / seq / config.LABEL_2D
    n = n_scans if limit is None else min(limit, n_scans)
    for k in range(n):
        image = Image.fromarray(np.asarray(ds.loaders[config.CAMERA][k]))
        size = (image.width // DOWNSCALE, image.height // DOWNSCALE)
        has_2d = int((label_2d_dir / f"{Path(camera_files[k]).stem}.png").exists())
        yield {
            "observation.images.front": np.asarray(image.resize(size, Image.BILINEAR)),
            "observation.bev.elevation": np.asarray(ds.loaders[config.BEV_ELEVATION][k], dtype=np.float32),
            "observation.bev.semantic": _grid3(ds.loaders[config.BEV_SEMANTIC][k]),
            "observation.bev.trav": _grid3(ds.loaders[config.BEV_TRAV][k]),
            "observation.state": states[k],
            "observation.frame_info": np.array([splits[(seq, k)], k, has_2d], dtype=np.int64),
            "action": actions[k],
            "task": TASK,
        }


def features(height: int, width: int) -> dict:
    h, w = grid_shape()
    grid = {"shape": (h, w, 3), "names": ["height", "width", "channel"]}
    return {
        "observation.images.front": {"dtype": "video", "shape": (height, width, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.bev.elevation": {"dtype": "float32", "shape": (3, h, w), "names": ELEVATION_NAMES},
        # Lossless PNG: mp4 would corrupt the label values.
        "observation.bev.semantic": {"dtype": "image", **grid},
        "observation.bev.trav": {"dtype": "image", **grid},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.frame_info": {"dtype": "int64", "shape": (3,), "names": FRAME_INFO_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
    }


def to_lerobot(root, views, repo_id: str, *, lerobot_root=None, limit: int | None = None):
    """Write the BEV sequences to a local LeRobotDataset (one episode each). Never pushed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    views = [Path(v) for v in views]
    splits = lidar_splits(root)
    probe = next(frames(root, views[0], splits, limit=1))
    h, w, _ = probe["observation.images.front"].shape
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features(h, w),
                                    robot_type=ROBOT_TYPE, root=lerobot_root)
    for view in views:
        for frame in frames(root, view, splits, limit=limit):
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return dataset


def check(repo_id: str, lerobot_root) -> dict:
    """Reload a written BEV dataset and report its features, episodes, value sets and grid statistics."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=lerobot_root)
    hf = dataset.hf_dataset.with_format(None)
    values = {}
    for key in ("observation.bev.semantic", "observation.bev.trav"):
        seen = set()
        for image in hf[key]:
            seen |= set(np.unique(np.asarray(image)[..., 0]).tolist())
        values[key] = sorted(seen)
    trav = np.stack([np.asarray(im)[..., 0] for im in hf["observation.bev.trav"]])
    info = np.asarray(hf["observation.frame_info"])
    item = dataset[len(dataset) // 2]   # decodes the video frame too
    elev = item["observation.bev.elevation"].numpy()
    return {
        "features": sorted(dataset.meta.features),
        "episodes": dataset.meta.total_episodes,
        "frames": dataset.meta.total_frames,
        "values": values,
        "trav_cell_shares": {name: float((trav == v).mean()) for name, v in
                             (("driven", 1), ("seen_not_driven", 0), ("unknown", 255))},
        "split_counts": {name: int((info[:, 0] == sid).sum()) for name, sid in SPLIT_IDS.items()},
        "item_mid_shapes": {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")},
        "item_mid_elevation_known_share": float(np.isfinite(elev).mean()),
        "item_mid_mean_height_median": float(np.nanmedian(elev[2])),
    }
