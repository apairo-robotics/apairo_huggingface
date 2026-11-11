"""BEV-processed TartanDrive recordings -> one LeRobotDataset.

One episode per recording, cut into consecutive episodes of at most
``MAX_EPISODE_FRAMES`` frames, of equal length: LeRobot holds an episode's float grids in
memory (~8 MB of RSS per frame here, so ~8 GB for 1000 frames).
``observation.frame_info`` and ``meta/apairo_recordings.json`` map every
episode back to its recording and scans.

Every scan is a LeRobot frame (10 Hz). apairo's ``synchronize`` puts the
camera image and the ``cmd`` row nearest each scan on the scan clock (within
``config.TOLERANCE``, frames without a match are dropped); the BEV grids and
the pose are read at the scan itself:

- ``observation.images.front``   ``(272, 512, 3)`` video: the left camera, halved;
- ``observation.bev.elevation``  ``(3, 200, 200)`` float32: min / max / mean
  height above the ground under the robot, NaN unknown (float32: the apairo
  channel is float16, LeRobot's statistics want float32);
- ``observation.bev.geometry``   ``(2, 200, 200)`` float32: roughness / step (m), NaN unknown;
- ``observation.bev.trav``       ``(200, 200, 3)`` PNG: 1 driven / 0 seen, not
  driven / 255 unknown, the value in every channel;
- ``observation.state``          ``(7,)`` float32: GICP pose relative to the
  episode's first frame, ``[x, y, z, qx, qy, qz, qw]``, in that frame's
  vehicle frame (not gravity-aligned: the grids are, the state is not);
- ``action``                     ``(2,)`` float32: executed forward speed (m/s)
  and yaw rate (rad/s), from the GICP poses of this scan and the next one --
  the action of the RELLIS BEV export;
- ``observation.command``        ``(2,)`` float32: the raw ``cmd`` channel,
  ``[linear, angular]``, kept for provenance (its linear part is 0 in most
  recordings, the angular part is the steering command in [-1, 1]);
- ``observation.frame_info``     ``(3,)`` int64: ``[recording_index, scan_index,
  camera_index]``, the recording (its position in ``meta/apairo_recordings.json``),
  the scan and the matched image in it.

TartanDrive repeats some scans under the same stamp (identical points, pose
re-registered within mm); the export keeps one scan of each repeat -- the last:
``synchronize`` resolves a shared reference stamp to the later scan -- so
LeRobot frames are distinct instants.
Scans without a valid GICP pose or INS attitude (all-unknown grids) are not
exported either.
Grid layout: :mod:`apairo_huggingface.bev_grid` (row 0 = 30 m ahead, column
0 = 20 m left, the robot drives up the image).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from ..bev_grid import grid_shape
from ..poses import executed_actions, relative_states
from . import config
from .bev import GEOMETRY_NAMES, open_view, scan_poses, valid_scans

MAX_EPISODE_FRAMES = 1000
DOWNSCALE = 2                 # 1024x544 -> 512x272: context for the grids, not the product
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["forward_speed", "yaw_rate"]
COMMAND_NAMES = ["linear", "angular"]
SYNC_KEYS = [config.CAMERA, config.ACTION, config.BEV_TRAV]


def _grid3(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(grid, dtype=np.uint8)[:, :, None], 3, axis=2)


def scan_clock(view):
    """The recording on the scan clock: ``(ds, synced, keep)``, ``keep`` the synced
    frames exported: one per repeated stamp (``synchronize`` maps every tick of a
    shared stamp to the last scan at it, so that scan is the one kept), and only
    scans with a valid pose and attitude (the others' grids are all-unknown)."""
    ds = open_view(view, SYNC_KEYS + [config.POSES, config.POSE_VALID, config.ATTITUDE,
                                      config.BEV_ELEVATION, config.BEV_GEOMETRY])
    synced = ds.synchronize(reference=config.BEV_TRAV, method="nearest", tolerance=config.TOLERANCE)
    stamps = np.asarray(ds.timestamps[config.BEV_TRAV], dtype=np.float64)
    scans = np.asarray(synced.frame_indices[config.BEV_TRAV])
    first = np.concatenate([[True], np.diff(stamps[scans]) > 0])
    keep = np.flatnonzero(first & valid_scans(ds)[scans])
    return ds, synced, keep


def episode_bounds(n: int) -> list[tuple[int, int]]:
    """``n`` frames cut into equal consecutive episodes of at most ``MAX_EPISODE_FRAMES``
    (a 1-frame tail episode breaks LeRobot's stats aggregation): ``[(start, end), ...]``."""
    edges = np.linspace(0, n, -(-n // MAX_EPISODE_FRAMES) + 1).round().astype(int).tolist()
    return list(zip(edges[:-1], edges[1:]))


def frames(view, recording: int = 0, limit: int | None = None):
    """Yield LeRobot frame dicts for one BEV-processed recording view, episode
    after episode (:func:`episode_bounds`); the state restarts at each episode."""
    ds, synced, keep = scan_clock(Path(view))
    scans = np.asarray(synced.frame_indices[config.BEV_TRAV])[keep]
    cameras = np.asarray(synced.frame_indices[config.CAMERA])[keep]
    stamps = np.asarray(ds.timestamps[config.BEV_TRAV], dtype=np.float64)[scans]
    poses = scan_poses(ds)[scans]
    actions = executed_actions(poses, stamps)
    n = len(keep) if limit is None else min(limit, len(keep))
    for start, end in episode_bounds(n):
        states = relative_states(poses[start:end])
        for f in range(start, end):
            k = int(scans[f])
            sample = synced[int(keep[f])]
            image = Image.fromarray(np.asarray(sample.data[config.CAMERA]))
            yield {
                "observation.images.front": np.asarray(
                    image.resize((image.width // DOWNSCALE, image.height // DOWNSCALE), Image.BILINEAR)),
                "observation.bev.elevation": np.asarray(ds.loaders[config.BEV_ELEVATION][k], dtype=np.float32),
                "observation.bev.geometry": np.asarray(ds.loaders[config.BEV_GEOMETRY][k], dtype=np.float32),
                "observation.bev.trav": _grid3(sample.data[config.BEV_TRAV]),
                "observation.state": states[f - start],
                "observation.frame_info": np.array([recording, k, cameras[f]], dtype=np.int64),
                "observation.command": np.asarray(sample.data[config.ACTION], dtype=np.float32),
                "action": actions[f],
                "task": config.TASK,
            }


def features(height: int, width: int) -> dict:
    h, w = grid_shape()
    return {
        "observation.images.front": {"dtype": "video", "shape": (height, width, 3),
                                     "names": ["height", "width", "channel"]},
        "observation.bev.elevation": {"dtype": "float32", "shape": (3, h, w), "names": ["layer", "row", "col"]},
        "observation.bev.geometry": {"dtype": "float32", "shape": (len(GEOMETRY_NAMES), h, w),
                                     "names": ["layer", "row", "col"]},
        # Lossless PNG: mp4 would corrupt the label values.
        "observation.bev.trav": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.frame_info": {"dtype": "int64", "shape": (3,),
                                   "names": ["recording_index", "scan_index", "camera_index"]},
        "observation.command": {"dtype": "float32", "shape": (2,), "names": COMMAND_NAMES},
        "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
    }


def to_lerobot(views, repo_id: str, *, lerobot_root=None, limit: int | None = None):
    """Write the BEV recordings to a local LeRobotDataset. Never pushed."""
    import json

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    views = [Path(v) for v in views]
    probe = next(frames(views[0], limit=1))
    h, w, _ = probe["observation.images.front"].shape
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=config.FPS, features=features(h, w),
                                    robot_type=config.ROBOT_TYPE, root=lerobot_root)
    recordings = []
    for r, view in enumerate(views):
        n = len(scan_clock(view)[2])
        n = n if limit is None else min(limit, n)
        ends = {end for _, end in episode_bounds(n)}
        first_episode = dataset.meta.total_episodes
        for f, frame in enumerate(frames(view, recording=r, limit=limit)):
            dataset.add_frame(frame)
            if f + 1 in ends:
                dataset.save_episode()
        recordings.append({"recording": view.name,
                           "episodes": list(range(first_episode, dataset.meta.total_episodes))})
    dataset.finalize()
    (Path(dataset.root) / "meta" / "apairo_recordings.json").write_text(json.dumps(recordings, indent=2))
    return dataset


def check(repo_id: str, lerobot_root) -> dict:
    """Reload a written BEV dataset and report its episodes, value sets and ranges."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=lerobot_root)
    hf = dataset.hf_dataset.with_format(None)
    trav_values, counts = set(), np.zeros(256, dtype=np.int64)
    for image in hf["observation.bev.trav"]:
        grid = np.asarray(image)[..., 0]
        counts += np.bincount(grid.ravel(), minlength=256)
        trav_values |= set(np.flatnonzero(np.bincount(grid.ravel(), minlength=256)).tolist())
    episodes = np.asarray(hf["episode_index"])
    action = np.asarray(hf["action"])
    report = {
        "features": sorted(dataset.meta.features),
        "episodes": dataset.meta.total_episodes,
        "frames": dataset.meta.total_frames,
        "frames_per_episode": np.bincount(episodes).tolist(),
        "trav_values": sorted(trav_values),
        "trav_cell_shares": {name: float(counts[v] / counts.sum()) for name, v in
                             (("driven", 1), ("seen_not_driven", 0), ("unknown", 255))},
        "action_range": [action.min(axis=0).tolist(), action.max(axis=0).tolist()],
        "command_range": [np.asarray(hf["observation.command"]).min(axis=0).tolist(),
                          np.asarray(hf["observation.command"]).max(axis=0).tolist()],
    }
    # Grid value ranges on every 50th frame (decoding every float grid is slow).
    elev, geom = [], []
    for i in range(0, len(dataset), 50):
        row = hf[i]
        elev.append(np.asarray(row["observation.bev.elevation"], dtype=np.float32))
        geom.append(np.asarray(row["observation.bev.geometry"], dtype=np.float32))
    elev, geom = np.stack(elev), np.stack(geom)
    report.update({
        "elevation_known_share": float(np.isfinite(elev[:, 2]).mean()),
        "mean_height_p1_p50_p99": np.nanpercentile(elev[:, 2], [1, 50, 99]).round(3).tolist(),
        "roughness_p50_p99": np.nanpercentile(geom[:, 0], [50, 99]).round(3).tolist(),
        "step_p50_p99": np.nanpercentile(geom[:, 1], [50, 99]).round(3).tolist(),
    })
    item = dataset[len(dataset) // 2]   # decodes the video frame too
    report["item_mid_shapes"] = {k: tuple(v.shape) for k, v in item.items() if hasattr(v, "shape")}
    return report
