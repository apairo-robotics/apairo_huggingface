"""Export a TartanDrive v2 recording to a LeRobotDataset with apairo.

TartanDrive is an asynchronous multi-rate sensor rig: camera, GPS/odometry and
drive commands each tick on their own clock, one channel per file. LeRobot wants
the opposite — synchronous, per-episode frames where every modality is aligned.

apairo bridges that gap on the read side. `TartanKittiDataset` loads the raw
channels, `.sequences` exposes one recording per episode, and `.synchronize()`
resamples the rig onto a single clock. Each frame then comes back as a plain dict
of numpy arrays. The ONLY code left to write is the three tiny mappings below —
apairo channel -> LeRobot feature. That contrast is the whole point of this
example: the messy part is a library call, the domain part is three functions.

Usage:
    python export_tartan_to_lerobot.py --dry-run                 # validate reads, no lerobot
    python export_tartan_to_lerobot.py --no-push                 # build a local dataset
    python export_tartan_to_lerobot.py --repo-id me/tartan_offroad   # build + push to the Hub
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from apairo import TartanKittiDataset
from apairo.core.sample import Sample


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Which TartanDrive channels feed each LeRobot feature.
CAMERA = "image_left_color"   # (H, W, 3) uint8  -- the left colour camera
STATE = "gps_odom"            # nav_msgs/Odometry row: pose (7) + twist (6)
ACTION = "cmd"                # twist command [linear_velocity, angular_velocity]

# Synchronization: resample every channel onto the camera clock, so each LeRobot
# frame carries a *real* image (never an interpolated one) plus the state and
# command nearest in time (within TOLERANCE).
REFERENCE = CAMERA
METHOD = "nearest"
TOLERANCE = 0.05              # seconds; drop frames with no match within +/-50 ms
FPS = 10                      # nominal rate declared to LeRobot (camera ~10 Hz)

TASK = "drive off-road"

# gps_odom holds a full Odometry row; we keep the pose (position + orientation).
POSE_DIM = 7
STATE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
ACTION_NAMES = ["linear_velocity", "angular_velocity"]


# --------------------------------------------------------------------------- #
# Mappings: one apairo channel -> one LeRobot feature (the only domain code)
# --------------------------------------------------------------------------- #

def image_front(sample: Sample) -> np.ndarray:
    """Left colour camera, ready as-is for a LeRobot video feature (H, W, 3 uint8)."""
    return sample.data[CAMERA]


def odom_to_state(sample: Sample) -> np.ndarray:
    """Robot pose (x, y, z, quaternion) from the odometry row."""
    return sample.data[STATE][:POSE_DIM].astype(np.float32)


def twist_to_action(sample: Sample) -> np.ndarray:
    """Drive command as [linear_velocity, angular_velocity]."""
    return sample.data[ACTION].astype(np.float32)


def frame_of(sample: Sample) -> dict:
    """Assemble one LeRobot frame from an aligned apairo sample."""
    return {
        "observation.images.front": image_front(sample),
        "observation.state": odom_to_state(sample),
        "action": twist_to_action(sample),
        "task": TASK,
    }


# --------------------------------------------------------------------------- #
# LeRobot dataset schema, sized from a real frame
# --------------------------------------------------------------------------- #

def build_features(probe: Sample) -> dict:
    """Feature dict for LeRobotDataset.create, with the image shape taken from
    an actual frame so the example works at any camera resolution."""
    height, width, _ = image_front(probe).shape
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (POSE_DIM,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }


# --------------------------------------------------------------------------- #
# Read side: TartanDrive -> aligned per-episode frames (all apairo)
# --------------------------------------------------------------------------- #

def episodes(tartan: TartanKittiDataset) -> list:
    """One apairo dataset per recording. A root directory holds many recordings
    (`.sequences`); a single sequence directory is itself one episode."""
    try:
        return list(tartan.sequences)
    except AttributeError:
        return [tartan]


def aligned_frames(episode) -> object:
    """Resample one recording onto the camera clock -> synchronous frames."""
    return episode.synchronize(reference=REFERENCE, method=METHOD, tolerance=TOLERANCE)


# --------------------------------------------------------------------------- #
# Write side: aligned frames -> LeRobotDataset
# --------------------------------------------------------------------------- #

def export(root: str, repo_id: str, push: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tartan = TartanKittiDataset(root, keys=[CAMERA, STATE, ACTION])
    recordings = episodes(tartan)

    probe = aligned_frames(recordings[0])[0]
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=build_features(probe),
        robot_type="tartandrive",
    )

    # One TartanDrive recording = one LeRobot episode. apairo has already aligned
    # the multi-rate sensors, so all that remains is to copy the frames across.
    for recording in recordings:
        for sample in aligned_frames(recording):
            dataset.add_frame(frame_of(sample))
        dataset.save_episode()

    dataset.finalize()          # required in LeRobot v3 before the dataset is valid
    if push:
        dataset.push_to_hub()
    return dataset


def dry_run(root: str):
    """Exercise the apairo read pipeline without lerobot, asserting every mapped
    array matches the declared feature schema. Validates the half that shows
    apairo's value end-to-end."""
    tartan = TartanKittiDataset(root, keys=[CAMERA, STATE, ACTION])
    recordings = episodes(tartan)
    features = build_features(aligned_frames(recordings[0])[0])

    total = 0
    for i, recording in enumerate(recordings):
        frames = aligned_frames(recording)
        for sample in frames:
            frame = frame_of(sample)
            for name, spec in features.items():
                array = np.asarray(frame[name])
                assert array.shape == spec["shape"], f"{name}: {array.shape} != {spec['shape']}"
        total += len(frames)
        print(f"  episode {i}: {len(frames)} frames")

    print(f"\n{len(recordings)} episode(s), {total} frames -- schema OK")
    for name, spec in features.items():
        print(f"  {name:28} {spec['dtype']:8} {spec['shape']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=os.environ.get("APAIRO_TARTAN_ROOT", "/data/tartan"),
                        help="TartanDrive root (many recordings) or a single sequence directory")
    parser.add_argument("--repo-id", default="tartandrive_offroad",
                        help="target LeRobotDataset repo id")
    parser.add_argument("--no-push", action="store_true", help="build locally, do not push to the Hub")
    parser.add_argument("--dry-run", action="store_true", help="validate reads only, without lerobot")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.root)
    else:
        dataset = export(args.root, args.repo_id, push=not args.no_push)
        print(f"Wrote {dataset.num_episodes} episodes / {dataset.num_frames} frames to {args.repo_id}")


if __name__ == "__main__":
    main()
