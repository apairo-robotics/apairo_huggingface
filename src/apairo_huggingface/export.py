"""The reusable sink: an apairo TartanDrive dataset -> a LeRobotDataset.

This module is where apairo's value shows: `to_lerobot` reads top to bottom as
load -> align -> copy frames -> finalize, because apairo already did the loading,
resampling and episode segmentation. The per-frame domain logic is three
functions in :mod:`.mappings`.
"""

from __future__ import annotations

import numpy as np
from apairo import TartanKittiDataset

from . import config
from .features import build_features
from .mappings import CHANNEL_TRANSFORMS, frame_of


def load_tartan(root: str):
    """Load the TartanDrive channels feeding the export: camera, state, action,
    and the traversability mask written by :mod:`.labeling`.

    ``root`` may be a TartanDrive root (many recordings) or a single sequence.
    A root construction fails as a whole when any sequence lacks a channel
    (e.g. one recording was downloaded without the camera modality, or was
    never labeled); in that case fall back to probing per sequence and keep
    only the complete ones.
    """
    keys = [config.CAMERA, config.STATE, config.ACTION, config.TRAV_MASK]
    try:
        return TartanKittiDataset(root, keys=keys)
    except KeyError:
        probe = TartanKittiDataset(root, keys=[config.ACTION])
        if not getattr(probe, "_is_root", False):
            raise
        complete = [s for s in probe.sequences if set(keys) <= set(s.available)]
        skipped = sorted(set(s.name for s in probe.sequences) - set(s.name for s in complete))
        if not complete:
            raise
        print(f"skipping {len(skipped)} sequence(s) missing {keys}: {', '.join(skipped)}")
        return [TartanKittiDataset(s.root_dir, keys=keys) for s in complete]


def episodes(tartan) -> list:
    """One apairo dataset per recording. A root holds many (`.sequences`); a
    single sequence directory is itself one episode; :func:`load_tartan` may
    already have expanded a partially-complete root into a list."""
    if isinstance(tartan, list):
        return tartan
    try:
        return list(tartan.sequences)
    except AttributeError:
        return [tartan]


def aligned_frames(episode):
    """Resample one recording onto the camera clock, then cast/slice each channel.

    Transforms are applied *after* ``synchronize`` on purpose: a transform set on
    the pre-sync dataset is dropped, because the synchronized view resamples from
    the raw channels.
    """
    frames = episode.synchronize(
        reference=config.REFERENCE, method=config.METHOD, tolerance=config.TOLERANCE
    )
    for channel, transform in CHANNEL_TRANSFORMS.items():
        frames = frames.transform(channel, transform)
    return frames


def to_lerobot(tartan: TartanKittiDataset, repo_id: str, *, push: bool = False, root=None):
    """Write an apairo TartanDrive dataset to a LeRobotDataset and return it.

    Args:
        tartan:   dataset from :func:`load_tartan` (root or single sequence).
        repo_id:  target LeRobotDataset id (``<user>/<name>`` when pushing).
        push:     upload to the Hugging Face Hub after writing.
        root:     local output directory (defaults to LeRobot's cache home).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    recordings = episodes(tartan)
    probe = aligned_frames(recordings[0])[0]

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.FPS,
        features=build_features(probe),
        robot_type=config.ROBOT_TYPE,
        root=root,
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


def validate(tartan: TartanKittiDataset):
    """Exercise the apairo read path without lerobot, asserting every mapped
    array matches the declared schema. Returns ``(features, per_episode_counts)``."""
    recordings = episodes(tartan)
    features = build_features(aligned_frames(recordings[0])[0])

    counts = []
    for recording in recordings:
        frames = aligned_frames(recording)
        for sample in frames:
            frame = frame_of(sample)
            for name, spec in features.items():
                shape = np.asarray(frame[name]).shape
                if shape != spec["shape"]:
                    raise ValueError(f"{name}: {shape} != {spec['shape']}")
        counts.append(len(frames))
    return features, counts
