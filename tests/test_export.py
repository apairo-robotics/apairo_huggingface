"""End-to-end tests for the TartanDrive -> LeRobot export."""

from __future__ import annotations

from apairo_huggingface import config, load_tartan, to_lerobot, validate


def test_validate_matches_schema(tartan_root, n_episodes, n_frames):
    """The apairo read path yields frames matching the declared feature schema."""
    features, counts = validate(load_tartan(tartan_root))

    assert len(counts) == n_episodes
    assert sum(counts) == n_frames
    assert features["observation.state"]["shape"] == (config.POSE_DIM,)
    assert features["action"]["shape"] == (len(config.ACTION_NAMES),)
    assert features["observation.images.front"]["dtype"] == "video"


def test_to_lerobot_roundtrip(tartan_root, tmp_path, n_episodes, n_frames):
    """Exporting produces a valid LeRobotDataset that reloads with the expected
    features and shapes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = tmp_path / "dataset"
    repo_id = "apairo-demo/tartandrive_offroad"

    written = to_lerobot(load_tartan(tartan_root), repo_id, push=False, root=out)
    assert written.num_episodes == n_episodes
    assert written.num_frames == n_frames

    reloaded = LeRobotDataset(repo_id, root=out)
    assert reloaded.num_episodes == n_episodes
    assert reloaded.fps == config.FPS

    frame = reloaded[0]
    assert frame["observation.state"].shape == (config.POSE_DIM,)
    assert frame["action"].shape == (len(config.ACTION_NAMES),)
    assert frame["task"] == config.TASK
    assert frame["observation.images.front"].ndim == 3   # (C, H, W)
