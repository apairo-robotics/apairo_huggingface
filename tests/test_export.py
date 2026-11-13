"""End-to-end tests for the TartanDrive -> LeRobot export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from apairo_huggingface import aligned_frames, config, episodes, load_tartan, to_lerobot, validate


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


def _utm_odometry(seq, step=0.26, yaw=np.radians(30.0)):
    """Overwrite a recording's gps_odom with UTM-scale rows (x ~ 4.48e6 m, where
    float32 spacing is 0.5 m) driving ``step`` metres per frame along ``yaw``."""
    path = Path(seq) / config.STATE / "odometry.npy"
    rows = np.load(path)
    times = np.loadtxt(Path(seq) / config.STATE / "timestamps.txt")
    k = (times - times[0]) / (times[1] - times[0]) / 5   # frames since the first row (50 Hz rows, 10 Hz frames)
    rows[:, 0] = 4_479_748.098 + step * k * np.cos(yaw)
    rows[:, 1] = -603_117.905 + step * k * np.sin(yaw)
    rows[:, 2] = 300.516
    rows[:, 3:7] = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    np.save(path, rows)


def test_state_is_relative_to_the_episode_start_and_not_quantised(tartan_root):
    """UTM-scale positions cast straight to float32 fall on a 0.5 m grid; the
    exported state is the pose relative to the episode's first frame, in its
    vehicle frame, subtracted in float64 before the cast."""
    for seq in sorted(Path(tartan_root).iterdir()):
        if seq.is_dir():
            _utm_odometry(seq)
    for recording in episodes(load_tartan(tartan_root)):
        states = np.stack([np.asarray(s.data[config.STATE]) for s in aligned_frames(recording)])
        assert states.dtype == np.float32
        np.testing.assert_allclose(states[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-7)
        # 0.26 m straight ahead per frame, the heading folded into the first frame.
        expected = np.column_stack([0.26 * np.arange(len(states)), np.zeros((len(states), 5)),
                                    np.ones(len(states))])
        np.testing.assert_allclose(states, expected, atol=1e-4)


def test_dataset_card_states_the_code_s_tolerance_and_state():
    """The published card shows the export's own synchronization and state op."""
    card = (Path(__file__).parents[1] / "hub" / "dataset_card.md").read_text(encoding="utf-8")
    assert f"tolerance={config.TOLERANCE})" in card
    assert f"±{config.TOLERANCE * 1000:.0f} ms" in card
    assert "RelativeTo(origin)" in card
