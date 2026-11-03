from apairo_huggingface.fomo.cli import _trajectories


def test_trajectories_are_the_dirs_holding_trav_mask(tmp_path):
    for traj in ("2025-01-10/red_a", "2025-03-10/red_b"):
        mask = tmp_path / traj / "trav_mask"
        mask.mkdir(parents=True)
        (mask / "timestamps.txt").write_text("0.0\n")
    (tmp_path / "2025-04-15/red_c/trav_mask").mkdir(parents=True)  # not processed

    assert _trajectories(tmp_path) == [tmp_path / "2025-01-10/red_a", tmp_path / "2025-03-10/red_b"]


def test_saturated_camera_trajectory_is_excluded_from_the_image_export(tmp_path):
    from apairo_huggingface.fomo.export import excluded

    assert excluded(tmp_path / "2025-01-29/red_2025-01-30-09-58")
    assert not excluded(tmp_path / "2025-01-10/red_2025-01-10-12-02")
