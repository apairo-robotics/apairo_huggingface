import numpy as np
import pytest
from PIL import Image

from apairo_huggingface.fomo import config
from apairo_huggingface.fomo.cli import _trajectories


def test_trajectories_are_the_dirs_holding_the_camera_mask(tmp_path):
    for traj in ("2025-01-10/red_a", "2025-03-10/red_b"):
        mask = tmp_path / traj / config.TRAV_MASKS["zedx_left"]
        mask.mkdir(parents=True)
        (mask / "timestamps.txt").write_text("0.0\n")
    (tmp_path / "2025-04-15/red_c" / config.TRAV_MASKS["zedx_left"]).mkdir(parents=True)  # unfinished
    basler = tmp_path / "2025-05-28/red_d" / config.TRAV_MASKS["basler"]                 # Basler only
    basler.mkdir(parents=True)
    (basler / "timestamps.txt").write_text("0.0\n")
    (tmp_path / "2025-06-26/red_e/trav_mask").mkdir(parents=True)  # older flipped-Basler masks
    (tmp_path / "2025-06-26/red_e/trav_mask/timestamps.txt").write_text("0.0\n")

    assert config.CAMERA == "zedx_left"
    assert _trajectories(tmp_path) == [tmp_path / "2025-01-10/red_a", tmp_path / "2025-03-10/red_b"]
    assert _trajectories(tmp_path, "basler") == [tmp_path / "2025-05-28/red_d"]


def _images(traj, camera, rgba):
    (traj / camera).mkdir(parents=True)
    for i in range(5):
        Image.fromarray(rgba).save(traj / camera / f"{1_700_000_000_000_000 + i}.png")


def test_exclusion_is_measured_on_the_exported_camera(tmp_path):
    from apairo_huggingface.fomo.export import excluded, image_quality

    traj = tmp_path / "2025-01-29/red_2025-01-30-09-58"
    white = np.full((12, 16, 4), 255, dtype=np.uint8)
    scene = np.random.default_rng(0).integers(0, 200, (12, 16, 4)).astype(np.uint8)
    scene[..., 3] = 255                                   # opaque alpha must not count
    _images(traj, "basler", white)
    _images(traj, "zedx_left", scene)

    assert image_quality(traj, "basler")["saturated_share"] == 1.0
    assert excluded(traj, "basler")
    assert image_quality(traj, "zedx_left")["saturated_share"] == 0.0
    assert not excluded(traj, "zedx_left")


def test_check_scans_accepts_matching_scans_and_rejects_the_rest(tmp_path):
    from apairo_huggingface.fomo.format import RECORD
    from apairo_huggingface.fomo.labeling import check_scans

    stamps = [1732203312000459, 1732203312100461]
    (tmp_path / "robosense").mkdir()
    (tmp_path / "trav_traj").mkdir()
    (tmp_path / "trav_traj" / "timestamps.txt").write_text("".join(f"{t * 1e-6:.6f}\n" for t in stamps))
    for i, (t, n) in enumerate(zip(stamps, (10, 7))):
        np.zeros(n, dtype=RECORD).tofile(tmp_path / "robosense" / f"{t}.bin")
        np.save(tmp_path / "trav_traj" / f"{i:06d}.npy", np.zeros(n, dtype=np.uint8))
    assert check_scans(tmp_path) == 2

    np.zeros(8, dtype=RECORD).tofile(tmp_path / "robosense" / f"{stamps[1]}.bin")
    with pytest.raises(ValueError, match="points"):
        check_scans(tmp_path)
    (tmp_path / "robosense" / f"{stamps[1]}.bin").unlink()
    with pytest.raises(ValueError, match="stamps"):
        check_scans(tmp_path)
