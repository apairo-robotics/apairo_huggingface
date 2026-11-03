import numpy as np
import pyarrow.parquet as pq
import pytest

from apairo_huggingface.fomo.cli import main
from apairo_huggingface.fomo.point_labels import mask

STAMPS_US = [1760456485500467, 1760456485600469, 1760456485700470]


@pytest.fixture
def trajectory(tmp_path):
    traj = tmp_path / "fomo" / "2025-10-14" / "red_2025-10-14-11-41"
    rng = np.random.default_rng(0)
    labels = []
    for channel in ("trav_traj", "kissicp_poses"):
        (traj / channel).mkdir(parents=True)
        (traj / channel / "timestamps.txt").write_text("".join(f"{t * 1e-6:.18e}\n" for t in STAMPS_US))
    for i, n in enumerate((1000, 0, 2500)):
        lab = (rng.random(n) < 0.02).astype(np.uint8)
        np.save(traj / "trav_traj" / f"{i:06d}.npy", lab)
        pose = np.eye(4)
        pose[:3, 3] = [i, 0.5 * i, 0.0]
        np.save(traj / "kissicp_poses" / f"{i:06d}.npy", pose)
        labels.append((lab, pose))
    return traj, labels


def test_masks_rebuilt_from_the_indices_equal_the_original_labels(tmp_path, trajectory):
    traj, labels = trajectory
    out = tmp_path / "out"
    main(["export-labels", str(tmp_path / "fomo"), "--out", str(out)])

    table = pq.read_table(out / "data" / "2025-10-14" / "red_2025-10-14-11-41.parquet").to_pylist()
    assert [r["robosense_file"] for r in table] == [f"{t}.bin" for t in STAMPS_US]
    assert [r["scan_stem"] for r in table] == ["000000", "000001", "000002"]
    for row, (lab, pose) in zip(table, labels):
        assert row["trajectory"] == "2025-10-14/red_2025-10-14-11-41"
        assert row["n_points"] == len(lab)
        np.testing.assert_array_equal(mask(row["n_points"], row["driven_indices"]), lab)
        np.testing.assert_array_equal(np.reshape(row["pose"], (4, 4)), pose)


def test_real_scans_round_trip():
    """The rebuilt mask equals the original npy on a few real scans, when present."""
    from pathlib import Path

    from apairo_huggingface.fomo.point_labels import trajectory_table

    traj = Path("/mnt/vault-fellowship/fomo/2025-10-14/red_2025-10-14-11-41")
    if not (traj / "trav_traj" / "timestamps.txt").exists():
        pytest.skip("real FoMo trajectory not available")
    rows = trajectory_table(traj).to_pylist()
    for row in rows[::400]:
        lab = np.load(traj / "trav_traj" / f"{row['scan_stem']}.npy")
        np.testing.assert_array_equal(mask(row["n_points"], row["driven_indices"]), lab)
