import numpy as np
import pytest

from apairo.testing import check_format
from apairo_huggingface.fomo.format import RECORD, FomoBinFormat, FomoBinLoader


def _scan(path, n, t0):
    rec = np.zeros(n, dtype=RECORD)
    rec["x"] = np.arange(n, dtype=np.float32)
    rec["y"] = 1.0
    rec["z"] = 2.0
    rec["i"] = 3.0
    rec["r"] = np.arange(n) % 128
    rec["t"] = t0 + np.arange(n) * 10  # µs
    rec.tofile(path)
    return rec


def _channel(tmp_path, frames=3, n=5):
    for k in range(frames):
        t0 = 1_760_456_485_500_000 + k * 100_000
        _scan(tmp_path / f"{t0}.bin", n, t0)
    return tmp_path


def test_follows_the_format_contract(tmp_path):
    check_format(FomoBinFormat(), _channel(tmp_path))


def test_default_fields_are_an_n_by_4_cloud(tmp_path):
    loader = FomoBinLoader(str(_channel(tmp_path)))
    frame = loader[0]
    assert frame.shape == (5, 4) and frame.dtype == np.float32
    np.testing.assert_array_equal(frame[:, 0], np.arange(5))
    np.testing.assert_array_equal(frame[0, 1:], [1.0, 2.0, 3.0])


def test_t_is_seconds_from_the_scan_start(tmp_path):
    loader = FomoBinLoader(str(_channel(tmp_path)), fields=["x", "y", "z", "t"])
    np.testing.assert_allclose(loader[1][:, 3], np.arange(5) * 1e-5, rtol=1e-6)


def test_unknown_field_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no field"):
        FomoBinLoader(str(_channel(tmp_path)), fields=["x", "intensity"])
    assert FomoBinFormat().validate("lidar", {"fields": ["intensity"]}, tmp_path)


def test_truncated_scan_is_refused(tmp_path):
    loader = FomoBinLoader(str(_channel(tmp_path)))
    path = tmp_path / loader.files[0]
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="26-byte"):
        loader[0]


def test_detect_leaves_kitti_scans_to_the_builtin_bin(tmp_path):
    np.zeros((13, 4), np.float32).tofile(tmp_path / "000000.bin")  # 208 B: 16 and 26 divide it
    assert not FomoBinFormat().detect(tmp_path)


def test_entry_point_makes_the_format_available():
    from apairo.core.formats import get_format

    assert get_format("fomo_bin").name == "fomo_bin"
