import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from apairo_huggingface.fomo import config
from apairo_huggingface.fomo.camera_labels import decode_mask, export, overlay, preview_stamps


def _masks(traj, camera, stamps):
    d = traj / config.TRAV_MASKS[camera]
    d.mkdir(parents=True)
    masks = []
    for i, us in enumerate(stamps):
        m = np.full((12, 16), 255, dtype=np.uint8)
        m[8:, 4 + i:10] = 1
        m[2, ::3] = 0
        Image.fromarray(m).save(d / f"{us}.png")
        masks.append(m)
    (d / "timestamps.txt").write_text("".join(f"{us * 1e-6:.6f}\n" for us in stamps))
    return masks


def test_tables_are_keyed_by_the_raw_image_stamp_and_lossless(tmp_path):
    traj = tmp_path / "2024-11-21" / "red_2024-11-21-10-34"
    stamps = [1732203252090223, 1732203252223546, 1732203252290188]
    written = {c: _masks(traj, c, stamps) for c in ("zedx_left", "basler")}
    export([traj], tmp_path / "out", cameras=("zedx_left", "basler"))

    for camera, masks in written.items():
        rows = pq.read_table(tmp_path / "out" / camera / "2024-11-21" / "red_2024-11-21-10-34.parquet").to_pylist()
        assert [r["image_file"] for r in rows] == [f"{us}.png" for us in stamps]
        assert [r["timestamp_us"] for r in rows] == stamps
        assert {r["footprint"] for r in rows} == {"past" if camera == "basler" else "future"}
        for r, m in zip(rows, masks):
            np.testing.assert_array_equal(decode_mask(r["mask_png"]), m)


def test_preview_takes_one_frame_every_period():
    us = [0, 66_000, 133_000] + [10_000_000 + 100_000 * i for i in range(5)] + [20_500_000]
    assert preview_stamps(us, period_s=10.0) == [0, 10_000_000, 20_500_000]


def test_overlay_marks_driven_and_seen_and_keeps_no_data():
    img = np.full((2, 3, 4), 100, dtype=np.uint8)            # RGBA, as the ZED PNGs
    mask = np.array([[1, 0, 255], [255, 255, 1]], dtype=np.uint8)
    out = overlay(img, mask)
    assert out.shape == (2, 3, 3)
    assert out[0, 0, 1] > out[0, 0, 0]                        # green tint
    assert tuple(out[0, 1]) == (255, 40, 40)                  # red
    assert tuple(out[0, 2]) == (100, 100, 100)                # untouched
    Image.fromarray(out).save(io.BytesIO(), "JPEG")
