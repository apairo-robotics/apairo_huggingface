"""The card figure, from the ``preview`` tables: masks drawn over the raw images.

One row per trajectory (seasons when available), one column per camera
(ZED X left, ZED X right, rear Basler); the middle preview frame of each.

    python scripts/fomo_overlay.py --labels /mnt/vault-fellowship/fomo_labels --out hub/fomo_trav_mask_overlay.jpg
"""

import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw

CAMERAS = ("zedx_left", "zedx_right", "basler")
# One per season when available: snow, spring, summer, autumn.
PREFERRED = ("2025-01-10", "2025-05-28", "2025-08-20", "2025-10-14")
SCALE = 2                     # preview frames are 960 x 600; panels 480 x 300


def main() -> None:
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=4)
    args = p.parse_args()

    files = sorted((args.labels / "preview").glob("*/*.parquet"))
    chosen = [f for f in files if f.parent.name in PREFERRED] or files
    chosen = chosen[: args.rows]
    if len(chosen) < args.rows:        # few trajectories done: several frames of each
        chosen = (chosen * args.rows)[: args.rows]
    panels = []
    for r, f in enumerate(chosen):
        t = pq.read_table(f).to_pandas()
        for c in CAMERAS:
            rows = t[t.camera == c].reset_index(drop=True)
            k = (r + 1) * len(rows) // (args.rows + 1) if chosen.count(f) > 1 else len(rows) // 2
            im = Image.open(io.BytesIO(rows.overlay_jpeg[k])).convert("RGB")
            im = im.resize((im.width // SCALE, im.height // SCALE), Image.BILINEAR)
            label = f"{f.parent.name} {c} ({rows.footprint[k]} footprint)"
            ImageDraw.Draw(im).text((8, 6), label, fill=(255, 255, 0))
            panels.append(im)
    w, h = panels[0].size
    sheet = Image.new("RGB", (len(CAMERAS) * w, len(chosen) * h))
    for i, im in enumerate(panels):
        sheet.paste(im, ((i % len(CAMERAS)) * w, (i // len(CAMERAS)) * h))
    sheet.save(args.out, quality=85)
    print(args.out, [str(f) for f in chosen])


if __name__ == "__main__":
    main()
