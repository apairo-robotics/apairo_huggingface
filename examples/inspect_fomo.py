"""Browse the FoMo traversability labels of one trajectory in Rerun.

The labels dataset holds no images or scans: this script joins every label
channel with FoMo's raw files, read from a local copy (``--raw``) when present
and otherwise fetched from the public FoMo bucket, frame by frame.

Logged on one timeline (the FoMo microsecond stamps):

- ``<camera>/image`` + ``<camera>/mask`` for each camera: the raw image and its
  mask as a segmentation layer (1 driven / 0 seen, not driven / 255 no data);
  the ZED cameras carry the future footprint, the rear Basler the past one;
- ``world/trajectory`` and ``world/robot``: the KISS-ICP poses of the scans;
- ``world/robot/lidar`` (with ``--lidar``): the RoboSense scan coloured by its
  future / past driven labels.

Start a viewer first, e.g. on the machine holding the labels:

    rerun --serve-web --web-viewer-port 9090 --port 9876

then:

    python examples/inspect_fomo.py --labels /path/to/fomo_labels \\
        --trajectory 2024-11-21/red_2024-11-21-10-34 --stride 20 --lidar
"""

import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
from PIL import Image

from apairo_huggingface.fomo.fetch import BUCKET, get
from apairo_huggingface.fomo.format import RECORD

CAMERAS = ("zedx_left", "zedx_right", "basler")
# Segmentation class ids: the stored values 1 / 0 / 255 mapped to 1 / 2 / 0.
CLASS_OF_VALUE = {1: 1, 0: 2, 255: 0}
SCALE = 2  # images and masks are logged at half resolution


def raw_file(trajectory: str, sensor: str, name: str, raw: Path | None) -> bytes:
    if raw is not None:
        local = raw / trajectory / sensor / name
        if local.exists():
            return local.read_bytes()
    return get(f"{BUCKET}/data/{trajectory}/{sensor}/{name}")


def mask_classes(png: bytes) -> np.ndarray:
    values = np.asarray(Image.open(io.BytesIO(png)))
    if values.ndim == 3:
        values = values[..., 0]
    values = values[::SCALE, ::SCALE]
    out = np.zeros(values.shape, np.uint8)
    for value, cls in CLASS_OF_VALUE.items():
        out[values == value] = cls
    return out


def jpeg(png: bytes, quality: int) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img = img.resize((img.width // SCALE, img.height // SCALE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="root of the labels dataset")
    ap.add_argument("--trajectory", required=True, help="<deployment>/<trajectory>")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=CAMERAS)
    ap.add_argument("--stride", type=int, default=20, help="log every n-th row of each channel")
    ap.add_argument("--lidar", action="store_true", help="also log the labelled RoboSense scans")
    ap.add_argument("--raw", type=Path, default=None, help="local FoMo root, tried before the bucket")
    ap.add_argument("--url", default="rerun+http://localhost:9876/proxy")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    args = ap.parse_args()

    deployment, name = args.trajectory.split("/")
    rr.init(f"fomo_labels_{name}")
    rr.connect_grpc(args.url)

    def at(stamp_us: int) -> None:
        rr.set_time("stamp", timestamp=np.datetime64(int(stamp_us), "us"))

    # Raw files come from the bucket at ~2 s each: fetch ahead in threads.
    pool = ThreadPoolExecutor(8)
    for camera in args.cameras:
        rows = pq.read_table(args.labels / camera / deployment / f"{name}.parquet").to_pylist()
        footprint = rows[0]["footprint"] if rows else "?"
        rr.log(f"{camera}/mask", rr.AnnotationContext([
            (0, "no data", (0, 0, 0, 0)),
            (1, f"driven ({footprint} footprint)", (0, 200, 0, 255)),
            (2, "seen, not driven", (220, 30, 30, 255)),
        ]), static=True)
        picked = rows[::args.stride]
        pngs = pool.map(lambda r: raw_file(args.trajectory, camera, r["image_file"], args.raw), picked)
        for r, png in zip(picked, pngs):
            at(r["timestamp_us"])
            rr.log(f"{camera}/image", rr.EncodedImage(
                contents=jpeg(png, args.jpeg_quality), media_type="image/jpeg"))
            rr.log(f"{camera}/mask", rr.SegmentationImage(mask_classes(r["mask_png"])))
        print(f"{camera}: logged every {args.stride}th of {len(rows)} masks ({footprint} footprint)")

    lidar = pq.read_table(args.labels / "robosense" / deployment / f"{name}.parquet").to_pylist()
    poses = np.array([r["pose"] for r in lidar], np.float64).reshape(-1, 4, 4)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/trajectory", rr.LineStrips3D([poses[:, :3, 3]], colors=[(120, 120, 255)]), static=True)
    picked = list(range(0, len(lidar), args.stride))
    scans = pool.map(lambda k: raw_file(args.trajectory, "robosense", lidar[k]["robosense_file"], args.raw),
                     picked) if args.lidar else iter(lambda: None, 0)
    for k, scan in zip(picked, scans):
        r = lidar[k]
        at(r["timestamp_us"])
        rr.log("world/robot", rr.Transform3D(translation=poses[k, :3, 3], mat3x3=poses[k, :3, :3]))
        rr.log("world/robot/axes", rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]))
        if args.lidar:
            rec = np.frombuffer(scan, dtype=RECORD)
            if len(rec) != r["n_points"]:
                raise SystemExit(f"{r['robosense_file']}: {len(rec)} points, labels say {r['n_points']}")
            xyz = np.column_stack([rec["x"], rec["y"], rec["z"]])
            colors = np.full((len(rec), 3), 150, np.uint8)
            colors[np.asarray(r["driven_indices_past"], np.int64)] = (255, 160, 0)
            colors[np.asarray(r["driven_indices"], np.int64)] = (0, 200, 0)
            rr.log("world/robot/lidar", rr.Points3D(xyz, colors=colors, radii=0.03))
    print(f"robosense: logged every {args.stride}th of {len(lidar)} scans")


if __name__ == "__main__":
    main()
