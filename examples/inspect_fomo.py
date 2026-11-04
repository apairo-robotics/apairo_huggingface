"""Browse one episode of the FoMo LeRobot export in Rerun.

Logs, per frame: the camera image, the traversability mask as a segmentation
layer on top of it (1 driven / 0 seen, not driven / 255 no data), the pose on
the full trajectory, the action and the environment scalars.

Start a viewer first, e.g. on the machine holding the data:

    rerun --serve-web --web-viewer-port 9090 --port 9876

then:

    python examples/inspect_fomo.py --root /path/to/fomo_lerobot_v2 --episode 0 --stride 5
"""

import argparse
import io
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import rerun as rr
from PIL import Image

VIDEO_KEY = "observation.images.front"
MASK_KEY = "observation.traversability_mask"
# Segmentation class ids: the stored values 1 / 0 / 255 mapped to 1 / 2 / 0.
CLASS_OF_VALUE = {1: 1, 0: 2, 255: 0}


def episode_meta(root: Path, episode: int) -> dict:
    tables = [pq.read_table(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))]
    for t in tables:
        for row in t.to_pylist():
            if row["episode_index"] == episode:
                return row
    raise SystemExit(f"episode {episode} not found")


def episode_rows(root: Path, info: dict, meta: dict) -> list[dict]:
    path = root / info["data_path"].format(
        chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"])
    rows = pq.read_table(path).to_pylist()
    return [r for r in rows if r["episode_index"] == meta["episode_index"]]


def video_frames(root: Path, info: dict, meta: dict):
    """Yield (frame_index, rgb) for the episode's span of its video file."""
    prefix = f"videos/{VIDEO_KEY}/"
    path = root / info["video_path"].format(
        video_key=VIDEO_KEY,
        chunk_index=meta[prefix + "chunk_index"], file_index=meta[prefix + "file_index"])
    t0, t1 = meta[prefix + "from_timestamp"], meta[prefix + "to_timestamp"]
    fps = info["fps"]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(t0 / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t0 - 0.5 / fps:
                continue
            if t >= t1 - 0.5 / fps:
                break
            yield int(round((t - t0) * fps)), frame.to_ndarray(format="rgb24")


def mask_classes(png: bytes) -> np.ndarray:
    values = np.asarray(Image.open(io.BytesIO(png)))[..., 0]
    out = np.zeros(values.shape, np.uint8)
    for value, cls in CLASS_OF_VALUE.items():
        out[values == value] = cls
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5, help="log every n-th frame")
    ap.add_argument("--url", default="rerun+http://localhost:9876/proxy")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    args = ap.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    meta = episode_meta(args.root, args.episode)
    rows = episode_rows(args.root, info, meta)
    if len(rows) != meta["length"]:
        raise SystemExit(f"episode {args.episode}: {len(rows)} rows, meta says {meta['length']}")

    rr.init(f"fomo_episode_{args.episode}")
    rr.connect_grpc(args.url)

    rr.log("camera/mask", rr.AnnotationContext([
        (0, "no data", (0, 0, 0, 0)),
        (1, "driven", (0, 200, 0, 255)),
        (2, "seen, not driven", (220, 30, 30, 255)),
    ]), static=True)
    xyz = np.array([r["observation.state"][:3] for r in rows], np.float32)
    rr.log("world/trajectory", rr.LineStrips3D([xyz], colors=[(120, 120, 255)]), static=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    by_index = {r["frame_index"]: r for r in rows}
    for idx, rgb in video_frames(args.root, info, meta):
        if idx % args.stride or idx not in by_index:
            continue
        r = by_index[idx]
        rr.set_time("frame", sequence=idx)
        rr.set_time("time", duration=float(r["timestamp"]))
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=args.jpeg_quality)
        rr.log("camera/image", rr.EncodedImage(contents=buf.getvalue(), media_type="image/jpeg"))
        rr.log("camera/mask", rr.SegmentationImage(mask_classes(r[MASK_KEY]["bytes"])))
        x, y, z, qx, qy, qz, qw = r["observation.state"]
        rr.log("world/robot", rr.Transform3D(
            translation=[x, y, z], rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])))
        rr.log("world/robot/axes", rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]))
        v, w = r["action"]
        rr.log("action/linear_velocity", rr.Scalars(v))
        rr.log("action/angular_velocity", rr.Scalars(w))
        snow, temp = r["observation.environment"]
        rr.log("environment/snow_depth_m", rr.Scalars(snow))
        rr.log("environment/air_temperature_c", rr.Scalars(temp))
    print(f"episode {args.episode}: logged every {args.stride}th of {len(rows)} frames")


if __name__ == "__main__":
    main()
