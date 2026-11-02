"""Stream the Isaac Sim mcap into one apairo sequence directory.

The bag is read once, in recording order, with ``rosbags`` (no message is held
longer than it takes to write it). Every channel is stamped with its HEADER
stamp, i.e. simulation time; ``/cmd_vel`` has no header and takes the latest
``/clock``. Layout of ``out_dir``::

    lidar/            000000.npy ...   (N, 3) float32 x y z, frame 'sensor'
    rgb/              000000.png ...   (720, 1280, 3) uint8
    semantic/         000000.png ...   (720, 1280) uint8 training class id
    depth/            000000.png ...   (720, 1280) uint16 mm, 0 = no return
    points_semantic/  000000.npy ...   (N, 4) float32 x y z class id
    pose/ odometry/ imu/ cmd_vel/      one stacked .npy each
    semantic_labels.jsonl              the raw per-frame runtime id -> name maps
    .apairo/channels.yaml, .apairo/calibration.yaml (fixed /tf edges + intrinsics)
    extract_report.json

Semantic ids: ``/cam2/segmentation`` holds RUNTIME Isaac label ids, which are
only meaningful together with the ``/cam2/semantic_labels`` message of the same
stamp. Each image is remapped through that message and ``name_to_id`` of
``semantic_classes.json``; names outside ``name_to_id`` (``BACKGROUND``,
``UNLABELLED``) become 0 = unlabeled and are counted in the report.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from apairo.core.config import register_intrinsics, register_raw_channel, register_static_transform

from . import config

logger = logging.getLogger(__name__)

FRAME_CHANNELS = {
    config.TOPIC_LIDAR: config.LIDAR,
    config.TOPIC_RGB: config.RGB,
    config.TOPIC_SEG: config.SEMANTIC,
    config.TOPIC_DEPTH: config.DEPTH,
    config.TOPIC_POINTS_SEMANTIC: config.POINTS_SEMANTIC,
}
TOPICS = [*FRAME_CHANNELS, config.TOPIC_SEG_LABELS, config.TOPIC_CAMERA_INFO, config.TOPIC_ODOMETRY,
          config.TOPIC_IMU, config.TOPIC_CMD_VEL, config.TOPIC_CLOCK, config.TOPIC_TF]


def stamp_of(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def pose7_to_matrix(pose) -> np.ndarray:
    """``[tx, ty, tz, qx, qy, qz, qw]`` -> 4x4."""
    tx, ty, tz, x, y, z, w = (float(v) for v in pose)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w), tx],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w), ty],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y), tz],
        [0.0, 0.0, 0.0, 1.0],
    ])


def runtime_lut(labels: dict, unknown: dict) -> np.ndarray:
    """Runtime label id -> training class id, from one semantic_labels message."""
    lut = np.zeros(max([int(k) for k in labels if k.isdigit()] + [0]) + 1, dtype=np.uint8)
    for key, value in labels.items():
        if not key.isdigit():
            continue                                    # "time_stamp"
        name = value.get("class", "")
        cid = config.NAME_TO_ID.get(name)
        if cid is None:
            unknown[name] = unknown.get(name, 0) + 1
            cid = 0
        lut[int(key)] = cid
    return lut


def decode_points_semantic(msg) -> np.ndarray:
    """x y z + packed float rgb -> (N, 4) float32 x y z class id (255: no camera)."""
    raw = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
    rgb = raw[:, 3].copy().view(np.uint32)
    key = ((rgb >> 16) & 255) * 65536 + ((rgb >> 8) & 255) * 256 + (rgb & 255)
    cls = np.full(len(raw), config.NO_DATA, dtype=np.float32)
    for cid, (_, color, _) in config.CLASSES.items():
        cls[key == color[0] * 65536 + color[1] * 256 + color[2]] = cid
    return np.column_stack([raw[:, :3], cls]).astype(np.float32)


def check_classes_json(path) -> None:
    data = json.loads(Path(path).read_text())
    if data["name_to_id"] != config.NAME_TO_ID:
        raise ValueError(f"{path}: name_to_id {data['name_to_id']} != config {config.NAME_TO_ID}")
    for c in data["classes"]:
        name, color, trav = config.CLASSES[c["id"]]
        if (c["name"], tuple(c["color"]), c["traversable"]) != (name, color, trav):
            raise ValueError(f"{path}: class {c} differs from config.CLASSES")


def extract(bag, out_dir, *, t_start: float | None = None, t_stop: float | None = None,
            classes_json=config.CLASSES_JSON, workers: int = 8) -> dict:
    """Extract ``[t_start, t_stop)`` (sim seconds, header clock) of ``bag`` into ``out_dir``."""
    from rosbags.highlevel import AnyReader

    t0 = time.time()
    check_classes_json(classes_json)
    out = Path(out_dir)
    if (out / ".apairo" / "channels.yaml").exists():
        raise FileExistsError(f"{out} already holds an extraction")
    for name in FRAME_CHANNELS.values():
        (out / name).mkdir(parents=True, exist_ok=True)

    stamps = {name: [] for name in FRAME_CHANNELS.values()}
    series = {config.POSE: [], config.ODOMETRY: [], config.IMU: [], config.CMD_VEL: []}
    static = {}                 # (parent, child) -> [first matrix, max deviation]
    moving_edges = set()
    labels_by_stamp, pending_seg = {}, {}
    unknown_names, label_sets = {}, set()
    camera_info = None
    camera_info_changes = 0
    frame_ids = {}
    clock = None
    in_window = lambda t: (t_start is None or t >= t_start) and (t_stop is None or t < t_stop)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = []
    labels_file = open(out / "semantic_labels.jsonl", "w")

    def save_png(array, path):
        Image.fromarray(array).save(path, compress_level=6)

    def write_seg(stamp, seg):
        lut = runtime_lut(labels_by_stamp.pop(stamp), unknown_names)
        if seg.max() >= len(lut):
            raise ValueError(f"segmentation at {stamp:.6f} has id {seg.max()} not in its label map")
        idx = len(stamps[config.SEMANTIC])
        futures.append(pool.submit(save_png, lut[seg], out / config.SEMANTIC / f"{idx:06d}.png"))
        stamps[config.SEMANTIC].append(stamp)

    with AnyReader([Path(bag)]) as reader:
        conns = [c for c in reader.connections if c.topic in TOPICS]
        for conn, _, raw in reader.messages(connections=conns):
            topic = conn.topic
            msg = reader.deserialize(raw, conn.msgtype)
            if topic == config.TOPIC_CLOCK:
                clock = msg.clock.sec + msg.clock.nanosec * 1e-9
                if t_stop is not None and clock > t_stop + 1.0:
                    break                       # the window is over (1 s slack for late topics)
                continue
            if topic == config.TOPIC_TF:
                for tr in msg.transforms:
                    edge = (tr.header.frame_id, tr.child_frame_id)
                    p = tr.transform.translation
                    q = tr.transform.rotation
                    pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                    if edge == (config.ODOM_FRAME, config.BASE_FRAME):
                        t = stamp_of(tr)
                        if in_window(t):
                            series[config.POSE].append((t, pose))
                    elif edge in config.STATIC_EDGES:
                        M = pose7_to_matrix(pose)
                        if edge not in static:
                            static[edge] = [M, 0.0]
                        static[edge][1] = max(static[edge][1], float(np.abs(M - static[edge][0]).max()))
                    else:
                        moving_edges.add(edge)
                continue
            if topic == config.TOPIC_CMD_VEL:
                if clock is not None and in_window(clock):
                    series[config.CMD_VEL].append((clock, [msg.linear.x, msg.linear.y, msg.linear.z,
                                                           msg.angular.x, msg.angular.y, msg.angular.z]))
                continue
            if topic == config.TOPIC_SEG_LABELS:
                labels = json.loads(msg.data)
                ts = labels["time_stamp"]
                t = ts["sec"] + ts["nanosec"] * 1e-9
                if not in_window(t):
                    continue
                labels_file.write(msg.data + "\n")
                label_sets.add(tuple(sorted((k, v["class"]) for k, v in labels.items() if k.isdigit())))
                labels_by_stamp[t] = labels
                if t in pending_seg:
                    write_seg(t, pending_seg.pop(t))
                continue

            t = stamp_of(msg)
            if not in_window(t):
                continue
            frame_ids.setdefault(topic, msg.header.frame_id)
            if msg.header.frame_id != frame_ids[topic]:
                raise ValueError(f"{topic}: frame_id changed {frame_ids[topic]} -> {msg.header.frame_id}")

            if topic == config.TOPIC_CAMERA_INFO:
                info = (msg.height, msg.width, msg.distortion_model, tuple(msg.d), tuple(msg.k), tuple(msg.p))
                if camera_info is None:
                    camera_info = info
                elif info != camera_info:
                    camera_info_changes += 1
            elif topic == config.TOPIC_ODOMETRY:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                v, w = msg.twist.twist.linear, msg.twist.twist.angular
                series[config.ODOMETRY].append((t, [p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z]))
            elif topic == config.TOPIC_IMU:
                a, g, q = msg.linear_acceleration, msg.angular_velocity, msg.orientation
                series[config.IMU].append((t, [a.x, a.y, a.z, g.x, g.y, g.z, q.x, q.y, q.z, q.w]))
            elif topic == config.TOPIC_LIDAR:
                if [f.name for f in msg.fields][:3] != ["x", "y", "z"]:
                    raise ValueError(f"{topic}: unexpected fields {[f.name for f in msg.fields]}")
                pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)[:, :3]
                idx = len(stamps[config.LIDAR])
                np.save(out / config.LIDAR / f"{idx:06d}.npy", np.ascontiguousarray(pts))
                stamps[config.LIDAR].append(t)
            elif topic == config.TOPIC_POINTS_SEMANTIC:
                idx = len(stamps[config.POINTS_SEMANTIC])
                np.save(out / config.POINTS_SEMANTIC / f"{idx:06d}.npy", decode_points_semantic(msg))
                stamps[config.POINTS_SEMANTIC].append(t)
            elif topic == config.TOPIC_RGB:
                if msg.encoding != "rgb8":
                    raise ValueError(f"{topic}: encoding {msg.encoding}")
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
                idx = len(stamps[config.RGB])
                futures.append(pool.submit(save_png, img, out / config.RGB / f"{idx:06d}.png"))
                stamps[config.RGB].append(t)
            elif topic == config.TOPIC_DEPTH:
                d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
                mm = np.where(np.isfinite(d), np.round(d * 1000.0), 0).clip(0, 65535).astype(np.uint16)
                idx = len(stamps[config.DEPTH])
                futures.append(pool.submit(save_png, mm, out / config.DEPTH / f"{idx:06d}.png"))
                stamps[config.DEPTH].append(t)
            elif topic == config.TOPIC_SEG:
                seg = np.frombuffer(msg.data, dtype=np.int32).reshape(msg.height, msg.width).copy()
                if t in labels_by_stamp:
                    write_seg(t, seg)
                else:
                    pending_seg[t] = seg
            if len(futures) > 64:               # bound the queue of images in RAM
                for f in futures:
                    f.result()
                futures = []
            n = len(stamps[config.LIDAR])
            if topic == config.TOPIC_LIDAR and n % 500 == 0:
                logger.info("t=%.2f s sim  %d scans  %.0f s", t, n, time.time() - t0)

    for f in futures:
        f.result()
    pool.shutdown()
    labels_file.close()
    if pending_seg:
        logger.warning("%d segmentation images had no label map and were dropped", len(pending_seg))

    # -- sidecars: timestamps, channel registry, calibration
    loaders = {config.LIDAR: "npys", config.POINTS_SEMANTIC: "npys",
               config.RGB: "img", config.SEMANTIC: "img", config.DEPTH: "img"}
    frames = {config.LIDAR: config.LIDAR_FRAME, config.POINTS_SEMANTIC: config.LIDAR_FRAME,
              config.RGB: config.CAMERA_FRAME, config.SEMANTIC: config.CAMERA_FRAME,
              config.DEPTH: config.CAMERA_FRAME}
    for name, ts in stamps.items():
        np.savetxt(out / name / "timestamps.txt", np.asarray(ts), fmt="%.9f")
        register_raw_channel(out, name, loaders[name], frame=frames[name])
    for name, rows in series.items():
        if not rows:
            continue
        (out / name).mkdir(exist_ok=True)
        rows = sorted(rows, key=lambda r: r[0])
        ts = np.array([r[0] for r in rows])
        keep = np.concatenate([[True], np.diff(ts) > 0]) if len(ts) else np.zeros(0, bool)
        np.save(out / name / f"{name}.npy", np.array([r[1] for r in rows], dtype=np.float64)[keep])
        np.savetxt(out / name / "timestamps.txt", ts[keep], fmt="%.9f")
        transform = ({"parent": config.ODOM_FRAME, "child": config.BASE_FRAME, "source": config.TOPIC_TF,
                      "format": "t_xyz_q_xyzw"} if name == config.POSE else None)
        frame = config.BASE_FRAME if name == config.ODOMETRY else None
        register_raw_channel(out, name, "npy", transform=transform, frame=frame)
    for (parent, child), (M, dev) in static.items():
        if dev > config.STATIC_TOLERANCE:
            raise ValueError(f"/tf {parent}->{child} moves by {dev:.4f}: not a fixed mount")
        register_static_transform(out, parent, child, M)
    if camera_info is None:
        raise ValueError("no camera_info in the window")
    h, w, model, d, k, p = camera_info
    register_intrinsics(out, config.CAMERA_FRAME, K=np.reshape(k, (3, 3)), distortion=np.asarray(d),
                        model=model, width=w, height=h, P=np.reshape(p, (3, 4)))

    report = {
        "bag": str(bag),
        "window_sim_s": [t_start, t_stop],
        "frames": {k: len(v) for k, v in stamps.items()},
        "series": {k: len(v) for k, v in series.items()},
        "sim_time_range": [float(min(stamps[config.LIDAR])), float(max(stamps[config.LIDAR]))] if stamps[config.LIDAR] else None,
        "frame_ids": frame_ids,
        "static_edges_max_deviation": {f"{a}->{b}": v[1] for (a, b), v in static.items()},
        "moving_edges_not_extracted": sorted(f"{a}->{b}" for a, b in moving_edges),
        "camera_info_changes": camera_info_changes,
        "segmentation_names_not_in_name_to_id": unknown_names,
        "distinct_label_maps": [dict(s) for s in sorted(label_sets)],
        "segmentation_dropped_no_label_map": len(pending_seg),
        "seconds": round(time.time() - t0),
    }
    (out / "extract_report.json").write_text(json.dumps(report, indent=2))
    logger.info("%s", json.dumps(report["frames"]))
    return report
