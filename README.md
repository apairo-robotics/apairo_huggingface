# TartanDrive → LeRobotDataset, with apairo

An example that turns raw **TartanDrive v2** offroad-driving recordings into a
**LeRobotDataset** ready for policy training on the Hugging Face Hub — using
[apairo](https://github.com/apairo-robotics/apairo) for the read side.

## Why this is short

TartanDrive is an asynchronous, multi-rate sensor rig: camera, GPS/odometry and
drive commands each tick on their own clock, one channel per file. LeRobot wants
the opposite: synchronous, per-episode frames with every modality aligned.

apairo closes that gap in three calls:

| step | apairo |
|---|---|
| load the raw channels | `TartanKittiDataset(root, keys=[...])` |
| one episode per recording | `.sequences` |
| align sensors onto one clock | `.synchronize(reference="image_left_color", ...)` |

Each frame then comes back as a plain dict of numpy arrays. **The only code left
to write is the channel → feature mapping** — three small functions in
[`export_tartan_to_lerobot.py`](export_tartan_to_lerobot.py):

| LeRobot feature | ← TartanDrive channel | mapping |
|---|---|---|
| `observation.images.front` | `image_left_color` (H×W×3 uint8) | as-is |
| `observation.state` | `gps_odom` | pose `[x, y, z, qx, qy, qz, qw]` |
| `action` | `cmd` | twist `[linear_velocity, angular_velocity]` |

The messy part is a library call; the domain part is three functions. That
contrast is the point.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install "lerobot>=0.4.0"      # LeRobotDataset v3.0
pip install -e ../apairo          # apairo (sibling repo, editable)
```

## Run

```bash
# 1. Validate the apairo read pipeline only — no lerobot, no writes.
python export_tartan_to_lerobot.py --dry-run --root /path/to/tartan

# 2. Build a LeRobotDataset locally (encodes video, writes parquet).
python export_tartan_to_lerobot.py --no-push --root /path/to/tartan

# 3. Build and push to the Hub (needs `huggingface-cli login`).
python export_tartan_to_lerobot.py --repo-id <user>/tartandrive_offroad --root /path/to/tartan
```

`--root` accepts a TartanDrive **root** (many recordings → many episodes) or a
**single** sequence directory (→ one episode). It also reads `APAIRO_TARTAN_ROOT`.

## Notes

- **Data**: TartanDrive v2 is assumed already on disk; there is no downloader.
  The apairo test fixture `apairo/test/assets/mini_tartan` only ships
  `velodyne_0`/`cmd`/`multisense_imu` (no camera or odometry), so the full
  camera+state+action mapping needs a complete sequence.
- **Verified** against `lerobot 0.5.1` (LeRobotDataset v3.0) and `apairo 0.4.0`:
  `create → add_frame({..., "task"}) → save_episode → finalize → push_to_hub`.
  In v3, `task` is a key of the frame dict and `finalize()` is required before
  the dataset is valid.
