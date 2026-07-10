# apairo-huggingface

Label and export [apairo](https://github.com/apairo-robotics/apairo) robotics
datasets to the HuggingFace **LeRobotDataset** format. The first worked bridge:
raw **TartanDrive v2** offroad-driving recordings → self-supervised
traversability labels → a LeRobotDataset ready for policy training on the Hub.

## Why it's small

TartanDrive is an asynchronous, multi-rate sensor rig — camera, GPS/odometry and
drive commands each tick on their own clock, one channel per file. LeRobot wants
the opposite: synchronous, per-episode frames with every modality aligned.

apairo closes that gap on the read side, so **there is no bespoke domain code**
in the export: the per-frame value ops are `apairo_transform` callables
(`ChannelSelect` + `CastTo`) chained onto the synchronized view, and the frame
is pure naming:

| LeRobot feature | ← TartanDrive channel | transform |
|---|---|---|
| `observation.images.front` | `image_left_color` (H×W×3 uint8) | as-is (identity) |
| `observation.traversability_mask` | `trav_mask` (H×W uint8, from `.labeling`) | replicate to RGB (lossless image) |
| `observation.state` | `gps_odom` | `ChannelSelect(range(7))` → pose, `CastTo(float32)` |
| `action` | `cmd` | `CastTo(float32)` → `[linear_velocity, angular_velocity]` |

The `trav_mask` channel does not exist in the raw recordings — `.labeling`
creates it, from `apairo_preprocess` primitives (see below).

## Layout

```
src/apairo_huggingface/
  config.py      channels, sync + labeling params, task, feature dims
  calibration.py the TartanDrive rig: rectified intrinsics + velodyne→camera tf
  labeling.py    trajectory traversability -> trav_traj / lidar_uv / trav_mask
  mappings.py    channel transforms (ChannelSelect / CastTo) + frame naming
  features.py    build_features — the LeRobot schema, sized from a real frame
  export.py      to_lerobot() — the reusable sink; validate() — the dry-run
  cli.py         apairo-tartan-label · apairo-tartan-export
examples/
  export_tartan_to_lerobot.py   thin, linear entry point
  load_tartan_views.py          the same raw data as lidar / IMU / windowed / DataLoader views
hub/
  dataset_card.md               the Hub dataset card, ready to upload
tests/
  conftest.py       synthetic TartanDrive fixture (geometry matches the real rig)
  test_labeling.py  channels written + registered, label semantics, idempotence
  test_export.py    dry-run + export + reload round-trip
```

The interesting file is `export.py`: `to_lerobot` reads top to bottom as
*load → align → copy frames → finalize*, because apairo already did the loading,
resampling and episode segmentation.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../apairo          # apairo core (sibling repo, editable)
pip install -e ".[test]"          # this package + lerobot>=0.4.0
```

## Getting the data

TartanDrive v2 ships as ROS bags. Two steps bring them into the apairo layout
this exporter reads.

**1. Download the bags.** Upstream ships a GUI downloader that pulls one sequence
at a time. This fork adds an interactive terminal client that lets you pick
**modalities × sequences and download them all in one pass**:

```bash
git clone https://github.com/augustin-bresset/tartan_drive_2.0
cd tartan_drive_2.0
pip install -r requirements.txt   # TartanDrive download dependencies
python scripts/tartan_cli.py      # interactive client: choose modalities + sequences, download in bulk
```

Pick the **rosbag** distribution — *not* the pre-converted KITTI one. apairo turns
bags into its native layout directly, keeping `/tf`, static calibration and each
channel's `frame_id` that the KITTI export flattens away.

**2. Extract to apairo format** with
[apairo_extractor](https://github.com/apairo-robotics/apairo_extractor), which
reads ROS bags and writes the `.apairo` sidecars (`channels.yaml`, calibration
from `/tf_static`) — ready for `TartanKittiDataset` / `RawDataset`:

```bash
pip install -e ../apairo_extractor

apairo extractor      # launches the interactive TUI in the terminal
```

The TUI walks you through it: point it at your rosbag directory, pick the topics
to extract (left colour camera, GPS odometry, drive command — plus `/tf` /
`/tf_static` for poses and calibration) and the output path. The result loads
straight into the exporter below via `--root`.

## Use

```bash
# 1. Label: write trav_traj / lidar_uv / trav_mask on every recording.
#    Needs per-scan poses (gicp_poses; else run apairo_preprocess.KissICPOdometry).
apairo-tartan-label --root /path/to/tartan

# 2. Validate the apairo read pipeline only — no lerobot, no writes.
apairo-tartan-export --dry-run --root /path/to/tartan

# 3. Build a LeRobotDataset locally, or push to the Hub (needs `huggingface-cli login`).
apairo-tartan-export --no-push --root /path/to/tartan
apairo-tartan-export --repo-id <user>/tartandrive_offroad --root /path/to/tartan
```

The labels are self-supervised — where the robot actually drove: the future
trajectory labels the lidar points inside the robot footprint
(`TraversabilityFromTrajectory`, path-length capped), the scan projects into
the left camera (`LidarCameraProjection`, rig in `calibration.py`), and the
mask is the splatted point labels plus the **driven corridor** rasterized as
filled quads — dense coverage, occlusion-aware. All primitives are
[`apairo_preprocess`](https://github.com/apairo-robotics/apairo_preprocess);
`labeling.py` is the ~100-line composition.

Or as a library:

```python
from apairo_huggingface import load_tartan, to_lerobot

dataset = to_lerobot(load_tartan("/path/to/tartan"), "me/tartandrive_offroad", push=True)
```

`--root` accepts a TartanDrive **root** (many recordings → many episodes) or a
**single** sequence directory (→ one episode); it also reads `APAIRO_TARTAN_ROOT`.

## Beyond the export: many views of the same raw data

The LeRobot export is one slice of the recordings — the camera clock with pose
and command attached. Changing `keys=` and `reference=` re-slices the same
apairo dataset into other views with no new loader code: lidar-clock perception,
IMU-rate dynamics, causal windows for action chunking, or a direct PyTorch
`DataLoader`. See [`examples/load_tartan_views.py`](examples/load_tartan_views.py)
(runs against apairo's `mini_tartan` fixture).

## Publish the dataset card

`push_to_hub` generates a minimal README on the dataset repo. Replace it with
the full card in [`hub/dataset_card.md`](hub/dataset_card.md) — features table,
the apairo story, reproduction command and TartanDrive citation:

```bash
hf upload <user>/tartandrive_offroad hub/dataset_card.md README.md --repo-type dataset
```

## Test

```bash
pytest        # builds a synthetic TartanDrive dataset, exports it, reloads it
```

## Notes

- **Data**: see [Getting the data](#getting-the-data) for the rosbag download +
  apairo_extractor workflow. The apairo fixture `apairo/test/assets/mini_tartan`
  ships only `velodyne_0`/`cmd`/`imu`, so the full camera+state+action mapping
  needs a complete sequence — hence the synthetic fixture in `tests/`.
- **Verified** against `lerobot 0.5.1` (LeRobotDataset v3.0) and `apairo 0.4.0`:
  `create → add_frame({..., "task"}) → save_episode → finalize → push_to_hub`.
  In v3, `task` is a key of the frame dict and `finalize()` is required before the
  dataset is valid.
