# apairo-huggingface

Raw **TartanDrive v2** rosbags → **self-supervised traversability labels** →
a **LeRobotDataset** ready for policy training on the Hub.

This repo is the worked example of the
[apairo](https://github.com/apairo-robotics/apairo) stack: it contains no
loader, no sync code and no geometry of its own — the labeling is a ~100-line
composition of
[`apairo_preprocess`](https://github.com/apairo-robotics/apairo_preprocess)
primitives, and the export is ~17 lines of channel→feature naming.

![The driven corridor projected into the camera — real frame, real labels](hub/trav_mask_overlay.jpg)

## Published on the Hub

| Repo | Format | Status |
|---|---|---|
| [`apairo-robotics/tartandrive_offroad`](https://huggingface.co/datasets/apairo-robotics/tartandrive_offroad) | LeRobotDataset | live — 55 episodes / 123,786 frames |
| `apairo-robotics/tartandrive_offroad_eval` | plain `datasets.Dataset` | not yet published — pipeline ready, see [`docs/eval_benchmark.md`](docs/eval_benchmark.md) |

## The pipeline

```
rosbags ──apairo_extractor──▶ .apairo layout (channels, timestamps, calibration)
        ──apairo-tartan-label──▶ trav_traj · lidar_uv · trav_mask   [derived channels]
        ──apairo-tartan-export──▶ LeRobotDataset: image · mask · pose · action
```

### 1. Label — where the robot drove is free ground truth

Per lidar scan, three `apairo_preprocess` preprocessors compose
([`labeling.py`](src/apairo_huggingface/labeling.py)):

| Stage | Primitive | Output channel |
|---|---|---|
| points inside the future robot footprint, look-ahead capped at 40 m of *path* | `TraversabilityFromTrajectory` | `trav_traj` (N,) |
| scan → rectified left camera, row-aligned `[u, v, depth]` | `LidarCameraProjection` | `lidar_uv` (N, 3) — opt-in (`--keep-uv`), recomputable |
| labels splatted nearest-depth-wins + z-buffer, then the **driven corridor** rasterized as occlusion-checked filled quads | `ImageMaskFromPointLabels` + corridor | `trav_mask` (H, W) — grayscale PNG, loader `img` |

The corridor is what makes the mask *dense*: the footprint ribbon interpolated
along the future trajectory fills the whole driven passage instead of a set of
lidar rings. Labels are conservative by construction — `1` is *demonstrated*
traversability (the platform drove there), `0` marks the projected lidar
points (measured, no label information — hence the ring pattern), `255` is
unobserved.

The rig (rectified multisense `P`, vehicle→optical extrinsic) lives in
[`calibration.py`](src/apairo_huggingface/calibration.py) with its provenance;
it reduces to `ds.calibration.get_intrinsics()` / `get_tf()` once the
recordings carry a `cameras:` calibration section. Channels persist via
`ChannelWriter` in bounded memory and register in `.apairo`, so they load like
any raw channel.

### 2. Export — no bespoke domain code

apairo already loads, aligns and cuts the asynchronous multi-rate recordings;
the per-frame value ops are `apairo_transform` callables and the frame is pure
naming:

| LeRobot feature | ← channel | transform |
|---|---|---|
| `observation.images.front` | `image_left_color` (H×W×3 uint8) | as-is (identity) |
| `observation.traversability_mask` | `trav_mask` (H×W uint8) | replicate to RGB — lossless `image`, mp4 would corrupt labels |
| `observation.state` | `gps_odom` | `ChannelSelect(range(7))` → pose, `CastTo(float32)` |
| `action` | `cmd` | `CastTo(float32)` → `[linear_velocity, angular_velocity]` |

The lidar *produces* the labels; it is not part of the published dataset.

## Use

```bash
# 1. Label: write the derived channels on every recording.
#    Needs per-scan poses (gicp_poses; else run apairo_preprocess.KissICPOdometry).
apairo-tartan-label --root /path/to/tartan

# 2. Validate the read pipeline only — no lerobot, no writes.
apairo-tartan-export --dry-run --root /path/to/tartan

# 3. Build the LeRobotDataset locally, or push to the Hub (`huggingface-cli login`).
apairo-tartan-export --no-push --root /path/to/tartan
apairo-tartan-export --repo-id <user>/tartandrive_offroad --root /path/to/tartan
```

Or as a library:

```python
from apairo_huggingface import label_root, load_tartan, to_lerobot

label_root("/path/to/tartan")
dataset = to_lerobot(load_tartan("/path/to/tartan"), "me/tartandrive_offroad", push=True)
```

`--root` accepts a TartanDrive **root** (many recordings → many episodes) or a
**single** sequence directory (→ one episode); it also reads `APAIRO_TARTAN_ROOT`.

## Layout

```
src/apairo_huggingface/
  config.py      channels, sync + labeling params, task, feature dims
  calibration.py the TartanDrive rig: rectified intrinsics + velodyne→camera tf
  labeling.py    trajectory traversability -> trav_traj / lidar_uv / trav_mask
  mappings.py    channel transforms (ChannelSelect / CastTo) + frame naming
  features.py    build_features — the LeRobot schema, sized from a real frame
  export.py      to_lerobot() — the reusable sink; validate() — the dry-run
  eval_labeling.py  ground_truth (toaster) -> trav_mask_gt, projected the same way as trav_mask
  eval_export.py    trav_mask_gt + raw points -> a plain HF dataset (not LeRobot; see docs/)
  cli.py         apairo-tartan-label · apairo-tartan-export · apairo-tartan-gt-label · apairo-tartan-gt-export
examples/
  export_tartan_to_lerobot.py   thin, linear entry point
  load_tartan_views.py          the same raw data as lidar / IMU / windowed / DataLoader views
hub/
  dataset_card.md               the Hub dataset card, ready to upload
docs/
  eval_benchmark.md             sequence selection + publishing design for the hand-labeled benchmark
tests/
  conftest.py       synthetic TartanDrive fixture (geometry matches the real rig)
  test_labeling.py  channels written + registered, label semantics, idempotence
  test_export.py    dry-run + export + reload round-trip
  test_eval.py      ground-truth projection + eval dataset round-trip
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../apairo -e ../apairo_preprocess   # sibling repos, editable
pip install -e ".[test]"                           # this package + lerobot>=0.4.0
```

> Note: `apairo_preprocess` and `apairo_transform` install editable from the `../` siblings for now, pending their own PyPI publishing.

## Getting the data

TartanDrive v2 ships as ROS bags. Two steps bring them into the apairo layout
this pipeline reads.

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
to extract (left colour camera, lidar, GPS odometry, drive command — plus `/tf`
/ `/tf_static` for poses and calibration) and the output path. The result feeds
straight into `apairo-tartan-label` via `--root`.

## Beyond the export: many views of the same raw data

The LeRobot export is one slice of the recordings — the camera clock with mask,
pose and command attached. Changing `keys=` and `reference=` re-slices the same
apairo dataset into other views with no new loader code: lidar-clock perception,
IMU-rate dynamics, causal windows for action chunking, or a direct PyTorch
`DataLoader`. See [`examples/load_tartan_views.py`](examples/load_tartan_views.py)
(runs against apairo's `mini_tartan` fixture).

## Hand-labeled evaluation benchmark

A held-out set of hand-labeled frames (via
[toaster](https://github.com/augustin-bresset/toaster)) to evaluate models
with real traversable/obstacle negatives, not just the self-supervised
`trav_mask`. Sequence selection and the publishing design are in
[`docs/eval_benchmark.md`](docs/eval_benchmark.md); labeling the target
sequences is in progress.

```bash
# 1. Import toaster's raw export as a 'ground_truth' apairo channel (lives in
#    apairo_experiments/scripts/import_toaster_gt.py, one sequence at a time).

# 2. Project the per-point labels into the camera -> trav_mask_gt.
apairo-tartan-gt-label --root /path/to/tartan

# 3. Build the eval benchmark (image + mask + raw points/labels) locally or push it.
apairo-tartan-gt-export --dry-run --root /path/to/tartan
apairo-tartan-gt-export --repo-id <user>/tartandrive_offroad_eval --root /path/to/tartan
```

This is a **plain `datasets.Dataset`**, not a LeRobotDataset: it's a curated
frame subset (not a continuous trajectory) and its point cloud is
ragged (LeRobotDataset features need one fixed shape). See
[`docs/eval_benchmark.md`](docs/eval_benchmark.md) for why.
