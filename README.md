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
rosbags ──apairo_ros──▶ .apairo layout (channels, timestamps, calibration)
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

## FoMo: seasonal traversability (in progress)

[FoMo](https://fomo.norlab.ulaval.ca) (Forêt Montmorency, CC BY 4.0) recorded the
same forest trails twelve times between November 2024 and November 2025, snow
included, and serves them from a public S3 bucket. `apairo_huggingface.fomo` derives
self-supervised traversability labels from it, one label channel per sensor,
published as labels only (keyed by the raw FoMo file names):

- `fomo_bin`, an apairo **format plugin** for FoMo's packed 26-byte lidar records;
- a declaration (`fomo/apairo.yaml`) reading the tree as published -- RoboSense
  scans, camera images and the ~50 Hz twist commands, each on its own clock;
- KISS-ICP odometry, scored against FoMo's GNSS-PPK `gt.txt` (ATE);
- per-point RoboSense labels of the future and the past driven footprint;
- one mask per frame of each camera, computed **at the camera instant** from
  the scans and the camera stamps of the bucket listing (no image needed):
  future footprint for the forward ZED X left/right, past footprint for the
  rear-facing Basler;
- `apairo-fomo verify`, which re-checks the published tables against FoMo's
  raw files (stamps, point counts, camera geometry, lidar/mask agreement).

`fomo/calibration.py` documents the rig as checked by projecting scans onto
images: the robot drives along the RoboSense -x axis, the ZED X looks forward
and the Basler looks backward, the extrinsics of `calib/transforms.json` are
read literally, and the camera intrinsics shipped in the bucket are
placeholders replaced by the factory calibrations of the FoMo SDK.

```bash
apairo-fomo list --color red                       # 12 trajectories, 8-21 GB of scans each
apairo-fomo run --root ~/data/fomo --color red --drop-lidar
apairo-fomo export-labels ~/data/fomo --out ~/data/fomo_labels       # robosense/
apairo-fomo export-masks ~/data/fomo --out ~/data/fomo_labels        # zedx_left/, zedx_right/, basler/
apairo-fomo verify 2024-11-21/red_2024-11-21-10-34 --labels ~/data/fomo_labels
```

`run` downloads (scans, calibration, metadata), then processes, one
trajectory at a time and skips those with an `apairo_fomo_report.json`, so it
resumes after an interruption. `--drop-lidar` deletes the scans once
labelled. `scripts/fomo_labels_pipeline.py` rebuilds the labels of
already-processed trajectories one at a time (scans downloaded, used,
deleted). The older `apairo-fomo export` (LeRobot images + masks) remains
for reference; its Basler-based product is withdrawn
(`hub/fomo_dataset_card.md`).

## RELLIS-3D: lidar labels and traversability in every image (in progress)

[RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) (CC BY-NC-SA 3.0)
annotates 13556 Ouster scans point by point, but only 6235 camera images.
Every scan is co-captured with an image, so `apairo_huggingface.rellis`
projects the scan labels into **all** 13556 images, adds a traversability mask
from the driven trajectory, and measures how well the projected labels agree
with the official 2D annotations:

- `semkitti_label`, an apairo **format plugin** for SemanticKITTI `.label` files;
- a declaration (`rellis/apairo.yaml`) reading a sequence as published; scans
  and labels borrow the camera clock (the filenames carry the stamp). The source
  tree is read through a symlink view, so nothing is written into it;
- `sem_mask` (ontology id, 255 = no lidar point) and `trav_mask`
  (1 driven / 0 seen, not driven / 255 no data) for every image;
- `apairo-rellis agreement`: per-class IoU against the official 2D labels,
  only where the projection has data;
- the export: image (video), both masks and the official 2D label (lossless
  PNG), pose, executed motion as `action`, official lidar split per frame.

`rellis/calibration.py` documents the calibration chain and its traps (an
extrinsic stored as the camera pose, to be inverted; rectified images, so no
distortion), each checked by scoring projections against the 2D annotations.

```bash
apairo-rellis process   --root /data/rellis --out ~/data/rellis_derived    # all 5 sequences
apairo-rellis coverage  --out ~/data/rellis_derived                        # coverage table, from the masks
apairo-rellis agreement --root /data/rellis --out ~/data/rellis_derived
apairo-rellis overlay ~/data/rellis_derived/00003 --every 500 --png-dir /tmp/overlays
apairo-rellis export ~/data/rellis_derived 00003 --root /data/rellis --lerobot-root ~/data/rellis_lerobot
```

The derived dataset inherits RELLIS-3D's **non-commercial, share-alike**
license.

## AMIAD Isaac Sim: perfect semantics, measured PU noise (in progress)

The lab's Isaac Sim recording (one ROS 2 mcap, Barakuda in a generated
forest) has a perfect semantic camera. `apairo_huggingface.sim_amiad` turns it
into an apairo sequence, transfers the sim semantics to the lidar, adds the
usual trajectory traversability, and measures how noisy that self-supervised
PU label is against the semantic truth -- the sim-to-real companion of the
FoMo / RELLIS / TartanDrive derived datasets. Local only: the data is the
lab's, nothing is pushed to the Hub.

- `extract`: a streaming mcap reader (rosbags) writing lidar, RGB, semantic
  (runtime Isaac ids remapped through each frame's `/cam2/semantic_labels` and
  `name_to_id` of `semantic_classes.json`), depth, the sim's own semantic
  cloud, poses, odometry, IMU and `/cmd_vel`, all on the header (sim) clock;
  fixed `/tf` edges and the camera_info go to `calibration.yaml`;
- `label`: `point_sem` (class of each lidar point, only if its depth agrees
  with the sim depth image), `trav_traj` (future footprint) and `trav_mask`;
- `agreement`: positive precision / recall, the share of the unlabeled set
  that is truly traversable, per range band, plus a point-by-point check of
  the label transfer against the sim's semantic cloud;
- the export: image (video), semantic and traversability masks (PNG), pose,
  `/cmd_vel` as `action`, every 4th scan; the fps is measured from the scan
  stamps (15 on this bag, whose scans are exactly 1/60 s apart) and an
  irregular sequence is refused. Images are matched to a scan within the same
  5 ms tolerance (`config.SYNC_TOLERANCE`) as the labelling.

Label transfer vs the sim's own semantic cloud (2247 clouds of the full bag,
every cloud point matched to a lidar point within 1 cm): **97.78 %** agreement
on the 5,353,101 points both sides give a real class. Counting `unlabeled`
(0) as a class gives 97.20 % on 5,385,122 points: the 32,021 extra points are
all points we label (30,488 tree, 1,533 vegetation) that the sim cloud calls
unlabeled; we never label a matched point 0. The 30 s segment
`seg_t035_t065`: 97.71 % (97.20 % with 0). `agreement.json` holds both.

`extract` needs the optional `[sim]` extra (`pip install -e ".[sim]"`, for
`rosbags`); the other commands do not. `sim_amiad/config.py` documents the frame trap: the camera headers say `ZED2`,
which is not an optical frame; the pinhole frame is its child `ZED2_pcl`.

```bash
apairo-sim-amiad extract --out ~/data/sim_amiad/full
apairo-sim-amiad label ~/data/sim_amiad/full
apairo-sim-amiad agreement ~/data/sim_amiad/full
apairo-sim-amiad overlay ~/data/sim_amiad/full --every 1000 --png-dir /tmp/overlays
apairo-sim-amiad export ~/data/sim_amiad/full --lerobot-root ~/data/sim_amiad/lerobot
apairo-sim-amiad check ~/data/sim_amiad/lerobot
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../apairo -e ../apairo_preprocess   # sibling repos, editable
pip install -e ".[test]"                           # this package + lerobot>=0.4.0
pip install -e ".[sim]"                            # + rosbags, for apairo-sim-amiad extract
```

With `uv`, `apairo_preprocess` pins `apairo-transform` to its git URL, which
conflicts with an editable sibling; override it:

```bash
printf "apairo-transform @ file://$PWD/../apairo_transform\napairo @ file://$PWD/../apairo\n" > /tmp/overrides.txt
uv pip install --override /tmp/overrides.txt -e ../apairo -e ../apairo_transform -e ../apairo_preprocess -e ".[test]"
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
[apairo_ros](https://github.com/apairo-robotics/apairo_ros), which
reads ROS bags and writes the `.apairo` sidecars (`channels.yaml`, calibration
from `/tf_static`) — ready for `TartanKittiDataset` / `RawDataset`:

```bash
pip install apairo-ros

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
