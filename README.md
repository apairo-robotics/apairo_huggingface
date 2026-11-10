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
  bev_grid.py    the robot-centred BEV grid shared by rellis/bev.py and tartan_bev/bev.py
  tartan_bev/    TartanDrive BEV grids (bev, quality, render, export, cli: apairo-tartan-bev)
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


## RELLIS-3D: robot-centred BEV grids (elevation, semantic, traversability)

RELLIS-3D has no bird's-eye-view labels, yet top-down grids are what planners
consume. `apairo_huggingface.rellis.bev` builds three grids for **every scan**,
centred on that scan's pose, as apairo **derived channels** (one
`FramePreprocessor` with three `output_keys`, materialised by
`run_preprocess`, recipe-hashed so an unchanged rerun is a no-op):

| channel | per frame | values |
|---|---|---|
| `bev_elevation` | `(3, 200, 200)` float16 | min / max / mean height above the ground under the robot (m), NaN unknown |
| `bev_semantic` | `(200, 200)` uint8 | majority ontology id of the points in the cell (void only if alone), 255 unknown |
| `bev_trav` | `(200, 200)` uint8 | 1 driven / 0 seen, not driven / 255 unknown |

Design choices (all in `rellis/config.py`, "BEV grids"):

- **Grid**: 0.2 m cells, 40 x 40 m, forward-biased: 30 m ahead, 10 m behind,
  20 m either side. Frame: origin at the lidar, +x the robot heading (the
  Ouster's -x) projected on the world xy plane, +y left, z the world z of the
  RELLIS poses (not checked against gravity, so slope and pitch drift of the
  pose world are kept).
  Row 0 is 30 m ahead, column 0 is 20 m left: the robot drives up the image.
- **Heights** are lidar z + 1.0 m (the measured lidar height), so the ground
  under the robot reads ~0.
- **Accumulation, past only**: the grid of scan j uses scans j, j-2, ..., j-18
  (10 scans spanning 1.8 s at 10 Hz, ~2.3 m at the median 1.3 m/s), placed with `poses.txt`.
  Nothing later than j enters the inputs (a test rewrites the future scans and
  checks the grid does not move). Moving objects (people walking beside the
  robot) smear over those 1.8 s.
- **Self-returns**: points within 2 m, and a sensor-frame box 0-3 m behind the
  lidar (robot-attached returns, mostly labelled void, present in most scans)
  are dropped. The lowest Ouster beam meets the ground ~3.4 m out, so a disc
  around the robot stays unknown.
- **Traversability** is a label and may use the future: the robot footprint
  (0.7 m either side of the path) from 10 m of path behind the scan to 30 m
  ahead. Driven cells are 1 even where no point fell; seen cells (any point)
  are 0 otherwise; frames with less than 10 m of path ahead are all 255, as
  in the image masks.
- **Clock**: scans have no stamp; the view's `poses` channel (the symlinked
  `poses.txt`, read by apairo's `csv` format) gets a `timestamps.txt` written
  once from the camera filenames, and the scans and labels borrow it -- no
  image is decoded to process a sequence.

`apairo-rellis bev-quality` checks the two independent sources against each
other: the share of driven cells whose majority class is driveable (dirt,
grass, asphalt, concrete, puddle, mud), the share that is an obstacle (tree,
bush, barrier, person, ...), and the driveable share of seen-not-driven cells
as the base rate. All five sequences (every frame with a trav grid):

| seq | frames | driven cells observed | driven: driveable | driven: obstacle | (of which person) | seen-not-driven: driveable |
|---|---|---|---|---|---|---|
| 00000 | 2521 | 0.659 | 0.939 | 0.061 | 0.042 | 0.333 |
| 00001 | 2037 | 0.674 | 0.966 | 0.028 | 0.022 | 0.296 |
| 00002 | 4013 | 0.651 | 0.894 | 0.101 | 0.070 | 0.608 |
| 00003 | 1866 | 0.686 | 0.962 | 0.034 | 0.030 | 0.616 |
| 00004 | 1870 | 0.672 | 0.941 | 0.054 | 0.053 | 0.616 |

89-97% of the driven cells are driveable, against a 30-62% base rate. Most
of the obstacle share is `person`: the operators walk along the trail the
robot later drives (the trav label uses the future path, the semantic grid
the past scans), plus RELLIS's ghost person labels on empty ground. Without
person the obstacle share is 0.1-3.1%. A third of the driven cells hold no
point (the unseen disc around the robot, sparse far ground).

Processing takes 0.12 s per scan (265 s for the 2184 scans of 00003, ~16 min
for all 13556 with two sequences in parallel, 0.55 GB peak RSS); the five
sequences hold 4.1 GB of grids (npy: 230 KB elevation + 2 x 40 KB per scan).

```bash
apairo-rellis bev-process --root /data/rellis --out ~/data/rellis_bev      # all 5 sequences
apairo-rellis bev-render ~/data/rellis_bev/00003 --every 500 --png-dir /tmp/bev
apairo-rellis bev-quality --out ~/data/rellis_bev
apairo-rellis bev-export ~/data/rellis_bev --root /data/rellis --lerobot-root ~/data/rellis_bev_lerobot
apairo-rellis bev-check ~/data/rellis_bev_lerobot
```

The export is a separate LeRobotDataset (one episode per sequence, one frame
per scan): the camera image at 480x300 (video), `observation.bev.elevation`
(float32, NaN unknown -- LeRobot's per-element statistics are NaN for it, so
mask before normalising), `observation.bev.semantic` and `observation.bev.trav`
(lossless PNG, the id in every channel), plus the pose, action and frame info
of the image export. It is written locally only: 13556 frames, 5 episodes,
2.1 GB (2.0 GB parquet, 126 MB video), 14 min, but 18 GB peak RSS (LeRobot
holds an episode's float arrays in memory; 00002 has 4147 frames).
Elevation is relative to the ground under the robot, not terrain-normalised:
on a climb the ground ahead reads up to +1.3 m.

## TartanDrive 2.0: the same BEV recipe on another dataset

`apairo_huggingface.tartan_bev` runs the RELLIS BEV recipe on TartanDrive
2.0, to show an apairo recipe is portable across datasets. The grid, its
frame, the binning, the elevation layer, the past-only accumulation, the
driven corridor and the preprocessor's shared machinery (`BevGridBase`:
recipe scalars incl. grid resolution and bounds, path length, scan cache)
live in one shared module, `apairo_huggingface/bev_grid.py`; state and
action from poses live in `apairo_huggingface/poses.py`, used by the three
exports. The RELLIS grids, recipe hash, states and actions are bit-identical
before and after the factoring. What the port changes:

- the **declaration** (`tartan_bev/apairo_bev.yaml`): the raw channels of a
  recording (`velodyne_0`; `gicp_poses` and its `valid_mask.npy` as
  `gicp_valid`, both on the lidar clock; `gps_odom`, the INS;
  `image_left_color`; `cmd`), read through a symlink view;
- the **up axis**: gravity from the INS attitude at each scan, not the GICP
  world z, which is the first scan's vehicle frame and drifts from gravity
  by 1-20 deg (median per recording, 26 deg at worst). Against GPS altitude,
  the height of the ground 15-25 m ahead is off by 0.14-0.30 m (median per
  recording) where GICP's z gave 0.33-4.70 m;
- **invalid scans** (pose flagged in `valid_mask.npy`, or no INS row within
  0.1 s; 13 scans) get all-unknown grids, enter no other grid nor the driven
  corridor, and are not exported;
- the **rig numbers**: heading = vehicle +x, lidar 1.77 m above the ground,
  footprint +-0.8 m, a 2 m radius for the self-returns;
- the **semantic layer**, which TartanDrive cannot have (no point labels), is
  replaced by `bev_geometry` `(2, H, W)` float16: `roughness` (std of the
  point heights in the cell) and `step` (largest mean-height jump to a known
  8-neighbour), both from the inputs only;
- the **quality check** becomes geometric: the flat share (roughness < 0.10 m
  and step < 0.20 m) of driven cells against that of seen-not-driven cells.

Processed: 13 of the 55 recordings with scans (`tartan_bev/config.SEQUENCES`:
every site, both seasons), 27,561 scans, 10.7 km of path. Over the 26,038
frames with a trav grid, cells are 3.9% driven / 38.9% seen / 57.2% unknown;
48% of driven cells hold a point. **81% of the observed driven cells are flat
against 35% of the seen-not-driven cells** (2.3x; 1.4-4.0x per recording;
lowest on the vegetated meadows / figure_8 / slag heap recordings, where the
Viking drives through brush that reads rough). Full table in the card.

Processing: 0.04 s per scan alone, ~0.1 s with three recordings in parallel
on the shared HDD (~13 min for the 13), 190 MB peak RSS per process; 12 GB of
grids (npy: 240 KB elevation + 160 KB geometry + 40 KB trav per scan). The
full release (~125.7k scans) would take ~55 GB of grids.

The source repeats some scans under the same stamp (413 of the 27,561:
identical points, pose re-registered within mm); `BevGrids` tells them apart
by their pose, the export keeps the first. The `cmd` channel's linear part is
0 in 10 of the 13 recordings (set only in sam_loop), and that `cmd` is the
action of the published tartandrive_offroad, so the BEV export's `action` is the executed speed and yaw rate from the poses, as
in the RELLIS BEV export, with `cmd` kept as `observation.command`.

Export (`apairo-tartan-bev export`, then `check`): 34 episodes / 27,135
frames (the 413 repeats and 13 invalid scans dropped, nothing dropped by the 0.1 s camera / cmd
tolerance), recordings cut into equal episodes of at most 1000 frames because
LeRobot holds an episode's float grids in memory: 9.0 GB peak RSS, 21 min,
5.8 GB (5.3 GB parquet, 504 MB video); the full release would add ~26 GB.
Reload check: trav values {0, 1, 255}, elevation known share 0.40, forward
speed p50 3.75 m/s.

```bash
apairo-tartan-bev process --root /mnt/vault-fellowship/tartan --out ~/data/tartan_bev   # config.SEQUENCES
apairo-tartan-bev render ~/data/tartan_bev/turnpike_2023-09-12-12-39-19 --every 500 --png-dir /tmp/tbev
apairo-tartan-bev quality --out ~/data/tartan_bev
apairo-tartan-bev export ~/data/tartan_bev --lerobot-root ~/data/tartan_bev_lerobot
apairo-tartan-bev check ~/data/tartan_bev_lerobot
```

Dataset card: [`hub/tartan_bev_dataset_card.md`](hub/tartan_bev_dataset_card.md).

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
