# Hand-labeled evaluation benchmark (TartanDrive 2.0)

Status: sequence selection settled; publishing format is an open decision
(see "Publishing hand labels" below).

## Why a separate eval benchmark

`tartandrive_offroad` (the LeRobotDataset this repo publishes) is
self-supervised: `trav_mask` only knows "the robot drove here" (`1`) vs.
"lidar saw this, no label" (`0`) vs. unobserved (`255`) — it has no notion of
a *true negative* (an obstacle the robot never approached). Hand-labeling a
subset of frames with [toaster](https://github.com/augustin-bresset/toaster)
gives real `traversable` / `obstacle` negatives to evaluate models against,
and to mix into training if useful.

## TartanDrive 2.0 has no official split

Checked the upstream `tartan_drive_2.0` repo (README, `tartandrive_cli.py`,
`tartandrive_gui.py`, `assets/files.yaml`): no train/eval split, no
benchmark subset is designated anywhere. Every one of the 55 recordings is
just `<site>_<timestamp>/`, flat, no metadata beyond the name.

## The 55 sequences, by site

Site clusters, inferred from naming + a look at the already-labeled data on
`/mnt/vault-fellowship/tartan`:

| Site | Terrain | # sequences | Frames (min–max) |
|---|---|---|---|
| `turnpike*` | gravel/double-track road, baseline terrain | 17 | 332–8714 |
| `gupta*` | wooded hillside, roots/ruts, most technical natural terrain | 12 (2 empty) | 97–5287 |
| `figure_8*` | dedicated test loop, controlled geometry | 10 | 73–7472 |
| `meadows*` / `down_meadows` | open grass field, flat but soft/tall-grass occlusion | 7 | 30–3368 |
| `warehouse` (combined with turnpike/gupta) | paved lot, near in-distribution / structured | 6 | 223–3139 |
| `slag_heap*` | steep loose rubble pile, extreme terrain | 3 | 245–1806 |
| `sam_loop*` | distinct site, underrepresented | 2 | 104–1253 |
| `rough_rider_turnpike_loop` | combined loop | 1 | 4669 |

`gupta_0` and `gupta_2023-09-14-10-44-56` have 0 frames (broken captures) —
excluded from consideration entirely.

## Proposed eval selection

One sequence per distinct terrain, sized for hand-labeling (not 30 frames,
not 8700):

| Sequence | Frames | Why |
|---|---|---|
| `turnpike_2023-09-12-12-42-49` | 2180 | baseline terrain, gravel road |
| `gupta_2023-09-14-11-12-19` | 4367 | most technical natural terrain |
| `meadows_2023-09-14-12-08-57` | 3368 | open field, tall-grass occlusion |
| `slag_heap_2023-09-14-12-23-05` | 1712 | steep/loose, sharp traversability contrast |
| `turnpike_warehouse_2023-09-13-19-03-05` | 3051 | paved/structured, near out-of-terrain-distribution |
| `figure_8_2023-09-14-11-37-49` | 3000 | controlled loop, sanity-check baseline |
| `sam_loop_2023-09-27-13-04-54` | 1253 | distinct site, cheap extra diversity |
| `2023-10-26-14-42-35_turnpike_afternoon_fall` | 5042 | seasonal/lighting domain shift on known terrain |

Optional, for stress-testing generalization further:
`figure_8_to_turnpike_2023-09-12-12-22-45` (terrain transition mid-sequence),
`turnpike_flat_2023-09-12-13-42-26` (easy/hard contrast within one site).

The other ~45 sequences stay in the training pool; every site is still
represented there, so nothing is entirely unseen at train time except the
`_fall` lighting shift.

## Publishing hand labels

toaster writes, per labeled scan: `scan_<idx>_toaster.npy` (per-point
`0`=unlabeled / `1`=traversable / `2`=obstacle) + a schema yaml. The
`apairo_experiments` repo already has an importer
(`scripts/import_toaster_gt.py`) that turns these into a sparse
`ground_truth` apairo channel, aligned to the raw point cloud.

Constraints found while scoping the HF/LeRobot side:

- **LeRobotDataset can't hold this directly.** Every feature is declared with
  a fixed `shape` up front (`feature_utils.py` maps 2D shapes to
  `datasets.Array2D`, etc.) — a lidar scan's point count varies frame to
  frame, so it doesn't fit the schema without padding/truncating (lossy).
- **Plain `datasets.Dataset` *can* store ragged point clouds** —
  `Sequence(Sequence(Value("float32"), length=3))` allows a variable outer
  length — this is a LeRobotDataset-specific limitation, not an HF Hub one.
- **The Hub viewer has no 3D widget either way.** Datasets on the Hub that
  ship `.ply`/`.pcd`/parquet point clouds render as plain file listings or
  raw list cells; people view them with Open3D/FiftyOne outside the Hub.

Decision: publish both — the 2D projected mask *and* the raw point cloud with
per-point labels, in the same row, so the benchmark supports image-based eval
today and invites 3D-native eval later (as the Hub's point-cloud story
improves). Implemented:

- `eval_labeling.label_ground_truth_sequence` / `label_ground_truth_root` —
  projects `ground_truth` through the same `LidarCameraProjection` +
  `ImageMaskFromPointLabels` primitives `trav_mask` already uses (no
  trajectory/corridor logic — the labels are already ground truth), writing
  a `trav_mask_gt` image channel (0 unlabeled / 1 traversable / 2 obstacle),
  registered `kind: preprocess` like every other derived channel. Recordings
  without `ground_truth` are skipped, not failed.
- `eval_export.to_eval_dataset` — builds a **plain `datasets.Dataset`** (not
  a LeRobotDataset — it's a curated frame subset, not a continuous
  trajectory, and its point-cloud column is ragged, which a LeRobotDataset
  feature cannot express). One row per hand-labeled frame:
  `sequence`, `timestamp`, `image`, `trav_mask_gt` (both `datasets.Image`,
  Hub-viewer-renderable), `points` (ragged `(N, 3)` float32), `point_labels`
  (ragged `(N,)` int8, aligned to `points`).
- CLI: `apairo-tartan-gt-label` (project) and `apairo-tartan-gt-export`
  (build/push, `--dry-run` to validate reads only) — mirrors
  `apairo-tartan-label` / `apairo-tartan-export`.

Does this integrate with LeRobot? No, deliberately — LeRobot is a
trajectory/policy-training format (continuous episodes, one fixed shape per
feature, video encoding); this is an evaluation benchmark of a sparse,
curated frame subset with a variable-size modality. The two stay separate
repos (`tartandrive_offroad` unchanged; new `tartandrive_offroad_eval`), but
join naturally on `sequence` + `timestamp` for apples-to-apples comparison
against the self-supervised `trav_mask`.

Not yet done: actually labeling frames with toaster (the sequences above are
the target list) and running `import_toaster_gt.py` + this pipeline against
real recordings.
