---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - LeRobot
  - lerobot
  - robotics
  - tartandrive
  - off-road
  - traversability
  - bev
  - elevation-map
  - apairo
pretty_name: TartanDrive 2.0 robot-centred BEV grids (LeRobot format)
---

# TartanDrive 2.0: robot-centred BEV grids (elevation, geometry, traversability)

One bird's-eye-view grid per lidar scan, centred on the vehicle, for
TartanDrive 2.0: elevation, a geometric roughness / step layer, and a
self-supervised traversability label from where the vehicle actually drove.

**The point of this dataset is the recipe, not the data.** The grids are
built by the *same* apairo recipe as the RELLIS-3D BEV grids
(`apairo-robotics/rellis3d_bev`): same grid, same past-only accumulation,
same driven corridor, same code (`apairo_huggingface.bev_grid`). Porting it
to TartanDrive took a different apairo declaration (`apairo_bev.yaml`, four
raw channels) and the few numbers the rig imposes. Everything else -- reading
an asynchronous multi-rate recording, zipping scans with their poses,
materialising three derived channels with recipe-hashed reuse, putting the
camera and the command on the scan clock -- is
[apairo](https://github.com/apairo-robotics/apairo)
(DOI [10.5281/zenodo.23768995](https://doi.org/10.5281/zenodo.23768995)).
Source: `apairo_huggingface.tartan_bev` (`apairo-tartan-bev`), in
[apairo_huggingface](https://github.com/apairo-robotics/apairo_huggingface).

![camera | mean height | roughness | step | trav](tartan_bev_preview.jpg)

## What was ported, and what changed

| | RELLIS-3D BEV | TartanDrive BEV |
|---|---|---|
| grid | 200 x 200, 0.2 m, 30 m ahead / 10 m behind / 20 m each side | same (shared module) |
| frame | gravity-aligned, origin at the lidar, +x heading, +y left | same |
| heading in the sensor frame | Ouster -x | vehicle +x |
| heights | lidar z + 1.0 m | lidar z + 1.77 m |
| accumulation | 10 scans, stride 2, past only (~2 s) | same |
| driven corridor | footprint +-0.7 m, 10 m back / 30 m ahead, all-unknown if < 10 m of path ahead | footprint +-0.8 m (Yamaha Viking), rest same |
| self-returns | 2 m radius + a box behind the lidar | 2 m radius (covers the roof and mounts) |
| per-point labels | yes: `bev_semantic` (majority class) | **none**: replaced by `bev_geometry` (roughness, step) |
| poses | `poses.txt` (csv channel, clock from the camera names) | `gicp_poses` (one per scan, declared on the lidar clock) |
| quality check | driven cells vs their class (driveable / obstacle) | driven cells vs their geometry (flat share) |

## Features (10 Hz, one frame per scan; each recording cut into episodes of at most 1000 frames)

| feature | shape / type | content |
|---|---|---|
| `observation.images.front` | (272, 512, 3) video | left multisense camera, rectified, halved; the image nearest the scan (within 0.1 s) |
| `observation.bev.elevation` | (3, 200, 200) float32 | min / max / mean point height in the cell above the ground under the vehicle (m), NaN unknown |
| `observation.bev.geometry` | (2, 200, 200) float32 | `roughness`: std of the point heights in the cell (NaN under 2 points); `step`: largest jump of the mean height to a known 8-neighbour (NaN if none); m |
| `observation.bev.trav` | (200, 200, 3) PNG | 1 driven / 0 seen (a point fell), not driven / 255 unknown; value replicated in the 3 channels |
| `observation.state` | (7,) float32 | GICP pose relative to the episode's first scan, `[x, y, z, qx, qy, qz, qw]` |
| `action` | (2,) float32 | executed forward speed (m/s) and yaw rate (rad/s), from this scan's and the next scan's pose |
| `observation.command` | (2,) float32 | the raw `cmd` channel `[linear, angular]` nearest the scan |
| `observation.frame_info` | (3,) int64 | `[recording_index, scan_index, camera_index]`; recording names in `meta/apairo_recordings.json` |

Grid layout: row 0 is 30 m ahead, column 0 is 20 m left, the vehicle (lidar)
sits at the corner of row 150 / column 100 and drives up the image. Grid
frame: origin at the lidar, +x the vehicle heading projected on the
horizontal plane, +y left, z up (world z of the GICP map).

Elevation and geometry use only the current and past scans (scan j
accumulates j, j-2, ..., j-18); `bev_trav` is a label and uses the future
path. Driven cells are 1 even where no point fell.

LeRobot's per-element statistics are NaN for the float grids: mask the NaNs
before normalising.

## Stats

13 recordings, every site of the release and both seasons (September and
October-November 2023): 27,561 scans, 46 min of driving, 10.7 km of GICP
path. The full release holds 55 recordings with scans (~125.7k scans).

| recording | scans | repeated | path (km) | frames with trav | driven / seen / unknown | driven observed | driven flat | near (15 m) driven flat | seen-not-driven flat |
|---|---|---|---|---|---|---|---|---|---|
| turnpike_2023-09-12-12-39-19 | 1764 | 149 | 1.15 | 1513 | 0.038 / 0.511 / 0.451 | 0.545 | 0.888 | 0.885 | 0.614 |
| turnpike_flat_2023-09-12-14-21-03 | 2390 | 45 | 0.80 | 2362 | 0.039 / 0.445 / 0.516 | 0.510 | 0.825 | 0.819 | 0.508 |
| turnpike_warehouse_2023-09-13-19-03-05 | 3050 | 32 | 1.46 | 2859 | 0.039 / 0.381 / 0.580 | 0.512 | 0.909 | 0.919 | 0.288 |
| figure_8_morning_slow_2023-09-12-11-06-32 | 1807 | 57 | 0.70 | 1782 | 0.040 / 0.380 / 0.580 | 0.511 | 0.878 | 0.871 | 0.313 |
| figure_8_2023-09-14-11-37-49 | 2999 | 58 | 0.69 | 2961 | 0.039 / 0.291 / 0.670 | 0.455 | 0.590 | 0.587 | 0.172 |
| gupta_skydio_2023-09-14-11-03-09 | 1467 | 7 | 0.50 | 1423 | 0.038 / 0.393 / 0.569 | 0.415 | 0.808 | 0.777 | 0.361 |
| warehouse_gupta_2023-09-13-16-28-57 | 1831 | 9 | 0.98 | 1798 | 0.040 / 0.422 / 0.539 | 0.514 | 0.883 | 0.894 | 0.397 |
| meadows_2023-09-14-12-18-22 | 2255 | 28 | 0.62 | 2160 | 0.038 / 0.285 / 0.677 | 0.421 | 0.575 | 0.588 | 0.182 |
| slag_heap_2023-09-14-12-23-05 | 1711 | 8 | 0.23 | 1544 | 0.037 / 0.329 / 0.634 | 0.408 | 0.687 | 0.695 | 0.237 |
| sam_loop_2023-09-27-13-04-54 | 1252 | 7 | 0.82 | 1236 | 0.040 / 0.380 / 0.580 | 0.511 | 0.889 | 0.898 | 0.219 |
| 2023-11-02-16-00-43_down_meadows | 2155 | 4 | 0.72 | 1790 | 0.038 / 0.405 / 0.557 | 0.517 | 0.809 | 0.825 | 0.294 |
| 2023-11-14-14-34-53_gupta | 2610 | 5 | 1.14 | 2533 | 0.040 / 0.462 / 0.498 | 0.496 | 0.884 | 0.884 | 0.456 |
| 2023-11-14-15-22-01_figure_8 | 2270 | 4 | 0.90 | 2077 | 0.039 / 0.409 / 0.552 | 0.453 | 0.788 | 0.771 | 0.271 |
| **all** | **27561** | **413** | **10.7** | **26038** | **0.039 / 0.388 / 0.573** | **0.483** | **0.801** | **0.799** | **0.346** |

"Flat" = roughness < 0.10 m and step < 0.20 m (both known); shares of flat
are over cells where both are known. The driven corridor is flat in 80% of
its observed cells against a 35% base rate in the seen-not-driven cells
(ratio 2.3x; 1.4x to 4.1x per recording). Each criterion alone: roughness
0.907, step 0.829. The lowest shares are the vegetated recordings (meadows
0.575, figure_8 0.590, slag heap 0.687), where the Viking drives through grass
and brush that reads rough: the proxy is geometric, not a label. Half the
driven cells hold no point (0.483 observed), as on RELLIS (0.65-0.69): the
unseen disc around the vehicle and the hidden rear. Frames with less than
10 m of path ahead (1523, mostly stops and recording ends) have an all-unknown
trav grid.

### Export check (local LeRobotDataset, reloaded)

`apairo-tartan-bev check` on the written dataset: 34 episodes, 27,148 frames
(= 27,561 scans - 413 repeats; no frame dropped by the 0.1 s camera / cmd
tolerance). Recordings are cut into equal consecutive episodes of at most
1000 frames (622-981 here), because LeRobot holds an episode's float grids in
memory (8.7 GB peak RSS for 1000-frame episodes);
`meta/apairo_recordings.json` maps episodes back to recordings and
`observation.frame_info` to scans.

| check | value |
|---|---|
| trav values | {0, 1, 255}; cells 3.7% driven / 36.9% seen / 59.5% unknown (all frames, incl. all-unknown ones) |
| elevation known share | 0.404 (every 50th frame) |
| mean height p1 / p50 / p99 | -2.80 / 0.86 / 9.93 m (trees, slopes; not terrain-normalised) |
| roughness p50 / p99 | 0.083 / 1.85 m |
| step p50 / p99 | 0.251 / 4.29 m |
| action forward speed p1 / p50 / p99 | -0.10 / 3.75 / 10.65 m/s (6 frames > 15 m/s and 100 < -1 m/s: pose jumps or reversing) |
| yaw rate range | -1.98 / 1.53 rad/s |
| command range | linear [0, 1], angular [-1, 1] |

Sizes: 12 GB of apairo grids (440 KB per scan) + 5.8 GB LeRobot (5.3 GB
parquet, 505 MB video) = 17.8 GB for the subset; the full release would be
~55 GB + ~26 GB. Time: ~15 min to process (three recordings in parallel,
185 MB RSS each), 31 min to export.

## Known limitations

- **Heights are not terrain-normalised.** They are relative to the ground
  under the vehicle along world z: on a climb the ground ahead reads metres
  high. Use `step` / `roughness` for local geometry.
- **World z is the first scan's vehicle z.** GICP poses start at identity, so
  the vehicle's attitude at the first scan (a few degrees) tilts the map and
  leaks into heights as a slope (~0.5 m over 30 m per degree).
- **Behind the vehicle is mostly unknown**: the cab hides the rear from the
  roof lidar, and no point falls within 2 m.
- **Vegetation reads rough.** Tall grass the Viking drives through has high
  roughness / step: the flat share is a conservative proxy of driveable, not
  a label.
- **Accumulation smears** moving objects and pose drift over ~2 s; ring
  patterns in the step layer come from scans placed a few cm apart in z.
- **Repeated scans.** The source repeats some scans under the same stamp
  (identical points; 4-149 per recording here, 413 in all); apairo keeps one grid per scan,
  the LeRobot export keeps the first of each repeat.
- **`cmd` linear is almost always 0**: in 10 of the 13 recordings it is 0
  throughout, in two it is nonzero in < 0.3% of the rows, only in sam_loop is
  it set (88%); the angular part is the steering command in [-1, 1]. `action`
  therefore comes from the poses, and `cmd` is kept as `observation.command`.
  The `action` of `apairo-robotics/tartandrive_offroad` is this `cmd`.
- **Pose glitches.** A few frames have implausible pose-derived speeds
  (6 above 15 m/s); `action` is not filtered.
- **A subset.** 13 of the 55 recordings with scans (see Stats); the full set
  is ~125.7k scans.

## Reproduce

```bash
pip install -e ../apairo -e ../apairo_preprocess -e ../apairo_transform -e ".[test]"
apairo-tartan-bev process --root /path/to/tartan --out ~/data/tartan_bev          # the 13 recordings of config.SEQUENCES
apairo-tartan-bev render ~/data/tartan_bev/turnpike_2023-09-12-12-39-19 --every 500 --png-dir /tmp/tbev
apairo-tartan-bev quality --out ~/data/tartan_bev
apairo-tartan-bev export ~/data/tartan_bev --lerobot-root ~/data/tartan_bev_lerobot
apairo-tartan-bev check ~/data/tartan_bev_lerobot
```

`--root` is the TartanDrive 2.0 tree in the apairo layout (one directory per
recording with `velodyne_0/`, `gicp_poses/`, `image_left_color/`, `cmd/`, as
written by [apairo_extractor](https://github.com/apairo-robotics/apairo_extractor)
from the rosbags; the GICP poses are those of the source). It is only read:
each recording is a symlink view under `--out`, where apairo writes the
derived channels and its registry.

## Source data

TartanDrive 2.0 was collected at Carnegie Mellon University's AirLab with a
modified Yamaha Viking ATV driven across diverse off-road terrain. The
recordings are the rosbag distribution converted to the apairo layout, the
same tree that produced `apairo-robotics/tartandrive_offroad`.

## Citation

TartanDrive 2.0:

```bibtex
@inproceedings{sivaprakasam2024tartandrive,
  title={TartanDrive 2.0: More Modalities and Better Infrastructure to Further Self-Supervised Learning Research in Off-Road Driving Tasks},
  author={Sivaprakasam, Matthew and Maheshwari, Parv and Castro, Mateo Guaman and Triest, Samuel and Nye, Micah and Willits, Steve and Saba, Andrew and Wang, Wenshan and Scherer, Sebastian},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

apairo (the pipeline):

```bibtex
@software{bresset2026apairo,
  title={apairo: a unified loader for robotics sensor datasets},
  author={Bresset, Augustin},
  year={2026},
  version={0.9.0},
  doi={10.5281/zenodo.23768995},
  url={https://github.com/apairo-robotics/apairo}
}
```

## License

The original [TartanDrive dataset](https://theairlab.org/tartandrive-dataset/)
is explicitly released under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). TartanDrive 2.0 does
not state a separate data license (its repository's MIT license covers the
code); this derived dataset is published under CC BY 4.0 accordingly, with
full attribution to the AirLab authors above. The conversion code
([apairo_huggingface](https://github.com/apairo-robotics/apairo_huggingface))
is MIT.
