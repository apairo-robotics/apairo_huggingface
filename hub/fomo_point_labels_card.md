---
license: cc-by-4.0
task_categories:
  - robotics
tags:
  - robotics
  - fomo
  - off-road
  - forest
  - multi-season
  - traversability
  - lidar
  - point-cloud
  - apairo
pretty_name: FoMo Multi-Season Traversability - per-point lidar labels
---

# FoMo Multi-Season Traversability - per-point lidar labels

> **A labels-only companion.** Self-supervised, per-point traversability
> labels for every RoboSense Ruby Plus scan of the twelve "red" trajectories
> of [FoMo](https://fomo.norlab.ulaval.ca) (November 2024 to November 2025,
> bare ground to over 1 m of snow), plus the KISS-ICP pose of each scan. The
> scans themselves are **not** redistributed: download them from FoMo and
> rebuild the label mask from the stored indices. Built with
> [**apairo**](https://github.com/apairo-robotics/apairo).
>
> Companion of the image dataset `apairo-robotics/fomo_traversability`
> (LeRobot format), whose camera masks are projected from these labels.

## Layout

```
data/<deployment>/<trajectory>.parquet      # one per trajectory, one row per scan
```

| Column | Type | Meaning |
|---|---|---|
| `trajectory` | string | `<deployment>/<trajectory>`, the FoMo bucket path |
| `scan_stem` | string | `NNNNNN`, the scan's rank in the trajectory (sorted by stamp) |
| `robosense_file` | string | `<us>.bin`, the raw FoMo file under `<trajectory>/robosense/` |
| `timestamp_us` | int64 | the file's microsecond epoch stamp (its name) |
| `n_points` | int32 | points in the raw scan (file size / 26 bytes) |
| `driven_indices` | list<int32> | row indices of the points labelled `1` |
| `pose` | 16 x float64 | row-major `T_world_lidar` (KISS-ICP, deskewed; world = the first scan's lidar frame) |

Labels: `1` = the point lies under the robot's footprint (1.4 m wide) swept
along its **future** driven trajectory, up to 30 m of path ahead, in a height
window around the ground; `0` = any other point. About 1.3 % of all points (64.7 M of
5.11 billion) are `1`, which is why the indices are stored rather than full
masks.
`0` means "not demonstrated", never "untraversable".

The labels are computed on **every** point of the raw scan, in file order (no
range or ring filtering), so `driven_indices` index the rows of FoMo's
`robosense/<us>.bin` directly. Checked against the bucket: on
`2025-10-14/red_2025-10-14-11-41`, all 1,692 file stamps equal the label
stamps and every file size equals 26 x `n_points`, and a downloaded scan
reads back with exactly `n_points` rows.

| Trajectory | Scans |
|---|---|
| `2024-11-21/red_2024-11-21-10-34` | 2,540 |
| `2024-11-28/red_2024-11-28-09-30` | 2,251 |
| `2025-01-10/red_2025-01-10-12-02` | 3,834 |
| `2025-01-29/red_2025-01-30-09-58` | 3,168 |
| `2025-03-10/red_2025-03-10-16-53` | 2,108 |
| `2025-04-15/red_2025-04-15-51-56` | 4,627 |
| `2025-05-28/red_2025-05-28-10-46` | 1,929 |
| `2025-06-26/red_2025-06-26-10-31` | 1,825 |
| `2025-08-20/red_2025-08-20-10-42` | 1,934 |
| `2025-09-24/red_2025-09-24-10-45` | 1,817 |
| `2025-10-14/red_2025-10-14-11-41` | 1,692 |
| `2025-11-03/red_2025-11-03-18-30` | 1,924 |
| **total** (114 MB) | **29,649** |

All twelve trajectories are included, `2025-01-29/red_2025-01-30-09-58`
too: its camera is saturated (white images, -24 degC) and it is left out of
the image dataset, but the lidar labels do not depend on the camera.

## Rebuild the per-point mask

A FoMo RoboSense file is a headerless array of packed little-endian 26-byte
records `x y z i` (float32), `r` ring (uint16), `t` per-point time (uint64,
us).

```python
import numpy as np
import pandas as pd
import urllib.request

RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"),
                   ("r", "<u2"), ("t", "<u8")])

scans = pd.read_parquet("data/2025-10-14/red_2025-10-14-11-41.parquet")
row = scans.iloc[944]
url = ("https://fomo-dataset.s3.amazonaws.com/data/"
       f"{row.trajectory}/robosense/{row.robosense_file}")
cloud = np.frombuffer(urllib.request.urlopen(url).read(), dtype=RECORD)
assert len(cloud) == row.n_points

label = np.zeros(row.n_points, dtype=np.uint8)
label[row.driven_indices] = 1                     # 1 driven / 0 other, row for row
xyz = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1)
T_world_lidar = np.asarray(row.pose).reshape(4, 4)
```

### With apairo

[`apairo_huggingface`](https://github.com/apairo-robotics/apairo_huggingface/tree/feature/fomo)
ships an apairo declaration of a FoMo trajectory and a `fomo_bin` format
plugin for the RoboSense files, so a downloaded trajectory loads with
per-sensor clocks read from the file names:

```python
import pandas as pd
from apairo import RawDataset
from apairo_huggingface.fomo.labeling import declaration
from apairo_huggingface.fomo.point_labels import mask

ds = RawDataset("/data/fomo/2025-10-14/red_2025-10-14-11-41",
                declare=declaration(), keys=["robosense"])
scans = pd.read_parquet("data/2025-10-14/red_2025-10-14-11-41.parquet")
i = 944                                            # scan_stem "000944"
assert round(ds.timestamps["robosense"][i] * 1e6) == scans.timestamp_us[i]
label = mask(scans.n_points[i], scans.driven_indices[i])   # rows of ds.loaders["robosense"][i]
```

## How the labels are built

1. **Odometry** - KISS-ICP on the deskewed RoboSense scans (per-point time,
   2-80 m range for registration only) gives one `T_world_lidar` per scan.
2. **Point labels** - every point of the raw scan inside the robot footprint
   swept along the future trajectory is labelled `1`
   (`TraversabilityFromTrajectory` from
   [`apairo_preprocess`](https://github.com/apairo-robotics/apairo_preprocess);
   half-width 0.7 m, look-ahead 30 m of path, height window -0.3/+0.5 m
   around the ground below the future pose).

KISS-ICP drift is 0.10-0.21 m RMSE over ~300 m on every trajectory against
FoMo's GNSS-PPK ground truth (per-trajectory table in the image dataset card).

```bash
pip install "apairo-huggingface @ git+https://github.com/apairo-robotics/apairo_huggingface@feature/fomo"
apairo-fomo run --root /data/fomo --color red --drop-lidar     # fetch, odometry, labels
apairo-fomo export-labels /data/fomo --out /data/fomo_point_labels
```

**Source, all open:**
[apairo_huggingface](https://github.com/apairo-robotics/apairo_huggingface)
(this exporter, branch `feature/fomo`) -
[apairo](https://github.com/apairo-robotics/apairo) (the data layer,
[doi:10.5281/zenodo.23768995](https://doi.org/10.5281/zenodo.23768995)).

## Citation

If you use these labels, please cite FoMo:

```bibtex
@misc{Boxan2026_fomo,
  title         = {{FoMo: A Multi-Season Dataset for Robot Navigation in For\^et Montmorency}},
  author        = {Matej Boxan and Gabriel Jeanson and Alexander Krawciw
                   and Effie Daum and Xinyuan Qiao and Sven Lilge
                   and Timothy D. Barfoot and Francois Pomerleau},
  year          = {2026},
  eprint        = {2603.08433},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2603.08433}
}
```

## License

The FoMo dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These derived
labels and poses are published under CC BY 4.0 accordingly, with full
attribution to the FoMo authors above. No FoMo sensor data is included. The
conversion code
([apairo_huggingface](https://github.com/apairo-robotics/apairo_huggingface))
is MIT.
