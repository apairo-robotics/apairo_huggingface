# apairo-huggingface

Export [apairo](https://github.com/apairo-robotics/apairo) robotics datasets to
the HuggingFace **LeRobotDataset** format. The first worked bridge: raw
**TartanDrive v2** offroad-driving recordings → a LeRobotDataset ready for policy
training on the Hub.

## Why it's small

TartanDrive is an asynchronous, multi-rate sensor rig — camera, GPS/odometry and
drive commands each tick on their own clock, one channel per file. LeRobot wants
the opposite: synchronous, per-episode frames with every modality aligned.

apairo closes that gap on the read side, so **the only code this package adds is
the channel → feature mapping**:

| LeRobot feature | ← TartanDrive channel | mapping |
|---|---|---|
| `observation.images.front` | `image_left_color` (H×W×3 uint8) | as-is |
| `observation.state` | `gps_odom` | pose `[x, y, z, qx, qy, qz, qw]` |
| `action` | `cmd` | twist `[linear_velocity, angular_velocity]` |

## Layout

```
src/apairo_huggingface/
  config.py      channels, sync params, task, feature dims
  mappings.py    image_front / odom_to_state / twist_to_action  (the domain code)
  features.py    build_features — the LeRobot schema, sized from a real frame
  export.py      to_lerobot() — the reusable sink; validate() — the dry-run
  cli.py         apairo-tartan-export
examples/
  export_tartan_to_lerobot.py   thin, linear entry point
tests/
  conftest.py    synthetic TartanDrive fixture
  test_export.py dry-run + export + reload round-trip
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

## Use

```bash
# Validate the apairo read pipeline only — no lerobot, no writes.
apairo-tartan-export --dry-run --root /path/to/tartan

# Build a LeRobotDataset locally, or push to the Hub (needs `huggingface-cli login`).
apairo-tartan-export --no-push --root /path/to/tartan
apairo-tartan-export --repo-id <user>/tartandrive_offroad --root /path/to/tartan
```

Or as a library:

```python
from apairo_huggingface import load_tartan, to_lerobot

dataset = to_lerobot(load_tartan("/path/to/tartan"), "me/tartandrive_offroad", push=True)
```

`--root` accepts a TartanDrive **root** (many recordings → many episodes) or a
**single** sequence directory (→ one episode); it also reads `APAIRO_TARTAN_ROOT`.

## Test

```bash
pytest        # builds a synthetic TartanDrive dataset, exports it, reloads it
```

## Notes

- **Data**: TartanDrive v2 is assumed already on disk (no downloader). The apairo
  fixture `apairo/test/assets/mini_tartan` ships only `velodyne_0`/`cmd`/`imu`, so
  the full camera+state+action mapping needs a complete sequence — hence the
  synthetic fixture in `tests/`.
- **Verified** against `lerobot 0.5.1` (LeRobotDataset v3.0) and `apairo 0.4.0`:
  `create → add_frame({..., "task"}) → save_episode → finalize → push_to_hub`.
  In v3, `task` is a key of the frame dict and `finalize()` is required before the
  dataset is valid.
