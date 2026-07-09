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
  - autonomous-driving
  - apairo
pretty_name: TartanDrive Off-Road (LeRobot format)
---

# TartanDrive Off-Road — LeRobotDataset

> **This dataset is a demonstration of a data pipeline.** All 54 episodes /
> ~119k frames were generated from raw, *asynchronous* TartanDrive 2.0 rosbags,
> and the only code specific to this conversion is **~17 lines of
> channel→feature mapping**. Everything hard — loading multi-rate sensors,
> time-aligning them, cutting episodes — is done by
> [**apairo**](https://github.com/apairo-robotics/apairo). The interesting part
> isn't the data, it's *how little, and how clearly, it takes to get there.*

## The entire conversion

TartanDrive is an **asynchronous, multi-rate sensor rig**: camera, GPS odometry
and drive commands each tick on their own clock, one channel per file. LeRobot
wants the opposite — synchronous, per-episode frames with every modality
aligned. That gap is normally where a pile of brittle glue code lives.

apairo closes it on the read side, and the per-frame value ops are **standard
[`apairo_transform`](https://github.com/apairo-robotics/apairo_transform)
callables** — cast the command, take the pose out of the odometry row, leave the
image untouched — chained onto the synchronized view, so there is **no bespoke
domain code left**:

```python
import numpy as np
from apairo import TartanKittiDataset, Compose
from apairo_transform import CastTo, ChannelSelect   # library transforms

frames = (TartanKittiDataset(root, keys=["image_left_color", "gps_odom", "cmd"])
          .synchronize(reference="image_left_color",       # multi-rate → camera clock
                       method="nearest", tolerance=0.05)
          .transform("gps_odom", Compose([ChannelSelect(range(7)),  # pose out of Odometry
                                          CastTo(np.float32)]))
          .transform("cmd", CastTo(np.float32)))            # image_left_color: as-is
```

The whole export is then *load → align → copy → finalize*, because apairo
already did the loading, resampling and episode segmentation, and the frame is
pure naming:

```python
for recording in tartan.sequences:              # one recording = one episode
    for sample in aligned_frames(recording):    # the synchronized+transformed view above
        dataset.add_frame({
            "observation.images.front": sample.data["image_left_color"],
            "observation.state":        sample.data["gps_odom"],
            "action":                   sample.data["cmd"],
            "task": "drive off-road",
        })
    dataset.save_episode()
dataset.finalize()
```

No timestamp-matching loop, no per-sensor resampling, no episode bookkeeping,
and no custom transform code — `synchronize()` resamples every channel onto the
camera clock (nearest within ±50 ms) and the value ops are library callables.
Full source: [`mappings.py`](https://github.com/apairo-robotics/apairo_huggingface/blob/main/src/apairo_huggingface/mappings.py).

**Full source, all open:**
[apairo_huggingface](https://github.com/apairo-robotics/apairo_huggingface)
(the exporter above) ·
[apairo](https://github.com/apairo-robotics/apairo) (the data layer) ·
[apairo_extractor](https://github.com/apairo-robotics/apairo_extractor)
(rosbags → `.apairo` layout). Reproduce with:

```bash
apairo-tartan-export --repo-id <you>/tartandrive_offroad --root /path/to/tartan
```

## What's in it

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("apairo-robotics/tartandrive_offroad")
frame = dataset[0]
frame["observation.images.front"]  # (H, W, 3) front camera
frame["observation.state"]         # [x, y, z, qx, qy, qz, qw] pose
frame["action"]                    # [linear_velocity, angular_velocity]
```

| Feature | Shape | Source (TartanDrive channel) |
|---|---|---|
| `observation.images.front` | (H, W, 3) uint8 video | `image_left_color` — left colour camera |
| `observation.state` | (7,) float32 | `gps_odom` — pose `[x, y, z, qx, qy, qz, qw]` |
| `action` | (2,) float32 | `cmd` — `[linear_velocity, angular_velocity]` |

- **fps**: 10 (camera clock) · **robot_type**: `tartandrive` (Yamaha Viking ATV)
  · **task**: `drive off-road`
- 54 recordings → 54 episodes, ~119k synchronized frames.

## One raw dataset, many views

This LeRobotDataset is **one** slice of the raw recordings — the camera clock,
with pose and command attached. The same apairo dataset re-slices into other
views by changing `keys=` and `reference=`, again with no new loader code:

```python
from apairo import TartanKittiDataset

# Perception / terrain mapping: lidar clock, point clouds + nearest command
lidar = (TartanKittiDataset(root, keys=["velodyne_0", "cmd"])
         .synchronize(reference="velodyne_0", method="nearest", tolerance=0.1))
lidar[0].data["velodyne_0"]         # (N, 3) point cloud

# Vehicle dynamics: IMU clock (~10x the camera rate), no images to decode
dyn = (TartanKittiDataset(root, keys=["multisense_imu", "cmd"])
       .synchronize(reference="multisense_imu", method="nearest", tolerance=0.02))

# Action chunking / world models: causal windows over any view
chunks = dyn.window(size=8, reduce=stack, boundary="drop")
chunks[0].data["cmd"]               # (8, 2) command sequence

# Training: every view is already a torch-compatible Dataset
loader = torch.utils.data.DataLoader(dyn, batch_size=32, collate_fn=collate)
```

Each view is lazy (indices at construction, reads on access) and slots straight
into a `DataLoader` — or back into the exporter to publish as another
LeRobotDataset like this one. The single-slice-here / many-slices-there gap is
exactly the point: LeRobot fixes one schema, apairo yields as many as there are
uses.

## Source data

TartanDrive 2.0 was collected at Carnegie Mellon University's AirLab with a
modified Yamaha Viking ATV driven aggressively across diverse off-road terrain.
This export uses the **rosbag** distribution (not the pre-converted KITTI one),
keeping `/tf`, static calibration and per-channel `frame_id` intact.

## Citation

If you use this dataset, please cite TartanDrive 2.0:

```bibtex
@inproceedings{sivaprakasam2024tartandrive,
  title={TartanDrive 2.0: More Modalities and Better Infrastructure to Further Self-Supervised Learning Research in Off-Road Driving Tasks},
  author={Sivaprakasam, Matthew and Maheshwari, Parv and Castro, Mateo Guaman and Triest, Samuel and Nye, Micah and Willits, Steve and Saba, Andrew and Wang, Wenshan and Scherer, Sebastian},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

## License

The original [TartanDrive dataset](https://theairlab.org/tartandrive-dataset/)
is explicitly released under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). TartanDrive 2.0 does
not state a separate data license (its repository's MIT license covers the
code); this conversion is published under CC BY 4.0 accordingly, with full
attribution to the AirLab authors above. The conversion code
([apairo_huggingface](https://github.com/apairo-robotics/apairo_huggingface))
is MIT.
