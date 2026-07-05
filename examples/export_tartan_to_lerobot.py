"""Export a TartanDrive recording to a LeRobotDataset — end to end.

Read it top to bottom: load the raw async sensor rig, hand it to the reusable
sink, done. Everything hard is inside the library:

  - loading, resampling and episode segmentation  -> apairo
  - channel -> feature mapping                     -> apairo_huggingface/mappings.py
  - the load -> align -> copy -> finalize pipeline  -> apairo_huggingface/export.py

Run:
    python examples/export_tartan_to_lerobot.py         # build locally
    APAIRO_TARTAN_ROOT=/data/tartan python examples/export_tartan_to_lerobot.py
"""

import os

from apairo_huggingface import load_tartan, to_lerobot

ROOT = os.environ.get("APAIRO_TARTAN_ROOT", "/data/tartan")
REPO_ID = "tartandrive_offroad"

tartan = load_tartan(ROOT)                          # async multi-rate TartanDrive rig
dataset = to_lerobot(tartan, REPO_ID, push=False)   # aligned, per-episode LeRobot frames

print(f"{dataset.num_episodes} episodes / {dataset.num_frames} frames -> {REPO_ID}")
