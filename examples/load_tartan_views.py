"""One raw TartanDrive dataset, many views.

The LeRobot export in this package is a single slice of the raw recordings:
the camera clock with pose and command attached. This example shows the other
slices the same apairo dataset yields by changing ``keys=`` and ``reference=``
-- no new loader code. Channel names come from ``.apairo/channels.yaml``;
adjust to what your extraction holds.

Run against any TartanDrive root, e.g. apairo's mini fixture:

    python examples/load_tartan_views.py /path/to/apairo/test/assets/mini_tartan
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from apairo import Sample, TartanKittiDataset

root = sys.argv[1]


def first_sequence(keys: list[str]):
    return list(TartanKittiDataset(root, keys=keys).sequences)[0]


# --- Perception / terrain mapping: lidar clock ------------------------------ #
# Point clouds are the reference; each cloud carries the command nearest in time.
lidar = first_sequence(["velodyne_0", "cmd"]).synchronize(
    reference="velodyne_0", method="nearest", tolerance=0.1
)
print(f"lidar view: {len(lidar)} frames, cloud {lidar[0].data['velodyne_0'].shape}")

# --- Vehicle dynamics: IMU clock (~10x the camera rate) --------------------- #
# No images to decode -- fast iteration for dynamics / traversability models.
dyn = first_sequence(["multisense_imu", "cmd"]).synchronize(
    reference="multisense_imu", method="nearest", tolerance=0.02
)
print(f"dynamics view: {len(dyn)} frames at IMU rate")


# --- Action chunking / world models: causal windows over any view ----------- #
def stack(samples: list[Sample]) -> Sample:
    return Sample(
        data={k: np.stack([s.data[k] for s in samples]) for k in samples[0].data},
        timestamp=samples[-1].timestamp,
    )


chunks = dyn.window(size=8, reduce=stack, boundary="drop")
print(f"window view: {len(chunks)} chunks, cmd {chunks[0].data['cmd'].shape}")


# --- Training: every view is already a torch-compatible Dataset ------------- #
def collate(batch: list[Sample]) -> dict[str, torch.Tensor]:
    return {
        k: torch.stack([torch.as_tensor(np.ascontiguousarray(s.data[k])) for s in batch])
        for k in batch[0].data
    }


loader = torch.utils.data.DataLoader(dyn, batch_size=32, collate_fn=collate)
batch = next(iter(loader))
print("dataloader batch:", {k: tuple(v.shape) for k, v in batch.items()})
