"""Agreement of the trajectory trav grid with the lidar-label semantic grid.

Two independent sources meet in the BEV grids: ``bev_trav`` comes from the
poses alone, ``bev_semantic`` from the human point labels. Over every frame
with a trav grid, counting cells:

- ``driven_driveable``   share of driven cells (with a known class) whose
                         majority class is driveable (``config.DRIVEABLE``:
                         dirt, grass, asphalt, concrete, puddle, mud);
- ``driven_obstacle``    share of driven cells whose majority class is an
                         obstacle (``config.OBSTACLE``: tree, bush, barrier,
                         person, ...) -- the corridor running into things;
- ``driven_observed``    share of driven cells that hold any point at all;
- ``seen_driveable``     share of seen-not-driven cells that are driveable:
                         the base rate the corridor has to beat;
- ``driven_class``       the class histogram of the driven cells.

Driven cells are counted out to the full grid extent; the footprint ahead
of the robot is where pose drift and the accumulation's sparsity weigh most,
so ``near`` repeats the first two numbers within ``NEAR`` metres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from apairo import RawDataset

from . import config
from .bev import cell_centres, declaration, grid_shape

NEAR = 15.0   # m from the lidar


def sequence_counts(view, every: int = 1) -> dict:
    view = Path(view)
    ds = RawDataset(view, declare=declaration(), keys=[config.BEV_SEMANTIC, config.BEV_TRAV])
    sem_l, trav_l = ds.loaders[config.BEV_SEMANTIC], ds.loaders[config.BEV_TRAV]
    near = (np.linalg.norm(cell_centres(), axis=1) <= NEAR).reshape(grid_shape())
    driveable = np.isin(np.arange(256), config.DRIVEABLE)
    obstacle = np.isin(np.arange(256), config.OBSTACLE)
    hist_driven = np.zeros(256, dtype=np.int64)
    hist_seen = np.zeros(256, dtype=np.int64)
    hist_near = np.zeros(256, dtype=np.int64)
    driven_cells = frames = 0
    for k in range(0, len(trav_l), every):
        trav = np.asarray(trav_l[k])
        if (trav == config.NO_DATA).all():
            continue
        sem = np.asarray(sem_l[k])
        frames += 1
        driven = trav == 1
        driven_cells += int(driven.sum())
        hist_driven += np.bincount(sem[driven], minlength=256)
        hist_near += np.bincount(sem[driven & near], minlength=256)
        hist_seen += np.bincount(sem[trav == 0], minlength=256)

    def share(hist, mask):
        known = hist[:255].sum()
        return float(hist[mask].sum() / known) if known else float("nan")

    return {
        "sequence": view.name,
        "frames": frames,
        "driven_cells": driven_cells,
        "driven_observed": float(hist_driven[:255].sum() / driven_cells) if driven_cells else float("nan"),
        "driven_driveable": share(hist_driven, driveable),
        "driven_obstacle": share(hist_driven, obstacle),
        "near_driven_driveable": share(hist_near, driveable),
        "near_driven_obstacle": share(hist_near, obstacle),
        "seen_driveable": share(hist_seen, driveable),
        "driven_class": {config.ONTOLOGY[i][0]: int(hist_driven[i])
                         for i in np.flatnonzero(hist_driven[:255]) if i in config.ONTOLOGY},
    }


def table(reports: list[dict]) -> str:
    head = (f"{'seq':6} {'frames':>6} {'driven_obs':>10} {'drv_driveable':>13} {'drv_obstacle':>12} "
            f"{'near_drv':>8} {'near_obs':>8} {'seen_driveable':>14}")
    rows = [head]
    for r in reports:
        rows.append(f"{r['sequence']:6} {r['frames']:6d} {r['driven_observed']:10.3f} {r['driven_driveable']:13.3f} "
                    f"{r['driven_obstacle']:12.3f} {r['near_driven_driveable']:8.3f} "
                    f"{r['near_driven_obstacle']:8.3f} {r['seen_driveable']:14.3f}")
    return "\n".join(rows)
