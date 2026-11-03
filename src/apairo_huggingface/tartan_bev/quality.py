"""Does the driven corridor sit on flat ground? Coverage of the trav grid.

TartanDrive has no point labels, so the RELLIS check (driven cells vs their
majority class) becomes a geometric one. Two independent sources meet in the
grids: ``bev_trav`` comes from the poses alone, ``bev_geometry`` from the
past scans alone. Over every frame with a trav grid, counting cells:

- ``driven_observed``  share of driven cells holding a point;
- ``driven_flat``      share of observed driven cells that are flat
                       (roughness < ``FLAT_ROUGHNESS`` and step < ``FLAT_STEP``):
                       the driveable proxy;
- ``driven_low_rough`` / ``driven_low_step``  each criterion alone;
- ``seen_flat``        the flat share of seen-not-driven cells: the base rate
                       the corridor has to beat;
- ``near_*``           the same within ``NEAR`` metres, where pose drift and
                       sparsity weigh least;
- ``coverage``         driven / seen / unknown share of all cells, frames
                       without a trav grid (end of a recording) excluded, and
                       ``frames_without_trav``.

Roughness is only defined with ``MIN_CELL_POINTS`` points, so a cell counts as
flat only if both layers are known.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..bev_grid import NO_DATA, cell_centres, grid_shape
from . import config
from .bev import open_view

NEAR = 15.0   # m from the lidar
_SUMS = ("cells", "driven", "seen", "unknown", "driven_observed", "driven_known", "driven_flat",
         "driven_low_rough", "driven_low_step", "seen_known", "seen_flat",
         "near_driven_known", "near_driven_flat", "near_seen_known", "near_seen_flat")


def sequence_counts(view, every: int = 1) -> dict:
    view = Path(view)
    ds = open_view(view, [config.BEV_GEOMETRY, config.BEV_TRAV])
    geom_l, trav_l = ds.loaders[config.BEV_GEOMETRY], ds.loaders[config.BEV_TRAV]
    near = (np.linalg.norm(cell_centres(), axis=1) <= NEAR).reshape(grid_shape())
    n = dict.fromkeys(_SUMS, 0)
    frames = no_trav = 0
    for k in range(0, len(trav_l), every):
        trav = np.asarray(trav_l[k])
        if (trav == NO_DATA).all():
            no_trav += 1
            continue
        geom = np.asarray(geom_l[k], dtype=np.float32)
        rough, step = geom[0], geom[1]
        frames += 1
        driven, seen = trav == 1, trav == 0
        known = np.isfinite(rough) & np.isfinite(step)
        flat = known & (rough < config.FLAT_ROUGHNESS) & (step < config.FLAT_STEP)
        observed = np.isfinite(step)
        n["cells"] += trav.size
        n["driven"] += int(driven.sum())
        n["seen"] += int(seen.sum())
        n["unknown"] += int((trav == NO_DATA).sum())
        n["driven_observed"] += int((driven & observed).sum())
        n["driven_known"] += int((driven & known).sum())
        n["driven_flat"] += int((driven & flat).sum())
        n["driven_low_rough"] += int((driven & known & (rough < config.FLAT_ROUGHNESS)).sum())
        n["driven_low_step"] += int((driven & known & (step < config.FLAT_STEP)).sum())
        n["seen_known"] += int((seen & known).sum())
        n["seen_flat"] += int((seen & flat).sum())
        n["near_driven_known"] += int((driven & known & near).sum())
        n["near_driven_flat"] += int((driven & flat & near).sum())
        n["near_seen_known"] += int((seen & known & near).sum())
        n["near_seen_flat"] += int((seen & flat & near).sum())
    return shares({"sequence": view.name, "frames": frames, "frames_without_trav": no_trav, **n})


def shares(r: dict) -> dict:
    def ratio(a, b):
        return float(r[a] / r[b]) if r[b] else float("nan")

    r.update({
        "coverage": {"driven": ratio("driven", "cells"), "seen": ratio("seen", "cells"),
                     "unknown": ratio("unknown", "cells")},
        "driven_observed_share": ratio("driven_observed", "driven"),
        "driven_flat_share": ratio("driven_flat", "driven_known"),
        "driven_low_rough_share": ratio("driven_low_rough", "driven_known"),
        "driven_low_step_share": ratio("driven_low_step", "driven_known"),
        "seen_flat_share": ratio("seen_flat", "seen_known"),
        "near_driven_flat_share": ratio("near_driven_flat", "near_driven_known"),
        "near_seen_flat_share": ratio("near_seen_flat", "near_seen_known"),
    })
    return r


def total(reports: list[dict]) -> dict:
    """All recordings pooled (cell counts summed, shares recomputed)."""
    r = {"sequence": "all", "frames": sum(x["frames"] for x in reports),
         "frames_without_trav": sum(x["frames_without_trav"] for x in reports)}
    r.update({k: sum(x[k] for x in reports) for k in _SUMS})
    return shares(r)


def table(reports: list[dict]) -> str:
    head = (f"{'recording':44} {'frames':>6} {'driven':>6} {'seen':>5} {'unkn':>5} {'drv_obs':>7} "
            f"{'drv_flat':>8} {'lo_rough':>8} {'lo_step':>7} {'near_flat':>9} {'seen_flat':>9} {'near_seen':>9}")
    rows = [head]
    for r in reports:
        c = r["coverage"]
        rows.append(f"{r['sequence']:44} {r['frames']:6d} {c['driven']:6.3f} {c['seen']:5.3f} {c['unknown']:5.3f} "
                    f"{r['driven_observed_share']:7.3f} {r['driven_flat_share']:8.3f} "
                    f"{r['driven_low_rough_share']:8.3f} {r['driven_low_step_share']:7.3f} "
                    f"{r['near_driven_flat_share']:9.3f} {r['seen_flat_share']:9.3f} {r['near_seen_flat_share']:9.3f}")
    return "\n".join(rows)
