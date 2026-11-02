"""How noisy is trajectory traversability? Measured against perfect sim semantics.

Self-supervised traversability labels a point positive when the robot later
drives over it (``trav_traj`` = 1) and leaves everything else UNLABELED (0):
a positive-unlabeled (PU) label. The sim gives each point its true class
(``point_sem``), and ``semantic_classes.json`` says which classes are
traversable (ground, vegetation). On the scored scans (at least
``MIN_FUTURE_PATH`` of path ahead) and the points with a visible label
(``point_sem`` not 255 and not 0 = unlabeled), this module reports:

- ``positive_precision``  P(sem trav | trav_traj = 1): how clean the positives are.
- ``positive_recall``     P(trav_traj = 1 | sem trav): share of the truly
  traversable points the trajectory labels.
- ``unlabeled_positive_rate``  P(sem trav | trav_traj = 0): the share of the
  unlabeled set that is in fact traversable -- the class prior a PU loss such
  as nnPU has to assume for the unlabeled data.
- ``prior``  P(sem trav) over all labeled points.
- per-class counts of positives and unlabeled points.

Each figure is given for every range band (sensor distance) in ``BANDS``, and
for two traversability definitions: ``json`` (ground + vegetation, the json's
flags) and ``ground_only`` (vegetation counted as not traversable).

``sim_cloud_agreement`` checks the label transfer itself against the sim's own
semantic cloud (``points_semantic``): the points of a scan stamped like a
semantic cloud are matched to it by nearest neighbour (< 1 cm), and the two
class ids are compared where both exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .labeling import POINT_SEM, TRAV_TRAJ

BANDS = {"all": (0.0, np.inf), "0-10m": (0.0, 10.0), "10-20m": (10.0, 20.0), "20m+": (20.0, np.inf)}
DEFINITIONS = {"json": config.TRAVERSABLE_IDS, "ground_only": [config.NAME_TO_ID["ground"]]}


def confusion(seq_dir) -> dict[str, np.ndarray]:
    """Per band: (2, 256) counts [trav_traj, point_sem] over the scored scans."""
    seq_dir = Path(seq_dir)
    scored = np.load(seq_dir / "scored_frames.npy")
    cm = {b: np.zeros((2, 256), dtype=np.int64) for b in BANDS}
    for k in scored:
        sem = np.load(seq_dir / POINT_SEM / f"{k:06d}.npy").astype(np.int64)
        trav = np.load(seq_dir / TRAV_TRAJ / f"{k:06d}.npy").astype(np.int64)
        rng = np.linalg.norm(np.load(seq_dir / config.LIDAR / f"{k:06d}.npy"), axis=1)
        for band, (lo, hi) in BANDS.items():
            sel = (rng >= lo) & (rng < hi)
            cm[band] += np.bincount(trav[sel] * 256 + sem[sel], minlength=512).reshape(2, 256)
    return cm


def scores(cm: np.ndarray, trav_ids) -> dict:
    """PU-noise figures from one (2, 256) confusion, on labeled classes only."""
    classes = [c for c in config.CLASSES if c != 0]
    is_trav = np.isin(classes, trav_ids)
    pos = cm[1, classes].astype(float)          # trav_traj = 1
    unl = cm[0, classes].astype(float)          # trav_traj = 0
    tp, fp = pos[is_trav].sum(), pos[~is_trav].sum()
    up, un = unl[is_trav].sum(), unl[~is_trav].sum()
    safe = lambda a, b: float(a / b) if b else float("nan")
    return {
        "points": int(tp + fp + up + un),
        "positives": int(tp + fp),
        "positive_precision": safe(tp, tp + fp),
        "positive_recall": safe(tp, tp + up),
        "unlabeled_positive_rate": safe(up, up + un),
        "prior": safe(tp + up, tp + fp + up + un),
    }


def per_class(cm: np.ndarray) -> dict:
    return {config.CLASSES[c][0]: {"trav_traj_1": int(cm[1, c]), "trav_traj_0": int(cm[0, c])}
            for c in config.CLASSES}


def sim_cloud_agreement(seq_dir, max_dist: float = 0.01) -> dict:
    """Our visible-point labels vs the sim's own semantic cloud, point by point."""
    seq_dir = Path(seq_dir)
    lidar_ts = np.loadtxt(seq_dir / config.LIDAR / "timestamps.txt", ndmin=1)
    cloud_ts = np.loadtxt(seq_dir / config.POINTS_SEMANTIC / "timestamps.txt", ndmin=1)
    cm = np.zeros((256, 256), dtype=np.int64)    # [ours, sim]
    matched = total = clouds = 0
    for j, t in enumerate(cloud_ts):
        k = int(np.argmin(np.abs(lidar_ts - t)))
        if abs(lidar_ts[k] - t) > 1e-4 or not (seq_dir / POINT_SEM / f"{k:06d}.npy").exists():
            continue
        cloud = np.load(seq_dir / config.POINTS_SEMANTIC / f"{j:06d}.npy")
        pts = np.load(seq_dir / config.LIDAR / f"{k:06d}.npy")
        ours = np.load(seq_dir / POINT_SEM / f"{k:06d}.npy")
        d, i = cKDTree(pts).query(cloud[:, :3], k=1)
        ok = d < max_dist
        clouds += 1
        total += len(cloud)
        matched += int(ok.sum())
        cm += np.bincount(ours[i[ok]].astype(np.int64) * 256 + cloud[ok, 3].astype(np.int64),
                          minlength=256 * 256).reshape(256, 256)
    both = [c for c in config.CLASSES]
    sub = cm[np.ix_(both, both)]
    return {
        "clouds_compared": clouds,
        "cloud_points": int(total),
        "matched_points": int(matched),
        "both_labeled": int(sub.sum()),
        "agreement_both_labeled": float(np.trace(sub) / sub.sum()) if sub.sum() else float("nan"),
        "ours_labeled_sim_not": int(cm[both, config.NO_DATA].sum()),
        "sim_labeled_ours_not": int(cm[config.NO_DATA, both].sum()),
        "confusion_ours_rows_sim_cols": {config.CLASSES[a][0]: {config.CLASSES[b][0]: int(cm[a, b]) for b in both
                                                               if cm[a, b]} for a in both},
    }


def agreement(seq_dir, out_json=None) -> dict:
    seq_dir = Path(seq_dir)
    cm = confusion(seq_dir)
    report = {
        "sequence": seq_dir.name,
        "scored_scans": int(len(np.load(seq_dir / "scored_frames.npy"))),
        "bands": {band: {name: scores(cm[band], ids) for name, ids in DEFINITIONS.items()} for band in BANDS},
        "per_class_all": per_class(cm["all"]),
        "sim_cloud": sim_cloud_agreement(seq_dir),
    }
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
    return report


def table(report: dict) -> str:
    lines = [f"{'band':8s} {'definition':12s} {'points':>10s} {'pos':>9s} {'pos_prec':>9s} "
             f"{'pos_rec':>8s} {'unl_pos':>8s} {'prior':>6s}"]
    for band, defs in report["bands"].items():
        for name, s in defs.items():
            lines.append(f"{band:8s} {name:12s} {s['points']:10d} {s['positives']:9d} {s['positive_precision']:9.3f} "
                         f"{s['positive_recall']:8.3f} {s['unlabeled_positive_rate']:8.3f} {s['prior']:6.3f}")
    return "\n".join(lines)
