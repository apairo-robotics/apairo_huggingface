"""Agreement of the projected lidar labels with the official 2D annotations.

On every image that has an official annotation (``pylon_camera_node_label_id``),
the projected ``sem_mask`` is compared pixel by pixel, **only where the
projection has data** (``sem_mask != 255``) and where neither side is ``void``
(0). Per class c, with P the projected and G the 2D label:

- ``iou``        |P=c and G=c| / |P=c or G=c|
- ``agreement``  |P=c and G=c| / |G=c|   (share of the 2D class the lidar agrees with)
- ``precision``  |P=c and G=c| / |P=c|

Counts are accumulated over all pixels of all images (not averaged per image).
Results are given for every annotated image and per official image split
(``Rellis_3D_image_split/{train,val,test}.lst``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

SPLITS = ("train", "val", "test")


def image_splits(root) -> dict[str, str]:
    """``"<seq>/<stem>" -> split`` from the official image split lists."""
    out = {}
    for split in SPLITS:
        for line in (Path(root) / "Rellis_3D_image_split" / f"{split}.lst").read_text().split("\n"):
            if line.strip():
                img = line.split()[0]                  # 00000/pylon_camera_node/frame...jpg
                seq, _, name = img.split("/")
                out[f"{seq}/{Path(name).stem}"] = split
    return out


def sequence_confusion(view, splits: dict[str, str]) -> dict[str, np.ndarray]:
    """256x256 counts [projected, 2D] over the annotated images of one view."""
    view = Path(view)
    seq = view.name
    cm = {s: np.zeros((256, 256), dtype=np.int64) for s in ("all", *SPLITS, "unsplit")}
    for ref_path in sorted((view / config.LABEL_2D).iterdir()):
        stem = ref_path.stem
        sem_path = view / config.SEM_MASK / f"{stem}.png"
        if not sem_path.exists():
            continue
        sem = np.asarray(Image.open(sem_path))
        ref = np.asarray(Image.open(ref_path))
        keep = (sem != config.NO_DATA) & (sem != 0) & (ref != 0)
        counts = np.bincount(sem[keep].astype(np.int64) * 256 + ref[keep], minlength=256 * 256)
        counts = counts.reshape(256, 256)
        cm["all"] += counts
        cm[splits.get(f"{seq}/{stem}", "unsplit")] += counts
    return cm


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    pred, gt = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    per_class = {}
    for cid, (name, _) in config.ONTOLOGY.items():
        if cid == 0 or (pred[cid] == 0 and gt[cid] == 0):
            continue
        union = pred[cid] + gt[cid] - tp[cid]
        per_class[name] = {
            "iou": tp[cid] / union if union else 0.0,
            "agreement": tp[cid] / gt[cid] if gt[cid] else float("nan"),
            "precision": tp[cid] / pred[cid] if pred[cid] else float("nan"),
            "pixels_2d": int(gt[cid]),
            "pixels_projected": int(pred[cid]),
        }
    total = cm.sum()
    ious = [v["iou"] for v in per_class.values() if v["pixels_2d"] > 0]
    return {
        "pixels": int(total),
        "pixel_accuracy": float(tp.sum() / total) if total else float("nan"),
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "per_class": per_class,
    }


def agreement(root, views, out_json=None) -> dict:
    splits = image_splits(root)
    total = {s: np.zeros((256, 256), dtype=np.int64) for s in ("all", *SPLITS, "unsplit")}
    per_seq = {}
    for view in views:
        cm = sequence_confusion(view, splits)
        per_seq[Path(view).name] = scores(cm["all"])
        for k in total:
            total[k] += cm[k]
    report = {
        "overall": {k: scores(v) for k, v in total.items() if v.sum()},
        "per_sequence": {k: {kk: vv for kk, vv in v.items() if kk != "per_class"} for k, v in per_seq.items()},
    }
    if out_json:
        Path(out_json).write_text(json.dumps(report, indent=2))
    return report


def table(report: dict, split: str = "all") -> str:
    r = report["overall"][split]
    lines = [f"split={split}  pixels={r['pixels']}  pixel_accuracy={r['pixel_accuracy']:.3f}  mIoU={r['miou']:.3f}",
             f"{'class':10s} {'IoU':>6s} {'agree':>6s} {'prec':>6s} {'px_2d':>11s} {'px_proj':>11s}"]
    for name, v in sorted(r["per_class"].items(), key=lambda kv: -kv[1]["pixels_2d"]):
        lines.append(f"{name:10s} {v['iou']:6.3f} {v['agreement']:6.3f} {v['precision']:6.3f} "
                     f"{v['pixels_2d']:11d} {v['pixels_projected']:11d}")
    return "\n".join(lines)
