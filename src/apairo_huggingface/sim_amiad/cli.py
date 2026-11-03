"""``apairo-sim-amiad``: Isaac Sim mcap -> apairo sequence -> sim labels -> LeRobotDataset.

    apairo-sim-amiad extract --out ~/data/sim_amiad/seg_t035_t065 --t-stop 65   # a segment (sim seconds)
    apairo-sim-amiad extract --out ~/data/sim_amiad/full                        # the whole bag
    apairo-sim-amiad label ~/data/sim_amiad/full                                # point_sem, trav_traj, trav_mask
    apairo-sim-amiad agreement ~/data/sim_amiad/full                            # trajectory vs semantic trav
    apairo-sim-amiad overlay ~/data/sim_amiad/full --every 1000 --png-dir /tmp/ov
    apairo-sim-amiad export ~/data/sim_amiad/full --lerobot-root ~/data/sim_amiad/lerobot
    apairo-sim-amiad check ~/data/sim_amiad/lerobot                             # reload + value sets

The bag is only read. There is no Hub upload: this is the lab's own data.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config

REPO_ID = "apairo-robotics/sim_amiad_offroad"


def cmd_extract(args) -> None:
    from .extract import extract

    report = extract(args.bag, args.out, t_start=args.t_start, t_stop=args.t_stop, classes_json=args.classes)
    print(json.dumps({k: report[k] for k in ("frames", "series", "sim_time_range", "seconds")}, indent=2))


def cmd_label(args) -> None:
    from .labeling import label_sequence

    report = label_sequence(args.seq_dir, overwrite=args.overwrite, limit=args.limit)
    if report:
        print(json.dumps(report, indent=2))


def cmd_agreement(args) -> None:
    from .agreement import agreement, table

    seq_dir = Path(args.seq_dir)
    report = agreement(seq_dir, out_json=seq_dir / "agreement.json")
    print(table(report))
    s = report["sim_cloud"]
    print(f"\nlabel transfer vs the sim's semantic cloud: {s['both_labeled']} points with a class on both "
          f"sides, agreement {s['agreement']:.4f} (with unlabeled as a class: {s['both_labeled_incl_unlabeled']} "
          f"points, {s['agreement_incl_unlabeled']:.4f})")


def cmd_overlay(args) -> None:
    import numpy as np

    from .overlay import overlay

    seq_dir = Path(args.seq_dir)
    for k in np.load(seq_dir / "scored_frames.npy")[:: args.every]:
        print(overlay(seq_dir, int(k), Path(args.png_dir) / f"{seq_dir.name}_{int(k):06d}.png"))


def cmd_export(args) -> None:
    from .export import to_lerobot

    to_lerobot(args.seq_dirs, args.repo_id, lerobot_root=args.lerobot_root, limit=args.limit)


def cmd_check(args) -> None:
    from .export import check

    for key, value in check(args.repo_id, args.lerobot_root).items():
        print(f"{key}: {value}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="apairo-sim-amiad", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="stream the mcap into an apairo sequence directory")
    p.add_argument("--bag", default=config.BAG)
    p.add_argument("--classes", default=config.CLASSES_JSON, help="semantic_classes.json")
    p.add_argument("--out", required=True)
    p.add_argument("--t-start", type=float, help="sim seconds (header clock)")
    p.add_argument("--t-stop", type=float, help="sim seconds (header clock)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("label", help="point_sem + trav_traj per scan, trav_mask per image")
    p.add_argument("seq_dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, help="first N scans only (smoke test)")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("agreement", help="trajectory traversability vs sim semantic traversability")
    p.add_argument("seq_dir")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("overlay", help="PNG grids of projection and labels")
    p.add_argument("seq_dir")
    p.add_argument("--every", type=int, default=1000)
    p.add_argument("--png-dir", required=True)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("export", help="labelled sequences -> one local LeRobotDataset (one episode each)")
    p.add_argument("seq_dirs", nargs="+")
    p.add_argument("--lerobot-root", required=True, help="local output directory")
    p.add_argument("--repo-id", default=REPO_ID, help="dataset name only; nothing is uploaded")
    p.add_argument("--limit", type=int, help="first N frames per sequence (smoke test)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check", help="reload an exported dataset: features, episodes, value sets")
    p.add_argument("lerobot_root")
    p.add_argument("--repo-id", default=REPO_ID)
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
