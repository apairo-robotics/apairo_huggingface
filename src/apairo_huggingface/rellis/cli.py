"""``apairo-rellis``: RELLIS-3D -> projected label + traversability masks -> LeRobotDataset.

    apairo-rellis calib-check --root /data/rellis --png-dir /tmp/calib              # verify the calibration first
    apairo-rellis process --root /data/rellis --out ~/data/rellis_derived            # every sequence
    apairo-rellis process --root /data/rellis --out ~/data/rellis_derived 00003      # one sequence
    apairo-rellis agreement --root /data/rellis --out ~/data/rellis_derived          # vs official 2D labels
    apairo-rellis overlay ~/data/rellis_derived/00003 --every 500 --png-dir /tmp/ov
    apairo-rellis export ~/data/rellis_derived 00003 --root /data/rellis --lerobot-root ~/data/rellis_lerobot
    apairo-rellis check ~/data/rellis_lerobot                                        # reload + value sets

``--root`` is the RELLIS-3D tree (``Rellis-3D/``, ``Rellis_3D/``, ``pt_*.lst``,
``Rellis_3D_image_split/``); it is only read. Everything is written under
``--out``: one symlink view per sequence holding the derived ``sem_mask`` and
``trav_mask`` channels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config


def cmd_process(args) -> None:
    from .labeling import label_sequence

    out = Path(args.out)
    for seq in args.sequences or config.SEQUENCES:
        report = label_sequence(args.root, seq, out, overwrite=args.overwrite, limit=args.limit)
        if report:
            (out / seq / "apairo_rellis_report.json").write_text(json.dumps(report, indent=2))


def cmd_calib_check(args) -> None:
    from .calibcheck import calibration_check

    report = calibration_check(args.root, args.sequences or config.SEQUENCES,
                               frames_per_seq=args.frames, png_dir=args.png_dir)
    for seq, scores in report.items():
        for name, value in scores.items():
            print(f"{seq}  {name}: {value if isinstance(value, dict) else round(value, 4)}")


def cmd_agreement(args) -> None:
    from .agreement import SPLITS, agreement, table

    out = Path(args.out)
    views = [out / s for s in (args.sequences or config.SEQUENCES) if (out / s / config.SEM_MASK).is_dir()]
    report = agreement(args.root, views, out_json=out / "agreement.json")
    for split in ("all", *SPLITS):
        if split in report["overall"]:
            print(table(report, split), end="\n\n")
    for seq, r in report["per_sequence"].items():
        print(f"{seq}: pixels={r['pixels']} pixel_accuracy={r['pixel_accuracy']:.3f} "
              f"mIoU={r['miou']:.3f} mIoU_no_sky={r['miou_no_sky']:.3f} fwIoU={r['fwiou']:.3f}")


def cmd_overlay(args) -> None:
    from .overlay import overlay

    view = Path(args.view)
    stems = sorted(p.stem for p in (view / config.SEM_MASK).glob("*.png"))
    for stem in stems[:: args.every]:
        print(overlay(view, stem, Path(args.png_dir) / f"{view.name}_{stem}.png"))


def cmd_export(args) -> None:
    from .export import to_lerobot

    out = Path(args.out)
    to_lerobot(args.root, [out / s for s in args.sequences or config.SEQUENCES], args.repo_id,
               lerobot_root=args.lerobot_root, push=args.push, limit=args.limit)


def cmd_check(args) -> None:
    from .export import check

    for key, value in check(args.repo_id, args.lerobot_root).items():
        print(f"{key}: {value}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="apairo-rellis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="sem_mask + trav_mask for every image")
    p.add_argument("sequences", nargs="*", help=f"default: all of {config.SEQUENCES}")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, help="first N images only (smoke test)")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("calib-check", help="score calibration chains against the 2D labels, write overlays")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--root", required=True)
    p.add_argument("--frames", type=int, default=8, help="annotated frames per sequence")
    p.add_argument("--png-dir", help="write point overlays here")
    p.set_defaults(func=cmd_calib_check)

    p = sub.add_parser("agreement", help="IoU / agreement vs the official 2D annotations")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("overlay", help="PNG overlays of the masks on their images")
    p.add_argument("view", help="a processed sequence view, e.g. <out>/00003")
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--png-dir", required=True)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("export", help="processed sequences -> one LeRobotDataset (one episode each)")
    p.add_argument("out", help="the --out of process")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="apairo-robotics/rellis3d_projected_labels")
    p.add_argument("--lerobot-root", help="local output directory (default: LeRobot's cache)")
    p.add_argument("--limit", type=int, help="first N frames per sequence (smoke test)")
    p.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check", help="reload an exported dataset: features, episodes, value sets")
    p.add_argument("lerobot_root")
    p.add_argument("--repo-id", default="apairo-robotics/rellis3d_projected_labels")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
