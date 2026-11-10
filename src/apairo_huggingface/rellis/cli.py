"""``apairo-rellis``: RELLIS-3D -> projected label + traversability masks -> LeRobotDataset.

    apairo-rellis calib-check --root /data/rellis --png-dir /tmp/calib              # verify the calibration first
    apairo-rellis process --root /data/rellis --out ~/data/rellis_derived            # every sequence
    apairo-rellis process --root /data/rellis --out ~/data/rellis_derived 00003      # one sequence
    apairo-rellis coverage --out ~/data/rellis_derived                               # the card's coverage table
    apairo-rellis agreement --root /data/rellis --out ~/data/rellis_derived          # vs official 2D labels
    apairo-rellis overlay ~/data/rellis_derived/00003 --every 500 --png-dir /tmp/ov
    apairo-rellis export ~/data/rellis_derived 00003 --root /data/rellis --lerobot-root ~/data/rellis_lerobot
    apairo-rellis check ~/data/rellis_lerobot                                        # reload + value sets

BEV grids (one robot-centred elevation / semantic / trav grid per scan, :mod:`.bev`):

    apairo-rellis bev-process --root /data/rellis --out ~/data/rellis_bev            # every sequence
    apairo-rellis bev-render ~/data/rellis_bev/00003 --every 500 --png-dir /tmp/bev
    apairo-rellis bev-quality --out ~/data/rellis_bev                                # trav vs semantic
    apairo-rellis bev-export ~/data/rellis_bev --root /data/rellis --lerobot-root ~/data/rellis_bev_lerobot
    apairo-rellis bev-check ~/data/rellis_bev_lerobot

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


def cmd_coverage(args) -> None:
    from .labeling import coverage

    out = Path(args.out)
    print("| seq | images | sem_mask pixels with data | frames with a trav mask "
          "| trav pixels with data | driven pixels |")
    print("|---|---|---|---|---|---|")
    for seq in args.sequences or config.SEQUENCES:
        if not (out / seq / config.TRAV_MASK).is_dir():
            continue
        c = coverage(out / seq)
        report_path = out / seq / "apairo_rellis_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"sequence": seq}
        report_path.write_text(json.dumps({**report, **c}, indent=2))
        print(f"| {seq} | {c['images']} | {100 * c['sem_mask_pixel_coverage_mean']:.1f} % "
              f"| {c['trav_mask_frames_with_data']} | {100 * c['trav_mask_pixel_coverage_mean']:.1f} % "
              f"| {100 * c['trav_mask_driven_fraction_mean']:.1f} % |")


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


def cmd_bev_process(args) -> None:
    from .bev import process_sequence

    out = Path(args.out)
    for seq in args.sequences or config.SEQUENCES:
        report = process_sequence(args.root, seq, out, overwrite=args.overwrite,
                                  accumulate=args.accumulate, stride=args.stride)
        (out / seq / "apairo_rellis_bev_report.json").write_text(json.dumps(report, indent=2))


def cmd_bev_render(args) -> None:
    from apairo import RawDataset

    from .bev import declaration
    from .bev_render import render_view

    view = Path(args.view)
    n = len(RawDataset(view, declare=declaration(), keys=[config.BEV_TRAV]).timestamps[config.BEV_TRAV])
    indices = args.frames or list(range(0, n, args.every))
    for path in render_view(view, indices, args.png_dir):
        print(path)


def cmd_bev_quality(args) -> None:
    from .bev_quality import sequence_counts, table

    out = Path(args.out)
    reports = [sequence_counts(out / s, every=args.every) for s in (args.sequences or config.SEQUENCES)
               if (out / s / config.BEV_TRAV).is_dir()]
    (out / "bev_quality.json").write_text(json.dumps(reports, indent=2))
    print(table(reports))


def cmd_bev_export(args) -> None:
    from .bev_export import to_lerobot

    out = Path(args.out)
    to_lerobot(args.root, [out / s for s in args.sequences or config.SEQUENCES], args.repo_id,
               lerobot_root=args.lerobot_root, limit=args.limit)


def cmd_bev_check(args) -> None:
    from .bev_export import check

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
    p.add_argument("--frames", type=int, default=40, help="annotated frames per sequence")
    p.add_argument("--png-dir", help="write point overlays here")
    p.set_defaults(func=cmd_calib_check)

    p = sub.add_parser("coverage", help="mask coverage per sequence, from the written masks")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_coverage)

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

    bev_repo = "apairo-robotics/rellis3d_bev"
    p = sub.add_parser("bev-process", help="robot-centred BEV elevation / semantic / trav grids, one per scan")
    p.add_argument("sequences", nargs="*", help=f"default: all of {config.SEQUENCES}")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--accumulate", type=int, default=config.BEV_ACCUMULATE, help="scans per grid (past only)")
    p.add_argument("--stride", type=int, default=config.BEV_STRIDE, help="scans between accumulated scans")
    p.add_argument("--overwrite", action="store_true", help="recompute (default: skip an unchanged recipe)")
    p.set_defaults(func=cmd_bev_process)

    p = sub.add_parser("bev-render", help="PNGs of the BEV grids beside the camera image")
    p.add_argument("view", help="a BEV-processed sequence view, e.g. <out>/00003")
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--frames", type=int, nargs="*", help="explicit scan indices (overrides --every)")
    p.add_argument("--png-dir", required=True)
    p.set_defaults(func=cmd_bev_render)

    p = sub.add_parser("bev-quality", help="trajectory trav grid vs lidar-label semantic grid, per sequence")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=1, help="every N-th frame")
    p.set_defaults(func=cmd_bev_quality)

    p = sub.add_parser("bev-export", help="BEV sequences -> one local LeRobotDataset (never pushed)")
    p.add_argument("out", help="the --out of bev-process")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default=bev_repo)
    p.add_argument("--lerobot-root", required=True, help="local output directory")
    p.add_argument("--limit", type=int, help="first N frames per sequence (smoke test)")
    p.set_defaults(func=cmd_bev_export)

    p = sub.add_parser("bev-check", help="reload a BEV export: features, episodes, value sets")
    p.add_argument("lerobot_root")
    p.add_argument("--repo-id", default=bev_repo)
    p.set_defaults(func=cmd_bev_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
