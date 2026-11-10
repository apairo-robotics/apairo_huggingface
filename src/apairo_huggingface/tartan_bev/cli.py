"""``apairo-tartan-bev``: TartanDrive 2.0 -> robot-centred BEV grids -> LeRobotDataset.

    apairo-tartan-bev process --root /mnt/vault-fellowship/tartan --out ~/data/tartan_bev        # config.SEQUENCES
    apairo-tartan-bev process --root /mnt/vault-fellowship/tartan --out ~/data/tartan_bev <seq>  # one recording
    apairo-tartan-bev render ~/data/tartan_bev/<seq> --every 500 --png-dir /tmp/tbev
    apairo-tartan-bev quality --out ~/data/tartan_bev
    apairo-tartan-bev export ~/data/tartan_bev --lerobot-root ~/data/tartan_bev_lerobot
    apairo-tartan-bev check ~/data/tartan_bev_lerobot

``--root`` is the TartanDrive tree (one directory per recording, as written by
apairo_extractor); it is only read. Everything is written under ``--out``:
one symlink view per recording holding the derived channels.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import config

REPO_ID = "apairo-robotics/tartandrive_bev"


def cmd_process(args) -> None:
    from .bev import process_sequence

    out = Path(args.out)
    for seq in args.sequences or config.SEQUENCES:
        report = process_sequence(args.root, seq, out, overwrite=args.overwrite,
                                  accumulate=args.accumulate, stride=args.stride)
        (out / seq / "apairo_tartan_bev_report.json").write_text(json.dumps(report, indent=2))


def cmd_render(args) -> None:
    from .bev import open_view
    from .render import render_view

    view = Path(args.view)
    n = len(open_view(view, [config.BEV_TRAV]).timestamps[config.BEV_TRAV])
    indices = args.frames or list(range(0, n, args.every))
    for path in render_view(view, indices, args.png_dir):
        print(path)


def _processed(out: Path, sequences) -> list[str]:
    return [s for s in (sequences or config.SEQUENCES) if (out / s / config.BEV_TRAV).is_dir()]


def cmd_quality(args) -> None:
    from .quality import sequence_counts, table, total

    out = Path(args.out)
    reports = [sequence_counts(out / s, every=args.every) for s in _processed(out, args.sequences)]
    reports.append(total(reports))
    (out / "bev_quality.json").write_text(json.dumps(reports, indent=2))
    print(table(reports))


def cmd_export(args) -> None:
    from .export import to_lerobot

    out = Path(args.out)
    to_lerobot([out / s for s in _processed(out, args.sequences)], args.repo_id,
               lerobot_root=args.lerobot_root, limit=args.limit)


def cmd_check(args) -> None:
    from .export import check

    for key, value in check(args.repo_id, args.lerobot_root).items():
        print(f"{key}: {value}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="apairo-tartan-bev", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("process", help="robot-centred BEV elevation / geometry / trav grids, one per scan")
    p.add_argument("sequences", nargs="*", help="recordings (default: config.SEQUENCES)")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--accumulate", type=int, default=config.BEV_ACCUMULATE, help="scans per grid (past only)")
    p.add_argument("--stride", type=int, default=config.BEV_STRIDE, help="scans between accumulated scans")
    p.add_argument("--overwrite", action="store_true", help="recompute (default: skip an unchanged recipe)")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("render", help="PNGs of the BEV grids beside the camera image")
    p.add_argument("view", help="a processed recording view, e.g. <out>/<seq>")
    p.add_argument("--every", type=int, default=500)
    p.add_argument("--frames", type=int, nargs="*", help="explicit scan indices (overrides --every)")
    p.add_argument("--png-dir", required=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("quality", help="driven cells vs geometry (flat share), coverage, per recording")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=1, help="every N-th frame")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("export", help="processed recordings -> one local LeRobotDataset (never pushed)")
    p.add_argument("out", help="the --out of process")
    p.add_argument("sequences", nargs="*")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--lerobot-root", required=True, help="local output directory")
    p.add_argument("--limit", type=int, help="first N frames per recording (smoke test)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check", help="reload a BEV export: features, episodes, value ranges")
    p.add_argument("lerobot_root")
    p.add_argument("--repo-id", default=REPO_ID)
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
