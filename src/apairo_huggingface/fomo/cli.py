"""``apairo-fomo``: FoMo trajectories -> traversability labels -> LeRobotDataset.

    apairo-fomo list                         # trajectories in the bucket, with sizes
    apairo-fomo fetch 2025-10-14/red_2025-10-14-11-41 --root ~/data/fomo
    apairo-fomo process ~/data/fomo/2025-10-14/red_2025-10-14-11-41
    apairo-fomo run --root ~/data/fomo --color red      # fetch + process, every deployment
    apairo-fomo export ~/data/fomo --repo-id apairo-robotics/fomo_traversability --out ~/data/fomo_lerobot
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from . import config


def _trajectories(root: Path) -> list[Path]:
    """Processed trajectory dirs under ``root`` (``<deployment>/<trajectory>``)."""
    return sorted(p.parent for p in root.glob(f"*/*/{config.TRAV_MASK}/timestamps.txt"))


def cmd_list(args) -> None:
    from .fetch import catalog

    for dep, trajs in catalog().items():
        for name, v in trajs.items():
            if args.color and not name.startswith(args.color + "_"):
                continue
            s = v["sensors"]
            gb = (s.get("robosense", 0) + s.get("basler", 0) + v["metadata"]) / 1e9
            print(f"{dep}/{name}\t{gb:6.1f} GB (robosense + basler + metadata)")


def process(traj_dir: Path, *, overwrite: bool = False, drop_lidar: bool = False) -> dict:
    from .evaluate import ate
    from .labeling import label_images, label_scans, odometry

    t0 = time.time()
    odometry(traj_dir, overwrite=overwrite)
    label_scans(traj_dir, overwrite=overwrite)
    label_images(traj_dir, overwrite=overwrite)
    report = {"trajectory": f"{traj_dir.parent.name}/{traj_dir.name}", **ate(traj_dir),
              "seconds": round(time.time() - t0)}
    (traj_dir / "apairo_fomo_report.json").write_text(json.dumps(report, indent=2))
    logging.info("%s", report)
    if drop_lidar:
        # The export needs images, masks, poses and commands -- not the scans.
        shutil.rmtree(traj_dir / config.LIDAR)
        logging.info("%s: dropped %s/", traj_dir.name, config.LIDAR)
    return report


def cmd_fetch(args) -> None:
    from .fetch import fetch_trajectory

    fetch_trajectory(args.trajectory, args.root)


def cmd_process(args) -> None:
    process(Path(args.traj_dir), overwrite=args.overwrite, drop_lidar=args.drop_lidar)


def cmd_run(args) -> None:
    from .fetch import catalog, fetch_trajectory

    root = Path(args.root)
    for dep, trajs in catalog().items():
        for name in trajs:
            if not name.startswith(args.color + "_"):
                continue
            traj_dir = root / dep / name
            if (traj_dir / "apairo_fomo_report.json").exists() and not args.overwrite:
                logging.info("%s/%s: already processed", dep, name)
                continue
            try:
                fetch_trajectory(f"{dep}/{name}", root)
                process(traj_dir, overwrite=args.overwrite, drop_lidar=args.drop_lidar)
            except Exception as exc:  # noqa: BLE001 -- one bad trajectory must not stop the batch
                logging.warning("%s/%s failed: %s", dep, name, exc)


def cmd_export(args) -> None:
    from .export import to_lerobot

    root = Path(args.root)
    trajs = [root] if (root / config.TRAV_MASK).is_dir() else _trajectories(root)
    if not trajs:
        raise SystemExit(f"no processed trajectory under {root}")
    logging.info("exporting %d trajectories", len(trajs))
    to_lerobot(trajs, args.repo_id, root=args.out, push=args.push)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="apairo-fomo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="trajectories in the FoMo bucket")
    p.add_argument("--color", help="only this trajectory colour (red, blue, ...)")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("fetch", help="download one trajectory (lidar, camera, calib, metadata)")
    p.add_argument("trajectory", help="<deployment>/<trajectory>")
    p.add_argument("--root", required=True)
    p.set_defaults(func=cmd_fetch)

    for name, func, helptext in (("process", cmd_process, "odometry, scan labels, image masks, ATE"),):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("traj_dir")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--drop-lidar", action="store_true", help="delete the scans once labelled")
        p.set_defaults(func=func)

    p = sub.add_parser("run", help="fetch + process one trajectory colour across every deployment")
    p.add_argument("--root", required=True)
    p.add_argument("--color", default="red")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--drop-lidar", action="store_true", help="delete the scans once labelled")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", help="processed trajectories -> one LeRobotDataset")
    p.add_argument("root", help="a processed trajectory, or a root of <deployment>/<trajectory>")
    p.add_argument("--repo-id", default="apairo-robotics/fomo_traversability")
    p.add_argument("--out", help="local output directory (default: LeRobot's cache)")
    p.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("kiss_icp").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
