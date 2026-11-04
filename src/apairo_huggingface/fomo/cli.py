"""``apairo-fomo``: FoMo trajectories -> traversability labels, one channel per sensor.

    apairo-fomo list                         # trajectories in the bucket, with sizes
    apairo-fomo fetch 2025-10-14/red_2025-10-14-11-41 --root ~/data/fomo
    apairo-fomo process ~/data/fomo/2025-10-14/red_2025-10-14-11-41
    apairo-fomo run --root ~/data/fomo --color red      # fetch + process, every deployment
    apairo-fomo export ~/data/fomo --repo-id apairo-robotics/fomo_traversability --out ~/data/fomo_lerobot
    apairo-fomo masks ~/data/fomo/2024-11-21/red_2024-11-21-10-34       # every camera, from the scans
    apairo-fomo export-masks ~/data/fomo --out ~/data/fomo_labels
    apairo-fomo verify 2024-11-21/red_2024-11-21-10-34 --labels ~/data/fomo_labels --n 20
    apairo-fomo export-labels ~/data/fomo --out ~/data/fomo_point_labels
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from . import config


def _trajectories(root: Path, camera: str = config.CAMERA) -> list[Path]:
    """Trajectory dirs under ``root`` (``<deployment>/<trajectory>``) holding
    a complete mask channel of ``camera``."""
    mask = config.TRAV_MASKS[camera]
    return sorted(p.parent.parent for p in root.glob(f"*/*/{mask}/timestamps.txt"))


def cmd_list(args) -> None:
    from .fetch import catalog

    for dep, trajs in catalog().items():
        for name, v in trajs.items():
            if args.color and not name.startswith(args.color + "_"):
                continue
            s = v["sensors"]
            gb = (s.get(config.LIDAR, 0) + v["metadata"]) / 1e9
            cams = "  ".join(f"{c} {s.get(c, 0) / 1e9:.1f}" for c in config.CAMERAS)
            print(f"{dep}/{name}\t{gb:6.1f} GB ({config.LIDAR} + metadata; images, not needed: {cams} GB)")


def process(traj_dir: Path, *, cameras=config.CAMERAS, overwrite: bool = False,
            drop_lidar: bool = False) -> dict:
    from .evaluate import ate
    from .calibration import FOOTPRINT
    from .labeling import camera_stamps, label_images, label_scans, odometry

    t0 = time.time()
    odometry(traj_dir, overwrite=overwrite)
    label_scans(traj_dir, overwrite=overwrite)
    if any(FOOTPRINT[c] == "past" for c in cameras):
        label_scans(traj_dir, footprint="past", overwrite=overwrite)
    traj = f"{traj_dir.parent.name}/{traj_dir.name}"
    for camera in cameras:
        label_images(traj_dir, camera=camera, stamps_us=camera_stamps(traj, camera), overwrite=overwrite)
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

    from .fetch import SENSORS

    fetch_trajectory(args.trajectory, args.root, subdirs=SENSORS + tuple(args.camera or ()))


def cmd_process(args) -> None:
    process(Path(args.traj_dir), cameras=args.camera or config.CAMERAS, overwrite=args.overwrite,
            drop_lidar=args.drop_lidar)


def cmd_masks(args) -> None:
    """Image masks from the scans, the stored labels and poses, and the camera
    stamps of the bucket listing (no image is downloaded)."""
    from .labeling import camera_stamps, check_scans, label_images, label_scans

    traj_dir = Path(args.traj_dir)
    traj = f"{traj_dir.parent.name}/{traj_dir.name}"
    logging.info("%s: %d scans match %s", traj, check_scans(traj_dir), config.TRAV_TRAJ)
    cameras = args.camera or list(config.CAMERAS)
    from .calibration import FOOTPRINT

    if any(FOOTPRINT[c] == "past" for c in cameras):
        label_scans(traj_dir, footprint="past", overwrite=args.overwrite)
    for camera in cameras:
        label_images(traj_dir, camera=camera, stamps_us=camera_stamps(traj, camera),
                     overwrite=args.overwrite)


def cmd_export_masks(args) -> None:
    from .camera_labels import export

    root = Path(args.root)
    cameras = args.camera or list(config.CAMERAS)
    trajs = sorted(p.parent.parent for p in root.glob(f"*/*/{config.TRAV_MASKS[cameras[0]]}/timestamps.txt"))
    if not trajs:
        raise SystemExit(f"no trajectory with {cameras[0]} masks under {root}")
    export(trajs, args.out, cameras)


def cmd_verify(args) -> None:
    from .verify import verify

    report = verify(args.trajectory, args.labels, args.cache, args.out,
                    cameras=args.camera or list(config.CAMERAS), n=args.n, rerun=args.rerun)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "overlays"} if isinstance(v, dict) else v)
                      for k, v in report.items()}, indent=2))


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
                fetch_trajectory(f"{dep}/{name}", root)  # lidar, calib, metadata: no images
                process(traj_dir, cameras=args.camera or config.CAMERAS, overwrite=args.overwrite,
                        drop_lidar=args.drop_lidar)
            except Exception as exc:  # noqa: BLE001 -- one bad trajectory must not stop the batch
                logging.warning("%s/%s failed: %s", dep, name, exc)


def cmd_export(args) -> None:
    from .export import MAX_SATURATED_SHARE, image_quality, to_lerobot

    root = Path(args.root)
    mask = config.TRAV_MASKS[args.camera]
    trajs = [root] if (root / mask).is_dir() else _trajectories(root, args.camera)
    kept = []
    for t in trajs:
        q = image_quality(t, args.camera)
        out = q["saturated_share"] > MAX_SATURATED_SHARE
        logging.info("%s/%s: %s mean %.1f, saturated %.3f -> %s", t.parent.name, t.name,
                     args.camera, q["mean_intensity"], q["saturated_share"],
                     "excluded" if out else "kept")
        if not out:
            kept.append(t)
    if not kept:
        raise SystemExit(f"no processed trajectory under {root}")
    logging.info("exporting %d trajectories", len(kept))
    to_lerobot(kept, args.repo_id, camera=args.camera, root=args.out, push=args.push)


def cmd_export_labels(args) -> None:
    from .point_labels import export

    root = Path(args.root)
    trajs = [root] if (root / config.TRAV_TRAJ).is_dir() else sorted(
        p.parent.parent for p in root.glob(f"*/*/{config.TRAV_TRAJ}/timestamps.txt"))
    if not trajs:
        raise SystemExit(f"no labelled trajectory under {root}")
    logging.info("exporting the point labels of %d trajectories", len(trajs))
    export(trajs, args.out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="apairo-fomo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="trajectories in the FoMo bucket")
    p.add_argument("--color", help="only this trajectory colour (red, blue, ...)")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("fetch", help="download one trajectory (lidar, calib, metadata)")
    p.add_argument("trajectory", help="<deployment>/<trajectory>")
    p.add_argument("--root", required=True)
    p.add_argument("--camera", action="append", choices=config.CAMERAS,
                   help="also download this camera's images; repeatable")
    p.set_defaults(func=cmd_fetch)

    for name, func, helptext in (("process", cmd_process, "odometry, scan labels, image masks, ATE"),):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("traj_dir")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--drop-lidar", action="store_true", help="delete the scans once labelled")
        p.add_argument("--camera", action="append", choices=config.CAMERAS, help="repeatable; default: all")
        p.set_defaults(func=func)

    p = sub.add_parser("masks", help="camera masks from the scans, stored labels and poses")
    p.add_argument("traj_dir")
    p.add_argument("--camera", action="append", choices=config.CAMERAS, help="repeatable; default: all")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_masks)

    p = sub.add_parser("export-masks", help="camera masks -> one parquet per camera and trajectory")
    p.add_argument("root", help="a root of <deployment>/<trajectory>")
    p.add_argument("--out", required=True)
    p.add_argument("--camera", action="append", choices=config.CAMERAS, help="repeatable; default: all")
    p.set_defaults(func=cmd_export_masks)

    p = sub.add_parser("verify", help="check the label tables against FoMo's raw files")
    p.add_argument("trajectory", help="<deployment>/<trajectory>")
    p.add_argument("--labels", required=True, help="root of the label tables (robosense/, <camera>/)")
    p.add_argument("--camera", action="append", choices=config.CAMERAS, help="repeatable; default: all")
    p.add_argument("--n", type=int, default=20, help="scans and frames per camera to download")
    p.add_argument("--cache", default="fomo_verify_cache", help="where raw files are downloaded")
    p.add_argument("--out", default="fomo_verify", help="overlays and the JSON report")
    p.add_argument("--rerun", action="store_true", help="also log the frames to a running Rerun viewer")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("run", help="fetch + process one trajectory colour across every deployment")
    p.add_argument("--root", required=True)
    p.add_argument("--color", default="red")
    p.add_argument("--camera", action="append", choices=config.CAMERAS, help="repeatable; default: all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--drop-lidar", action="store_true", help="delete the scans once labelled")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", help="processed trajectories -> one LeRobotDataset")
    p.add_argument("root", help="a processed trajectory, or a root of <deployment>/<trajectory>")
    p.add_argument("--repo-id", default="apairo-robotics/fomo_traversability")
    p.add_argument("--camera", default=config.CAMERA, choices=sorted(config.TRAV_MASKS))
    p.add_argument("--out", help="local output directory (default: LeRobot's cache)")
    p.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("export-labels",
                       help="per-point scan labels -> one parquet per trajectory (labels only)")
    p.add_argument("root", help="a processed trajectory, or a root of <deployment>/<trajectory>")
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_export_labels)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("kiss_icp").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
