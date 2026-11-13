"""Command-line entry points: label TartanDrive, then export it to LeRobot."""

from __future__ import annotations

import argparse
import os

from . import config
from .export import load_tartan, to_lerobot, validate


def label_main(argv=None):
    """``apairo-tartan-label`` — materialize the traversability channels."""
    from .labeling import label_root

    parser = argparse.ArgumentParser(
        description="Label TartanDrive recordings with trajectory traversability "
        "(writes trav_traj, lidar_uv and trav_mask channels)."
    )
    parser.add_argument("--root", default=os.environ.get("APAIRO_TARTAN_ROOT", "/data/tartan"),
                        help="TartanDrive root (many recordings) or a single sequence directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="relabel recordings whose channels already exist")
    parser.add_argument("--keep-uv", action="store_true",
                        help="also persist lidar_uv (~2 GB/recording; recomputable on demand)")
    parser.add_argument("--poses", choices=sorted(config.POSE_SOURCES), default="ins",
                        help="pose source: ins (the vehicle's INS, default) or a lidar "
                        "odometry computed beforehand, for comparison")
    args = parser.parse_args(argv)
    label_root(args.root, poses_key=config.POSE_SOURCES[args.poses],
               overwrite=args.overwrite, keep_uv=args.keep_uv)


def gt_label_main(argv=None):
    """``apairo-tartan-gt-label`` — project hand-labeled point clouds into trav_mask_gt."""
    from .eval_labeling import label_ground_truth_root

    parser = argparse.ArgumentParser(
        description="Project hand-labeled ground_truth point clouds (written by "
        "apairo_experiments/scripts/import_toaster_gt.py) into a trav_mask_gt image "
        "channel, per recording. Recordings without hand labels are skipped."
    )
    parser.add_argument("--root", default=os.environ.get("APAIRO_TARTAN_ROOT", "/data/tartan"),
                        help="TartanDrive root (many recordings) or a single sequence directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="reproject recordings whose trav_mask_gt already exists")
    args = parser.parse_args(argv)
    label_ground_truth_root(args.root, overwrite=args.overwrite)


def gt_export_main(argv=None):
    """``apairo-tartan-gt-export`` — build/push the hand-labeled eval benchmark."""
    from .eval_export import to_eval_dataset, validate_eval

    parser = argparse.ArgumentParser(
        description="Export the hand-labeled TartanDrive eval benchmark (image + "
        "projected mask + raw points/labels) as a plain HF dataset -- not a "
        "LeRobotDataset, see docs/eval_benchmark.md."
    )
    parser.add_argument("--root", default=os.environ.get("APAIRO_TARTAN_ROOT", "/data/tartan"),
                        help="TartanDrive root (many recordings) or a single sequence directory")
    parser.add_argument("--repo-id", default="tartandrive_offroad_eval", help="target dataset id")
    parser.add_argument("--no-push", action="store_true", help="build locally, do not push to the Hub")
    parser.add_argument("--dry-run", action="store_true", help="validate reads only, without building the dataset")
    args = parser.parse_args(argv)

    if args.dry_run:
        counts, total_points = validate_eval(args.root)
        for name, n in counts.items():
            print(f"  {name}: {n} labeled frame(s)")
        print(f"\n{len(counts)} recording(s), {sum(counts.values())} frame(s), "
              f"{total_points} point(s) -- schema OK")
    else:
        dataset = to_eval_dataset(args.root, args.repo_id, push=not args.no_push)
        print(f"Wrote {len(dataset)} hand-labeled frame(s) to {args.repo_id}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a TartanDrive recording to a LeRobotDataset.")
    parser.add_argument("--root", default=os.environ.get("APAIRO_TARTAN_ROOT", "/data/tartan"),
                        help="TartanDrive root (many recordings) or a single sequence directory")
    parser.add_argument("--repo-id", default="tartandrive_offroad", help="target LeRobotDataset id")
    parser.add_argument("--no-push", action="store_true", help="build locally, do not push to the Hub")
    parser.add_argument("--dry-run", action="store_true", help="validate reads only, without lerobot")
    args = parser.parse_args(argv)

    tartan = load_tartan(args.root)

    if args.dry_run:
        features, counts = validate(tartan)
        for i, n in enumerate(counts):
            print(f"  episode {i}: {n} frames")
        print(f"\n{len(counts)} episode(s), {sum(counts)} frames -- schema OK")
        for name, spec in features.items():
            print(f"  {name:28} {spec['dtype']:8} {spec['shape']}")
    else:
        dataset = to_lerobot(tartan, args.repo_id, push=not args.no_push)
        print(f"Wrote {dataset.num_episodes} episodes / {dataset.num_frames} frames to {args.repo_id}")


if __name__ == "__main__":
    main()
