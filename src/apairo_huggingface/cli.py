"""Command-line entry points: label TartanDrive, then export it to LeRobot."""

from __future__ import annotations

import argparse
import os

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
    args = parser.parse_args(argv)
    label_root(args.root, overwrite=args.overwrite, keep_uv=args.keep_uv)


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
