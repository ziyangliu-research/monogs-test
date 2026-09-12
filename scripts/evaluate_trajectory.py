#!/usr/bin/env python3
import argparse
import json

from utils.trajectory_eval import evaluate_pose_files


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frame-indexed trajectories in the common format: "
            "frame_id tx ty tz qx qy qz qw"
        )
    )
    parser.add_argument("--est", required=True, help="Estimated trajectory txt")
    parser.add_argument("--gt", required=True, help="Ground-truth trajectory txt")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--correct-scale",
        action="store_true",
        help="Use Sim(3) scale correction (normally only for monocular systems)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    result = evaluate_pose_files(
        args.est,
        args.gt,
        output_dir=args.output_dir,
        correct_scale=args.correct_scale,
        save_plot=not args.no_plot,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
