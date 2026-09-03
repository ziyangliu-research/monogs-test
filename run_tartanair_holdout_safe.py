import csv
import json
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

import slam as monogs_slam
from gaussian_splatting.utils.system_utils import mkdir_p
from run_tartanair_holdout import HoldoutFrontEnd, largest_contiguous_map
from utils.logging_utils import Log


class ProducerQueueProxy:
    """Producer-side queue wrapper for CUDA IPC messages.

    MonoGS's backend is the producer for frontend_queue. PyTorch warns against a
    process putting CUDA tensors into a queue and later getting from that same
    queue. The released backend does exactly that during shutdown cleanup.

    This wrapper preserves normal put() behavior during SLAM, but:
      * marks the producer queue so the child process does not wait for its
        feeder thread during process exit; and
      * reports empty=True to the backend's shutdown-only self-drain loop, so
        the producer never rebuilds its own CUDA IPC tensors.

    The main/frontend process still owns the original queue object and consumes
    messages normally while tracking is active.
    """

    def __init__(self, queue):
        self.queue = queue
        self._join_cancelled = False

    def put(self, item):
        if not self._join_cancelled:
            self.queue.cancel_join_thread()
            self._join_cancelled = True
        return self.queue.put(item)

    def empty(self):
        # frontend_queue.empty() is only called by BackEnd.run() in its final
        # shutdown self-drain. Normal backend operation only calls put().
        return True

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)


class BenchmarkBackEnd(monogs_slam.BackEnd):
    """Released MonoGS backend with benchmark-only lifecycle/timing hooks."""

    def set_hyperparams(self):
        super().set_hyperparams()
        if not isinstance(self.frontend_queue, ProducerQueueProxy):
            self.frontend_queue = ProducerQueueProxy(self.frontend_queue)

    def color_refinement(self):
        """Time the released color_refinement() without changing its logic."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        super().color_refinement()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        timing = {
            "color_refinement_sec": float(elapsed),
            "iterations": 26000,
        }
        save_dir = self.config["Results"]["save_dir"]
        with open(
            os.path.join(save_dir, "offline_timing.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(timing, f, indent=4)
        Log("Color refinement wall time", elapsed, tag="Eval")


def summarize_rows(rows):
    if not rows:
        raise RuntimeError("No rendering rows selected for this split")
    return {
        "num_frames": len(rows),
        "mean_psnr": float(np.mean([float(r["psnr"]) for r in rows])),
        "mean_ssim": float(np.mean([float(r["ssim"]) for r in rows])),
        "mean_lpips": float(np.mean([float(r["lpips"]) for r in rows])),
    }


def write_benchmark_row(
    save_dir,
    sequence,
    max_map,
    train_result,
    test_result,
    trajectory_result,
    online_fps,
    online_time_sec,
    offline_time_sec,
    gaussian_count,
):
    system_total_sec = float(online_time_sec) + float(offline_time_sec)
    row = {
        "Sequence": sequence,
        "MaxMap": int(max_map["frames"]),
        "Train PSNR": float(train_result["mean_psnr"]),
        "Train SSIM": float(train_result["mean_ssim"]),
        "Train LPIPS": float(train_result["mean_lpips"]),
        "Test PSNR": float(test_result["mean_psnr"]),
        "Test SSIM": float(test_result["mean_ssim"]),
        "Test LPIPS": float(test_result["mean_lpips"]),
        "ATE(m)": float(trajectory_result["ate_rmse_m"]),
        "Gaussians": int(gaussian_count),
        "Online FPS": float(online_fps),
        "Online Time(s)": float(online_time_sec),
        "Offline Refine(s)": float(offline_time_sec),
        "System Total(s)": float(system_total_sec),
    }
    path = os.path.join(save_dir, "benchmark_row.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def main():
    parser = ArgumentParser(
        description="MonoGS TartanAir 80/20 holdout benchmark with safe CUDA IPC lifetime"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--range", dest="range_spec", type=str, default=None)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--holdout-offset", type=int, default=4)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument(
        "--color-refinement",
        action="store_true",
        help=(
            "Run MonoGS's released global color refinement after online SLAM and "
            "evaluate Train/Test rendering metrics on the refined map"
        ),
    )
    args = parser.parse_args(sys.argv[1:])

    if args.holdout_every <= 1:
        raise ValueError("--holdout-every must be > 1")
    if not 0 <= args.holdout_offset < args.holdout_every:
        raise ValueError("--holdout-offset must be in [0, holdout_every)")

    mp.set_start_method("spawn")

    config = monogs_slam.load_config(args.config)
    monogs_slam.apply_sequence_override(config, args.sequence)
    monogs_slam.apply_range_override(config, args.range_spec)
    config["Dataset"]["holdout_every"] = int(args.holdout_every)
    config["Dataset"]["holdout_offset"] = int(args.holdout_offset)

    config["Results"]["save_results"] = True
    config["Results"]["use_gui"] = False
    config["Results"]["eval_rendering"] = True
    config["Results"]["use_wandb"] = False
    config["Results"]["save_trj"] = False

    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sequence_name = os.path.basename(config["Dataset"]["dataset_path"].rstrip(os.sep))
    dataset_tag = f'{config["Dataset"]["type"]}_{sequence_name}'
    range_tag = monogs_slam.dataset_range_label(config)
    split_tag = f"holdout_{args.holdout_every}_{args.holdout_offset}"
    if args.color_refinement:
        split_tag += "_color_refinement"
    save_dir = os.path.join(
        config["Results"]["save_dir"],
        dataset_tag,
        range_tag,
        split_tag,
        current_datetime,
    )
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)

    with open(os.path.join(save_dir, "config.yml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    Log("saving results in " + save_dir)

    # The wrapper owns output/evaluation. Disable the legacy in-frontend result
    # path while keeping the final SLAM-level evaluation enabled below.
    config["Results"]["save_results"] = False
    config["Results"]["save_trj"] = False

    monogs_slam.FrontEnd = HoldoutFrontEnd
    monogs_slam.BackEnd = BenchmarkBackEnd

    # Generic SLAM --eval renders all frames BEFORE backend shutdown. This is
    # essential: self.frontend.gaussians may contain CUDA IPC storage produced
    # by the backend, so evaluation must happen while that producer is alive.
    original_eval_rendering = monogs_slam.eval_rendering

    def benchmark_eval_rendering(*call_args, **call_kwargs):
        call_kwargs["save_images"] = bool(args.save_images)
        return original_eval_rendering(*call_args, **call_kwargs)

    monogs_slam.eval_rendering = benchmark_eval_rendering

    runner = monogs_slam.SLAM(
        config,
        save_dir=save_dir,
        experiment_mode="eval",
        color_refinement=bool(args.color_refinement),
    )

    # From here on, use only CPU/files produced while backend was alive. Do not
    # access runner.frontend.gaussians after the backend process has exited.
    with open(os.path.join(save_dir, "timing.json"), "r", encoding="utf-8") as f:
        timing_result = json.load(f)
    with open(
        os.path.join(save_dir, "trajectory", "trajectory_metrics.json"),
        "r",
        encoding="utf-8",
    ) as f:
        trajectory_result = json.load(f)
    with open(
        os.path.join(save_dir, "psnr", "final", "per_frame_metrics.json"),
        "r",
        encoding="utf-8",
    ) as f:
        per_frame = json.load(f)

    offline_time_sec = 0.0
    offline_timing_path = os.path.join(save_dir, "offline_timing.json")
    if args.color_refinement:
        if not os.path.isfile(offline_timing_path):
            raise RuntimeError(
                "Color refinement was requested but offline_timing.json was not produced"
            )
        with open(offline_timing_path, "r", encoding="utf-8") as f:
            offline_timing = json.load(f)
        offline_time_sec = float(offline_timing["color_refinement_sec"])
    else:
        offline_timing = {
            "color_refinement_sec": 0.0,
            "iterations": 0,
        }

    train_rows, test_rows = [], []
    for row in per_frame:
        frame_id = int(row["frame_id"])
        if frame_id % args.holdout_every == args.holdout_offset:
            test_rows.append(row)
        else:
            train_rows.append(row)

    train_result = summarize_rows(train_rows)
    test_result = summarize_rows(test_rows)
    max_map = largest_contiguous_map(runner.dataset, runner.frontend.cameras)
    online_fps = float(timing_result["streaming_fps"])
    online_time_sec = float(timing_result["end_to_end_without_evaluation_sec"])
    gaussian_count = int(timing_result["final_gaussian_count"])

    row = write_benchmark_row(
        save_dir,
        sequence_name,
        max_map,
        train_result,
        test_result,
        trajectory_result,
        online_fps,
        online_time_sec,
        offline_time_sec,
        gaussian_count,
    )

    summary = {
        "sequence": sequence_name,
        "split": {
            "holdout_every": int(args.holdout_every),
            "holdout_offset": int(args.holdout_offset),
            "train_frames": len(train_rows),
            "test_frames": len(test_rows),
        },
        "max_map": max_map,
        "timing": timing_result,
        "offline_timing": offline_timing,
        "system_total_sec_excluding_metric_evaluation": float(
            online_time_sec + offline_time_sec
        ),
        "trajectory": trajectory_result,
        "train_rendering": train_result,
        "test_rendering": test_result,
        "final_gaussian_count": gaussian_count,
        "benchmark_row": row,
        "cuda_ipc_safe_eval": True,
        "color_refinement": bool(args.color_refinement),
    }
    with open(
        os.path.join(save_dir, "holdout_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    Log(
        "Holdout split",
        f"train={len(train_rows)}, test={len(test_rows)}, "
        f"rule=frame_id % {args.holdout_every} == {args.holdout_offset}",
        tag="Eval",
    )
    Log(
        "Benchmark row",
        f'{sequence_name} | MaxMap={row["MaxMap"]} | '
        f'Train={row["Train PSNR"]:.3f}/{row["Train SSIM"]:.4f}/'
        f'{row["Train LPIPS"]:.4f} | '
        f'Test={row["Test PSNR"]:.3f}/{row["Test SSIM"]:.4f}/'
        f'{row["Test LPIPS"]:.4f} | '
        f'ATE={row["ATE(m)"]:.6f} m | '
        f'Gaussians={row["Gaussians"]:,} | '
        f'OnlineFPS={row["Online FPS"]:.3f} | '
        f'Online={row["Online Time(s)"]:.1f}s | '
        f'Offline={row["Offline Refine(s)"]:.1f}s | '
        f'Total={row["System Total(s)"]:.1f}s',
        tag="Eval",
    )
    Log("Done.")


if __name__ == "__main__":
    main()
