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
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from run_tartanair_holdout import HoldoutFrontEnd, largest_contiguous_map
from utils.logging_utils import Log


class ProducerQueueProxy:
    """Producer-side queue wrapper for CUDA IPC messages.

    Normal backend->frontend put() behavior is preserved.  The wrapper only
    avoids the released backend's shutdown-time self-drain of CUDA IPC tensors.
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
        return True

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)


class PairedBenchmarkBackEnd(monogs_slam.BackEnd):
    """Released MonoGS backend plus benchmark-only snapshot/timing hooks."""

    def set_hyperparams(self):
        super().set_hyperparams()
        if not isinstance(self.frontend_queue, ProducerQueueProxy):
            self.frontend_queue = ProducerQueueProxy(self.frontend_queue)

    def color_refinement(self):
        """Snapshot the exact online-final map, then time released CR logic."""
        save_dir = self.config["Results"]["save_dir"]
        pre_dir = os.path.join(save_dir, "point_cloud", "online_final")
        mkdir_p(pre_dir)
        pre_ply = os.path.join(pre_dir, "point_cloud.ply")

        # Snapshot I/O is deliberately outside Offline Refine(s).
        torch.cuda.synchronize()
        self.gaussians.save_ply(pre_ply)
        torch.cuda.synchronize()
        Log("Saved online-final Gaussian snapshot", pre_ply, tag="Eval")

        start = time.perf_counter()
        super().color_refinement()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        timing = {
            "color_refinement_sec": float(elapsed),
            "iterations": 26000,
            "online_final_snapshot": pre_ply,
        }
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


def split_rows(per_frame, holdout_every, holdout_offset):
    train_rows, test_rows = [], []
    for row in per_frame:
        frame_id = int(row["frame_id"])
        if frame_id % holdout_every == holdout_offset:
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def make_row(
    variant,
    sequence,
    total_frames,
    max_map,
    train_result,
    test_result,
    trajectory_result,
    online_fps,
    online_time_sec,
    offline_time_sec,
    gaussian_count,
):
    return {
        "Variant": variant,
        "Sequence": sequence,
        "TotalFrames": int(total_frames),
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
        "System Total(s)": float(online_time_sec + offline_time_sec),
    }


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser(
        description=(
            "MonoGS paired TartanAir benchmark: one online run, exact pre-CR "
            "snapshot, released 26k color refinement, paired pre/post metrics"
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--range", dest="range_spec", type=str, default=None)
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--holdout-offset", type=int, default=4)
    parser.add_argument("--save-images", action="store_true")
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
    split_tag = (
        f"holdout_{args.holdout_every}_{args.holdout_offset}_paired_color_refinement"
    )
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

    # Benchmark owns all output.  Keep released tracking/mapping behavior.
    config["Results"]["save_results"] = False
    config["Results"]["save_trj"] = False

    monogs_slam.FrontEnd = HoldoutFrontEnd
    monogs_slam.BackEnd = PairedBenchmarkBackEnd

    # Post-CR metrics are evaluated inside SLAM while the backend CUDA producer
    # is alive.  Image writing remains optional and does not affect metrics.
    original_eval_rendering = monogs_slam.eval_rendering

    def benchmark_eval_rendering(*call_args, **call_kwargs):
        call_kwargs["save_images"] = bool(args.save_images)
        return original_eval_rendering(*call_args, **call_kwargs)

    monogs_slam.eval_rendering = benchmark_eval_rendering

    runner = monogs_slam.SLAM(
        config,
        save_dir=save_dir,
        experiment_mode="eval",
        color_refinement=True,
    )

    # Everything below is metric evaluation / reporting and is excluded from
    # Online Time and Offline Refine Time.
    with open(os.path.join(save_dir, "timing.json"), "r", encoding="utf-8") as f:
        timing_result = json.load(f)
    with open(
        os.path.join(save_dir, "offline_timing.json"), "r", encoding="utf-8"
    ) as f:
        offline_timing = json.load(f)
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
        post_per_frame = json.load(f)

    pre_ply = offline_timing["online_final_snapshot"]
    if not os.path.isfile(pre_ply):
        raise RuntimeError(f"Online-final snapshot was not found: {pre_ply}")

    # Reload the exact pre-CR snapshot into local CUDA storage.  This avoids
    # touching any backend-produced CUDA IPC tensor after the producer exits.
    pre_gaussians = GaussianModel(runner.model_params.sh_degree, config=config)
    pre_gaussians.load_ply(pre_ply)
    torch.cuda.synchronize()

    original_eval_rendering(
        runner.frontend.cameras,
        pre_gaussians,
        runner.dataset,
        save_dir,
        runner.pipeline_params,
        runner.background,
        kf_indices=[],
        iteration="online_final",
        interval=1,
        skip_keyframes=False,
        save_images=bool(args.save_images),
        mask_nonzero=False,
    )
    torch.cuda.synchronize()
    del pre_gaussians
    torch.cuda.empty_cache()

    with open(
        os.path.join(save_dir, "psnr", "online_final", "per_frame_metrics.json"),
        "r",
        encoding="utf-8",
    ) as f:
        pre_per_frame = json.load(f)

    pre_train_rows, pre_test_rows = split_rows(
        pre_per_frame, args.holdout_every, args.holdout_offset
    )
    post_train_rows, post_test_rows = split_rows(
        post_per_frame, args.holdout_every, args.holdout_offset
    )

    pre_train = summarize_rows(pre_train_rows)
    pre_test = summarize_rows(pre_test_rows)
    post_train = summarize_rows(post_train_rows)
    post_test = summarize_rows(post_test_rows)

    total_frames = len(pre_train_rows) + len(pre_test_rows)
    max_map = largest_contiguous_map(runner.dataset, runner.frontend.cameras)
    online_fps = float(timing_result["streaming_fps"])
    online_time_sec = float(timing_result["end_to_end_without_evaluation_sec"])
    offline_time_sec = float(offline_timing["color_refinement_sec"])
    gaussian_count = int(timing_result["final_gaussian_count"])

    pre_row = make_row(
        "w/o CR",
        sequence_name,
        total_frames,
        max_map,
        pre_train,
        pre_test,
        trajectory_result,
        online_fps,
        online_time_sec,
        0.0,
        gaussian_count,
    )
    post_row = make_row(
        "+CR",
        sequence_name,
        total_frames,
        max_map,
        post_train,
        post_test,
        trajectory_result,
        online_fps,
        online_time_sec,
        offline_time_sec,
        gaussian_count,
    )
    paired_rows = [pre_row, post_row]

    write_rows(os.path.join(save_dir, "paired_benchmark_rows.csv"), paired_rows)
    # Keep the legacy single-row file as the final (+CR) map for compatibility.
    write_rows(os.path.join(save_dir, "benchmark_row.csv"), [post_row])

    summary = {
        "sequence": sequence_name,
        "protocol": "one online run -> exact snapshot -> released color refinement -> paired evaluation",
        "split": {
            "holdout_every": int(args.holdout_every),
            "holdout_offset": int(args.holdout_offset),
            "train_frames": len(pre_train_rows),
            "test_frames": len(pre_test_rows),
        },
        "max_map": max_map,
        "timing": timing_result,
        "offline_timing": offline_timing,
        "trajectory": trajectory_result,
        "online_final": {
            "train_rendering": pre_train,
            "test_rendering": pre_test,
            "benchmark_row": pre_row,
        },
        "post_color_refinement": {
            "train_rendering": post_train,
            "test_rendering": post_test,
            "benchmark_row": post_row,
        },
        "metric_evaluation_excluded_from_system_time": True,
        "cuda_ipc_safe_eval": True,
    }
    with open(
        os.path.join(save_dir, "paired_holdout_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    Log(
        "Paired benchmark w/o CR",
        f'{sequence_name} | Train={pre_row["Train PSNR"]:.3f}/'
        f'{pre_row["Train SSIM"]:.4f}/{pre_row["Train LPIPS"]:.4f} | '
        f'Test={pre_row["Test PSNR"]:.3f}/{pre_row["Test SSIM"]:.4f}/'
        f'{pre_row["Test LPIPS"]:.4f} | Time={online_time_sec:.1f}s',
        tag="Eval",
    )
    Log(
        "Paired benchmark +CR",
        f'{sequence_name} | Train={post_row["Train PSNR"]:.3f}/'
        f'{post_row["Train SSIM"]:.4f}/{post_row["Train LPIPS"]:.4f} | '
        f'Test={post_row["Test PSNR"]:.3f}/{post_row["Test SSIM"]:.4f}/'
        f'{post_row["Test LPIPS"]:.4f} | Online={online_time_sec:.1f}s | '
        f'CR={offline_time_sec:.1f}s | Total={online_time_sec + offline_time_sec:.1f}s',
        tag="Eval",
    )
    Log("Paired benchmark done.")


if __name__ == "__main__":
    main()
