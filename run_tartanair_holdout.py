import csv
import json
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml

import wandb
import slam as monogs_slam
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils
from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.trajectory_eval import evaluate_pose_files, source_frame_id


class HoldoutFrontEnd(monogs_slam.FrontEnd):
    """MonoGS FrontEnd with mapping excluded for deterministic holdout frames.

    Holdout frames still execute the original tracking path. After tracking they
    are cleaned up exactly like ordinary non-keyframes, but they do not enter
    keyframe judgement, window insertion, or backend mapping.
    """

    def is_holdout_frame(self, local_idx):
        every = int(self.config["Dataset"].get("holdout_every", 0))
        if every <= 0:
            return False
        offset = int(self.config["Dataset"].get("holdout_offset", every - 1))
        if hasattr(self.dataset, "frame_indices"):
            frame_id = int(self.dataset.frame_indices[local_idx])
        else:
            frame_id = int(local_idx)
        return frame_id % every == offset

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Original MonoGS tracking path: holdout frames are not skipped here.
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # Benchmark-only guard. The frame has completed pose tracking, but
                # it does not participate in keyframe judgement or map construction.
                if self.is_holdout_frame(cur_frame_idx):
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    toc.record()
                    torch.cuda.synchronize()
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # Preserve the released MonoGS 3 FPS keyframe pacing.
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break


def split_local_indices(dataset, cameras, every, offset):
    train, test = [], []
    for local_idx in sorted(int(i) for i in cameras.keys()):
        fid = source_frame_id(dataset, local_idx)
        if fid % every == offset:
            test.append(local_idx)
        else:
            train.append(local_idx)
    return train, test


def largest_contiguous_map(dataset, cameras):
    ids = sorted(source_frame_id(dataset, int(i)) for i in cameras.keys())
    if not ids:
        return {"frames": 0, "first_frame": None, "last_frame": None}

    best_start = best_end = ids[0]
    cur_start = cur_end = ids[0]
    for fid in ids[1:]:
        if fid == cur_end + 1:
            cur_end = fid
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = fid
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end

    return {
        "frames": int(best_end - best_start + 1),
        "first_frame": int(best_start),
        "last_frame": int(best_end),
    }


def write_benchmark_row(save_dir, sequence, max_map, train_result, test_result,
                        trajectory_result, fps, gaussian_count):
    row = {
        "Sequence": sequence,
        "MaxMap": int(max_map["frames"]),
        "Train PSNR": float(train_result["mean_psnr"]),
        "Train SSIM": float(train_result["mean_ssim"]),
        "Test PSNR": float(test_result["mean_psnr"]),
        "Test SSIM": float(test_result["mean_ssim"]),
        "ATE(m)": float(trajectory_result["ate_rmse_m"]),
        "FPS": float(fps),
        "Gaussians": int(gaussian_count),
    }
    path = os.path.join(save_dir, "benchmark_row.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def main():
    parser = ArgumentParser(description="MonoGS TartanAir 80/20 holdout benchmark")
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

    config = load_config(args.config)
    monogs_slam.apply_sequence_override(config, args.sequence)
    monogs_slam.apply_range_override(config, args.range_spec)
    config["Dataset"]["holdout_every"] = int(args.holdout_every)
    config["Dataset"]["holdout_offset"] = int(args.holdout_offset)

    # Core run uses MonoGS timing mode: no final rendering/evaluation is included
    # in the reported streaming FPS. The original frontend/backend schedule and
    # keyframe pacing are otherwise preserved.
    config["Results"]["save_results"] = True
    config["Results"]["use_gui"] = False
    config["Results"]["eval_rendering"] = False
    config["Results"]["use_wandb"] = False
    config["Results"]["save_trj"] = False

    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sequence_name = os.path.basename(config["Dataset"]["dataset_path"].rstrip(os.sep))
    dataset_tag = f'{config["Dataset"]["type"]}_{sequence_name}'
    range_tag = monogs_slam.dataset_range_label(config)
    split_tag = f"holdout_{args.holdout_every}_{args.holdout_offset}"
    save_dir = os.path.join(
        config["Results"]["save_dir"], dataset_tag, range_tag, split_tag, current_datetime
    )
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)

    with open(os.path.join(save_dir, "config.yml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    Log("saving results in " + save_dir)

    # Disable the old in-frontend result path after preserving the effective config.
    config["Results"]["save_results"] = False
    config["Results"]["save_trj"] = False

    monogs_slam.FrontEnd = HoldoutFrontEnd

    run = wandb.init(
        project="MonoGS",
        name=f"holdout_{sequence_name}_{current_datetime}",
        config=config,
        mode="disabled",
    )

    runner = monogs_slam.SLAM(
        config,
        save_dir=save_dir,
        experiment_mode="timing",
        color_refinement=False,
    )

    timing_path = os.path.join(save_dir, "timing.json")
    with open(timing_path, "r", encoding="utf-8") as f:
        timing_result = json.load(f)

    trajectory_dir = os.path.join(save_dir, "trajectory")
    trajectory_result = evaluate_pose_files(
        os.path.join(trajectory_dir, "trajectory_est.txt"),
        os.path.join(trajectory_dir, "trajectory_gt.txt"),
        output_dir=trajectory_dir,
        correct_scale=runner.monocular,
        save_plot=True,
    )

    train_indices, test_indices = split_local_indices(
        runner.dataset,
        runner.frontend.cameras,
        args.holdout_every,
        args.holdout_offset,
    )
    if not train_indices or not test_indices:
        raise RuntimeError(
            f"Invalid split: train={len(train_indices)}, test={len(test_indices)}"
        )

    Log(
        "Holdout split",
        f"train={len(train_indices)}, test={len(test_indices)}, "
        f"rule=frame_id % {args.holdout_every} == {args.holdout_offset}",
        tag="Eval",
    )

    gaussians = runner.frontend.gaussians
    train_frames = {idx: runner.frontend.cameras[idx] for idx in train_indices}
    test_frames = {idx: runner.frontend.cameras[idx] for idx in test_indices}

    train_result = eval_rendering(
        train_frames,
        gaussians,
        runner.dataset,
        save_dir,
        runner.pipeline_params,
        runner.background,
        kf_indices=[],
        iteration="train",
        interval=1,
        skip_keyframes=False,
        save_images=args.save_images,
        mask_nonzero=False,
    )
    test_result = eval_rendering(
        test_frames,
        gaussians,
        runner.dataset,
        save_dir,
        runner.pipeline_params,
        runner.background,
        kf_indices=[],
        iteration="test",
        interval=1,
        skip_keyframes=False,
        save_images=args.save_images,
        mask_nonzero=False,
    )

    save_gaussians(gaussians, save_dir, "final", final=True)
    gaussian_count = int(gaussians.get_xyz.shape[0])
    max_map = largest_contiguous_map(runner.dataset, runner.frontend.cameras)
    fps = float(timing_result["streaming_fps"])

    row = write_benchmark_row(
        save_dir,
        sequence_name,
        max_map,
        train_result,
        test_result,
        trajectory_result,
        fps,
        gaussian_count,
    )

    summary = {
        "sequence": sequence_name,
        "split": {
            "holdout_every": int(args.holdout_every),
            "holdout_offset": int(args.holdout_offset),
            "train_frames": len(train_indices),
            "test_frames": len(test_indices),
        },
        "max_map": max_map,
        "timing": timing_result,
        "trajectory": trajectory_result,
        "train_rendering": train_result,
        "test_rendering": test_result,
        "final_gaussian_count": gaussian_count,
        "benchmark_row": row,
    }
    with open(os.path.join(save_dir, "holdout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    Log(
        "Benchmark row",
        f'{sequence_name} | MaxMap={row["MaxMap"]} | '
        f'Train={row["Train PSNR"]:.3f}/{row["Train SSIM"]:.4f} | '
        f'Test={row["Test PSNR"]:.3f}/{row["Test SSIM"]:.4f} | '
        f'ATE={row["ATE(m)"]:.6f} m | FPS={row["FPS"]:.3f} | '
        f'Gaussians={row["Gaussians"]:,}',
        tag="Eval",
    )

    wandb.finish()
    Log("Done.")


if __name__ == "__main__":
    main()
