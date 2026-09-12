import json
import os
import re
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset_router import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
from utils.trajectory_eval import (
    evaluate_pose_files,
    save_frame_indexed_trajectories,
)


def apply_sequence_override(config, sequence):
    if sequence is None:
        return
    if config["Dataset"]["type"] != "tartanair_stereo":
        raise ValueError("--sequence is currently supported only for tartanair_stereo")

    sequence = sequence.upper()
    if re.fullmatch(r"S[EH]\d{3}", sequence) is None:
        raise ValueError(
            f"Invalid TartanAir stereo sequence '{sequence}'. "
            "Expected names such as SE000 or SH007."
        )

    current_path = config["Dataset"]["dataset_path"].rstrip(os.sep)
    config["Dataset"]["dataset_path"] = os.path.join(
        os.path.dirname(current_path), sequence
    )
    config["Dataset"].pop("pose_file", None)


def apply_range_override(config, range_spec):
    """Apply an inclusive START-END range from the CLI.

    Example: --range 100-299 processes exactly 200 source frames.
    Internally Dataset.end_idx remains exclusive.
    """
    if range_spec is None:
        return

    match = re.fullmatch(r"(\d+)-(\d+)", range_spec.strip())
    if match is None:
        raise ValueError("--range must use inclusive START-END syntax, e.g. 0-199")

    start = int(match.group(1))
    end = int(match.group(2))
    if end < start:
        raise ValueError(f"Invalid range {range_spec}: END must be >= START")

    config["Dataset"]["start_idx"] = start
    config["Dataset"]["end_idx"] = end + 1
    config["Dataset"]["frame_stride"] = 1


def dataset_range_label(config):
    start = int(config["Dataset"].get("start_idx", 0))
    end_exclusive = int(config["Dataset"].get("end_idx", -1))
    if end_exclusive < 0:
        return f"range_{start:06d}_end"
    return f"range_{start:06d}_{end_exclusive - 1:06d}"


class SLAM:
    def __init__(
        self,
        config,
        save_dir=None,
        experiment_mode=None,
        color_refinement=False,
    ):
        total_wall_start = time.perf_counter()

        self.config = config
        self.save_dir = save_dir
        self.experiment_mode = experiment_mode
        self.color_refinement = color_refinement

        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        initialization_sec = time.perf_counter() - total_wall_start
        torch.cuda.synchronize()
        streaming_wall_start = time.perf_counter()

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])
        torch.cuda.synchronize()

        streaming_wall_time_sec = time.perf_counter() - streaming_wall_start
        end_to_end_without_evaluation_sec = time.perf_counter() - total_wall_start
        n_frames = len(self.frontend.cameras)
        fps = n_frames / streaming_wall_time_sec if streaming_wall_time_sec > 0 else 0.0
        final_gaussian_count = int(self.frontend.gaussians.get_xyz.shape[0])

        timing_result = {
            "processed_frames": int(n_frames),
            "initialization_sec": float(initialization_sec),
            "streaming_wall_time_sec": float(streaming_wall_time_sec),
            "streaming_fps": float(fps),
            "end_to_end_without_evaluation_sec": float(
                end_to_end_without_evaluation_sec
            ),
            "final_gaussian_count": final_gaussian_count,
        }

        Log("Initialization time", initialization_sec, tag="Eval")
        Log("Streaming wall time", streaming_wall_time_sec, tag="Eval")
        Log("Streaming FPS", fps, tag="Eval")
        Log("Final Gaussian count", f"{final_gaussian_count:,}", tag="Eval")
        Log(
            "End-to-end time without final evaluation",
            end_to_end_without_evaluation_sec,
            tag="Eval",
        )

        if self.experiment_mode in {"eval", "timing"}:
            trajectory_paths = save_frame_indexed_trajectories(
                self.frontend.cameras, self.dataset, self.save_dir
            )
            with open(
                os.path.join(self.save_dir, "timing.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(timing_result, f, indent=4)

            if self.experiment_mode == "eval":
                trajectory_result = evaluate_pose_files(
                    trajectory_paths["estimated"],
                    trajectory_paths["ground_truth"],
                    output_dir=os.path.join(self.save_dir, "trajectory"),
                    correct_scale=self.monocular,
                    save_plot=True,
                )
                ate = trajectory_result["ate_rmse_m"]
                Log("RMSE ATE [m]", ate, tag="Eval")
                Log(
                    "Pose coverage",
                    f'{trajectory_result["matched_pose_count"]}/'
                    f'{trajectory_result["gt_pose_count"]} '
                    f'({trajectory_result["coverage"] * 100.0:.2f}%)',
                    tag="Eval",
                )

                self.gaussians = self.frontend.gaussians

                if self.color_refinement:
                    while not frontend_queue.empty():
                        frontend_queue.get()
                    backend_queue.put(["color_refinement"])
                    while True:
                        if frontend_queue.empty():
                            time.sleep(0.01)
                            continue
                        data = frontend_queue.get()
                        if data[0] == "sync_backend" and frontend_queue.empty():
                            self.gaussians = data[1]
                            break

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=[],
                    iteration="final",
                    interval=1,
                    skip_keyframes=False,
                    save_images=True,
                    mask_nonzero=False,
                )
                save_gaussians(
                    self.gaussians,
                    self.save_dir,
                    "final",
                    final=True,
                )

                experiment_summary = {
                    "mode": "eval",
                    "timing": timing_result,
                    "trajectory": trajectory_result,
                    "rendering": rendering_result,
                    "final_gaussian_count": final_gaussian_count,
                    "color_refinement": bool(self.color_refinement),
                }
                with open(
                    os.path.join(self.save_dir, "experiment_summary.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(experiment_summary, f, indent=4)

                wandb.log(
                    {
                        "ATE": ate,
                        "PSNR": rendering_result["mean_psnr"],
                        "SSIM": rendering_result["mean_ssim"],
                        "LPIPS": rendering_result["mean_lpips"],
                        "FPS": fps,
                        "Gaussian Count": final_gaussian_count,
                    }
                )

        elif self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ate = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            wandb.log(
                {
                    "ATE": ate,
                    "PSNR": rendering_result["mean_psnr"],
                    "SSIM": rendering_result["mean_ssim"],
                    "LPIPS": rendering_result["mean_lpips"],
                    "FPS": fps,
                    "Gaussian Count": final_gaussian_count,
                }
            )

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    parser = ArgumentParser(description="MonoGS experiment runner")
    parser.add_argument("--config", type=str, required=True)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--eval",
        action="store_true",
        help="One final ATE plus per-frame PSNR/SSIM/LPIPS and rendered images",
    )
    mode_group.add_argument(
        "--timing-only",
        action="store_true",
        help="Keep original MonoGS runtime behavior, but skip final ATE/rendering/PLY",
    )

    parser.add_argument(
        "--range",
        dest="range_spec",
        type=str,
        default=None,
        help="Inclusive source-frame range, e.g. 0-199 or 100-299",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="TartanAir stereo sequence override, e.g. SE000 or SH007",
    )
    parser.add_argument(
        "--color-refinement",
        action="store_true",
        help="Run MonoGS offline color refinement before final rendering metrics",
    )

    args = parser.parse_args(sys.argv[1:])
    mp.set_start_method("spawn")

    config = load_config(args.config)
    apply_sequence_override(config, args.sequence)
    apply_range_override(config, args.range_spec)

    experiment_mode = None
    if args.eval:
        experiment_mode = "eval"
    elif args.timing_only:
        experiment_mode = "timing"

    if experiment_mode is not None:
        Log(f"Running MonoGS in {experiment_mode.upper()} experiment mode")
        config["Results"]["save_results"] = True
        config["Results"]["use_gui"] = False
        config["Results"]["eval_rendering"] = experiment_mode == "eval"
        config["Results"]["use_wandb"] = False
        config["Results"]["save_trj"] = False

    save_dir = None
    should_save = config["Results"]["save_results"] or experiment_mode is not None
    if should_save:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        dataset_type = config["Dataset"]["type"]
        sequence_name = os.path.basename(
            config["Dataset"]["dataset_path"].rstrip(os.sep)
        )
        dataset_tag = f"{dataset_type}_{sequence_name}"
        range_tag = dataset_range_label(config)
        save_dir = os.path.join(
            config["Results"]["save_dir"],
            dataset_tag,
            range_tag,
            current_datetime,
        )
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)

        with open(os.path.join(save_dir, "config.yml"), "w", encoding="utf-8") as file:
            yaml.dump(config, file)
        Log("saving results in " + save_dir)

    if experiment_mode is not None:
        config["Results"]["save_results"] = False
        config["Results"]["save_trj"] = False

    run = wandb.init(
        project="MonoGS",
        name=(
            f"{os.path.splitext(args.config)[0]}_"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        ),
        config=config,
        mode=None if config["Results"]["use_wandb"] else "disabled",
    )
    wandb.define_metric("frame_idx")
    wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(
        config,
        save_dir=save_dir,
        experiment_mode=experiment_mode,
        color_refinement=args.color_refinement,
    )
    slam.run()
    wandb.finish()
    Log("Done.")
