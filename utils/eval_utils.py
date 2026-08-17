import csv
import json
import os

import matplotlib
matplotlib.use("Agg")

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from utils.trajectory_eval import source_frame_id


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \\[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    # Keep this simple so it remains compatible with both old and new matplotlib.
    gt_xyz = np.asarray(traj_ref.positions_xyz)
    est_xyz = np.asarray(traj_est_aligned.positions_xyz)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], label="GT")
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], label="Estimate")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"ATE RMSE: {ape_stat:.6f} m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    """Legacy MonoGS keyframe ATE evaluator kept for existing configs."""
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices=None,
    iteration="final",
    interval=5,
    skip_keyframes=True,
    save_images=False,
    mask_nonzero=True,
):
    """Render and evaluate frames from the final Gaussian map.

    For the unified experiment protocol use interval=1, skip_keyframes=False,
    save_images=True and mask_nonzero=False. The defaults preserve the released
    MonoGS behavior for legacy configs.
    """
    kf_indices = set(kf_indices or [])
    frame_ids = sorted(int(idx) for idx in frames.keys())
    if isinstance(iteration, int):
        frame_ids = [idx for idx in frame_ids if idx < iteration]
    frame_ids = frame_ids[:: max(1, int(interval))]

    psnr_array, ssim_array, lpips_array = [], [], []
    per_frame = []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)
    rendered_dir = os.path.join(psnr_save_dir, "rendered")
    gt_dir = os.path.join(psnr_save_dir, "gt")
    if save_images:
        mkdir_p(rendered_dir)
        mkdir_p(gt_dir)

    for idx in frame_ids:
        if skip_keyframes and idx in kf_indices:
            continue

        frame = frames[idx]
        gt_image, _, _ = dataset[idx]
        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        if mask_nonzero:
            mask = gt_image > 0
            psnr_score = psnr(
                (image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0)
            )
        else:
            psnr_score = psnr(image.unsqueeze(0), gt_image.unsqueeze(0))

        ssim_score = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        lpips_score = cal_lpips(image.unsqueeze(0), gt_image.unsqueeze(0))

        psnr_value = float(psnr_score.item())
        ssim_value = float(ssim_score.item())
        lpips_value = float(lpips_score.item())
        frame_id = source_frame_id(dataset, idx)

        psnr_array.append(psnr_value)
        ssim_array.append(ssim_value)
        lpips_array.append(lpips_value)
        per_frame.append(
            {
                "local_idx": int(idx),
                "frame_id": int(frame_id),
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
            }
        )

        if save_images:
            gt_u8 = (
                gt_image.detach().cpu().numpy().transpose((1, 2, 0)) * 255.0
            ).clip(0, 255).astype(np.uint8)
            pred_u8 = (
                image.detach().cpu().numpy().transpose((1, 2, 0)) * 255.0
            ).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(rendered_dir, f"{frame_id:06d}.png"),
                cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(gt_dir, f"{frame_id:06d}.png"),
                cv2.cvtColor(gt_u8, cv2.COLOR_RGB2BGR),
            )

    if not per_frame:
        raise RuntimeError("No frames were selected for rendering evaluation")

    output = {
        "num_frames": len(per_frame),
        "mean_psnr": float(np.mean(psnr_array)),
        "mean_ssim": float(np.mean(ssim_array)),
        "mean_lpips": float(np.mean(lpips_array)),
    }

    Log(
        f'mean psnr: {output["mean_psnr"]}, '
        f'ssim: {output["mean_ssim"]}, '
        f'lpips: {output["mean_lpips"]}, '
        f'frames: {output["num_frames"]}',
        tag="Eval",
    )

    with open(
        os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=4)

    with open(
        os.path.join(psnr_save_dir, "per_frame_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(per_frame, f, indent=4)

    with open(
        os.path.join(psnr_save_dir, "per_frame_metrics.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["local_idx", "frame_id", "psnr", "ssim", "lpips"]
        )
        writer.writeheader()
        writer.writerows(per_frame)

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
