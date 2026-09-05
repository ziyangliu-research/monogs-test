import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D


def _w2c_from_camera(frame, ground_truth=False):
    pose = np.eye(4, dtype=np.float64)
    if ground_truth:
        R = frame.R_gt
        T = frame.T_gt
    else:
        R = frame.R
        T = frame.T
    pose[:3, :3] = R.detach().cpu().numpy()
    pose[:3, 3] = T.detach().cpu().numpy()
    return pose


def _c2w_to_pose7(T_c2w):
    quat_wxyz = trimesh.transformations.quaternion_from_matrix(T_c2w)
    return np.array(
        [
            T_c2w[0, 3],
            T_c2w[1, 3],
            T_c2w[2, 3],
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
        ],
        dtype=np.float64,
    )


def _pose7_to_c2w(pose7):
    tx, ty, tz, qx, qy, qz, qw = pose7
    T = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
    T[:3, 3] = [tx, ty, tz]
    return T


def source_frame_id(dataset, local_idx):
    frame_indices = getattr(dataset, "frame_indices", None)
    if frame_indices is None:
        return int(local_idx)
    return int(frame_indices[local_idx])


def save_frame_indexed_trajectories(frames, dataset, save_dir):
    """Save estimated and GT C2W trajectories with original dataset frame IDs.

    Format per row:
        frame_id tx ty tz qx qy qz qw

    Using original frame IDs makes partial trajectories from other systems directly
    comparable by intersecting the available frame IDs.
    """
    trajectory_dir = os.path.join(save_dir, "trajectory")
    os.makedirs(trajectory_dir, exist_ok=True)
    est_path = os.path.join(trajectory_dir, "trajectory_est.txt")
    gt_path = os.path.join(trajectory_dir, "trajectory_gt.txt")

    local_ids = sorted(int(idx) for idx in frames.keys())
    header = "# frame_id tx ty tz qx qy qz qw\n"

    with open(est_path, "w", encoding="utf-8") as f_est, open(
        gt_path, "w", encoding="utf-8"
    ) as f_gt:
        f_est.write(header)
        f_gt.write(header)
        for local_idx in local_ids:
            frame = frames[local_idx]
            frame_id = source_frame_id(dataset, local_idx)
            est = _c2w_to_pose7(np.linalg.inv(_w2c_from_camera(frame, ground_truth=False)))
            gt = _c2w_to_pose7(np.linalg.inv(_w2c_from_camera(frame, ground_truth=True)))
            f_est.write(
                f"{frame_id:d} " + " ".join(f"{v:.9f}" for v in est) + "\n"
            )
            f_gt.write(
                f"{frame_id:d} " + " ".join(f"{v:.9f}" for v in gt) + "\n"
            )

    return {
        "estimated": est_path,
        "ground_truth": gt_path,
        "num_saved": len(local_ids),
    }


def load_frame_indexed_trajectory(path):
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    data = np.atleast_2d(data)
    if data.shape[1] != 8:
        raise ValueError(
            f"Trajectory file must have 8 columns "
            f"(frame_id tx ty tz qx qy qz qw): {path}"
        )
    frame_ids = data[:, 0].astype(np.int64)
    poses = {
        int(frame_id): _pose7_to_c2w(row[1:])
        for frame_id, row in zip(frame_ids, data)
    }
    return poses


def evaluate_pose_files(
    estimated_path,
    ground_truth_path,
    output_dir=None,
    correct_scale=False,
    save_plot=True,
):
    """Evaluate ATE on frame IDs present in both trajectory files.

    Stereo/RGB-D evaluation should use correct_scale=False. For a monocular system,
    set correct_scale=True when a Sim(3) scale correction is desired.
    """
    est_by_id = load_frame_indexed_trajectory(estimated_path)
    gt_by_id = load_frame_indexed_trajectory(ground_truth_path)
    matched_ids = sorted(set(est_by_id).intersection(gt_by_id))

    if len(matched_ids) < 2:
        raise ValueError(
            "At least two matched poses are required for ATE evaluation; "
            f"got {len(matched_ids)}"
        )

    poses_est = [est_by_id[idx] for idx in matched_ids]
    poses_gt = [gt_by_id[idx] for idx in matched_ids]
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=correct_scale
    )

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est_aligned))
    ate_rmse = float(ape_metric.get_statistic(metrics.StatisticsType.rmse))
    stats = {k: float(v) for k, v in ape_metric.get_all_statistics().items()}

    gt_count = len(gt_by_id)
    est_count = len(est_by_id)
    matched_count = len(matched_ids)
    result = {
        "ate_rmse_m": ate_rmse,
        "correct_scale": bool(correct_scale),
        "gt_pose_count": gt_count,
        "estimated_pose_count": est_count,
        "matched_pose_count": matched_count,
        "coverage": float(matched_count / gt_count) if gt_count else 0.0,
        "first_matched_frame": int(matched_ids[0]),
        "last_matched_frame": int(matched_ids[-1]),
        "ape_statistics": stats,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "trajectory_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, indent=4)

        if save_plot:
            gt_xyz = np.asarray(traj_ref.positions_xyz)
            est_xyz = np.asarray(traj_est_aligned.positions_xyz)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], label="GT")
            ax.plot(est_xyz[:, 0], est_xyz[:, 1], label="Estimate")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(
                f"ATE RMSE: {ate_rmse:.6f} m | "
                f"coverage: {matched_count}/{gt_count}"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "trajectory_xy.png"), dpi=120)
            plt.close(fig)

    return result
