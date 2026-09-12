import json
import os

import cv2
import numpy as np
import torch
import trimesh

from utils.dataset import MonocularDataset, StereoDataset


ETH3D_SEQUENCES = {
    "mannequin_face_1",
    "einstein_1",
    "sofa_3",
    "plant_scene_3",
}

# ETH3D's official SLAM evaluator uses 1/75 s as the default maximum timespan
# between two ground-truth measurements that may be interpolated. Image
# timestamps that fall into larger GT holes are excluded from trajectory
# evaluation rather than forcing a long-gap interpolation.
ETH3D_OFFICIAL_MAX_GT_INTERPOLATION_TIMESPAN_SEC = 1.0 / 75.0


def apply_eth3d_sequence_override(config, sequence):
    if sequence is None:
        return
    dataset_type = config["Dataset"]["type"]
    if dataset_type not in {"eth3d_stereo", "eth3d_mono"}:
        raise ValueError(
            "ETH3D --sequence override requires eth3d_stereo or eth3d_mono"
        )
    if sequence not in ETH3D_SEQUENCES:
        raise ValueError(
            f"Unsupported ETH3D sequence '{sequence}'. Expected one of: "
            + ", ".join(sorted(ETH3D_SEQUENCES))
        )

    current_path = config["Dataset"]["dataset_path"].rstrip(os.sep)
    config["Dataset"]["dataset_path"] = os.path.join(
        os.path.dirname(current_path), sequence
    )


def _normalize_quat_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Invalid ETH3D quaternion")
    return q / n


def _slerp_xyzw(q0, q1, alpha):
    q0 = _normalize_quat_xyzw(q0)
    q1 = _normalize_quat_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quat_xyzw(q0 + alpha * (q1 - q0))
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * alpha
    s0 = np.sin(theta0 - theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    return _normalize_quat_xyzw(s0 * q0 + s1 * q1)


def _pose_c2w(trans, quat_xyzw):
    qx, qy, qz, qw = _normalize_quat_xyzw(quat_xyzw)
    T = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    return T


def _load_rectified_calibration(dataset_path):
    path = os.path.join(dataset_path, "calibration.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ETH3D calibration.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    K_left = np.asarray(calib["K_rectified_left"], dtype=np.float64)
    K_right = np.asarray(calib["K_rectified_right"], dtype=np.float64)
    if K_left.shape != (3, 3) or K_right.shape != (3, 3):
        raise ValueError("ETH3D rectified K must be 3x3")
    if not np.allclose(K_left, K_right, atol=1e-5, rtol=1e-6):
        raise ValueError("ETH3D rectified left/right intrinsics differ")

    width, height = [int(x) for x in calib["rectified_size"]]
    baseline = float(calib["baseline_rectified_m"])
    if width <= 0 or height <= 0 or baseline <= 0:
        raise ValueError(
            f"Invalid ETH3D rectified geometry: size={width}x{height}, baseline={baseline}"
        )
    return calib, K_left, K_right, width, height, baseline


def _stereo_calibration_dict(K_left, K_right, width, height, baseline):
    def camera(K):
        values = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
        }
        return {
            "raw": dict(values),
            "opt": dict(values),
            "R": {
                "rows": 3,
                "cols": 3,
                "data": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        }

    return {
        "baseline": float(baseline),
        "distorted": False,
        "width": int(width),
        "height": int(height),
        "cam0": camera(K_left),
        "cam1": camera(K_right),
    }


def _mono_calibration_dict(K, width, height):
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "width": int(width),
        "height": int(height),
        "distorted": False,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
    }


class ETH3DRectifiedParser:
    """Parser for pre-rectified ETH3D RGB stereo sequences.

    Input convention:
      image_left/          = rectified ETH3D camera 2 (physical left)
      image_right/         = rectified ETH3D camera 1 (physical right)
      calibration.json     = generated rectified intrinsics/baseline
      groundtruth_left.txt = rectified-left camera c2w pose
      timestamps.txt       = image filename stems / timestamps in seconds

    ETH3D ground truth may contain holes. To match the official ETH3D SLAM
    evaluator, a frame is trajectory-evaluable only if its GT pose can be
    interpolated between two measurements whose time separation does not exceed
    max_gt_interpolation_timespan_sec. Frames in larger holes remain fully valid
    SLAM/rendering inputs; they are merely excluded from ATE evaluation.
    """

    def __init__(
        self,
        input_folder,
        start_idx=0,
        end_idx=-1,
        frame_stride=1,
        max_gt_interpolation_timespan_sec=ETH3D_OFFICIAL_MAX_GT_INTERPOLATION_TIMESPAN_SEC,
        require_right=True,
    ):
        self.input_folder = os.path.abspath(os.path.expanduser(input_folder))
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx) if end_idx is not None else -1
        self.frame_stride = int(frame_stride)
        self.max_gt_interpolation_timespan_sec = float(
            max_gt_interpolation_timespan_sec
        )

        if self.start_idx < 0:
            raise ValueError("ETH3D start_idx must be >= 0")
        if self.frame_stride <= 0:
            raise ValueError("ETH3D frame_stride must be > 0")
        if self.max_gt_interpolation_timespan_sec <= 0:
            raise ValueError(
                "ETH3D max_gt_interpolation_timespan_sec must be > 0"
            )

        timestamps_path = os.path.join(self.input_folder, "timestamps.txt")
        if not os.path.isfile(timestamps_path):
            raise FileNotFoundError(f"ETH3D timestamps.txt not found: {timestamps_path}")
        with open(timestamps_path, "r", encoding="utf-8") as f:
            stems = [line.strip() for line in f if line.strip()]
        if not stems:
            raise RuntimeError(f"ETH3D timestamps.txt is empty: {timestamps_path}")
        timestamps = np.asarray([float(s) for s in stems], dtype=np.float64)
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("ETH3D timestamps must be strictly increasing")

        all_left = [
            os.path.join(self.input_folder, "image_left", s + ".png") for s in stems
        ]
        all_right = [
            os.path.join(self.input_folder, "image_right", s + ".png") for s in stems
        ]
        for path in all_left:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"ETH3D left image missing: {path}")
        if require_right:
            for path in all_right:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"ETH3D right image missing: {path}")

        total = len(stems)
        stop = total if self.end_idx < 0 else min(self.end_idx, total)
        if self.start_idx >= stop:
            raise ValueError(
                f"Invalid ETH3D frame range [{self.start_idx}, {self.end_idx}) for {total} frames"
            )
        self.indices = list(range(self.start_idx, stop, self.frame_stride))
        self.timestamps = timestamps[self.indices]
        self.color_paths = [all_left[i] for i in self.indices]
        self.color_paths_r = [all_right[i] for i in self.indices] if require_right else []
        self.n_img = len(self.indices)

        self.pose_file = os.path.join(self.input_folder, "groundtruth_left.txt")
        self.poses, self.gt_valid_mask = self._load_poses(
            self.pose_file, self.timestamps
        )
        valid_count = int(np.count_nonzero(self.gt_valid_mask))
        invalid_count = int(self.n_img - valid_count)
        if valid_count < 2:
            raise ValueError(
                f"ETH3D sequence has fewer than two evaluable GT frames: {valid_count}"
            )
        if invalid_count:
            print(
                "MonoGS: ETH3D sparse GT: "
                f"{valid_count}/{self.n_img} image timestamps are ATE-evaluable; "
                f"{invalid_count} frames fall outside valid GT interpolation spans "
                f"(max span={self.max_gt_interpolation_timespan_sec:.9f}s) and "
                "will remain in SLAM/rendering but be excluded from ATE.",
                flush=True,
            )

    def _load_poses(self, pose_file, image_timestamps):
        if not os.path.isfile(pose_file):
            raise FileNotFoundError(f"ETH3D groundtruth_left.txt not found: {pose_file}")
        data = np.atleast_2d(np.loadtxt(pose_file, dtype=np.float64))
        if data.shape[1] != 8:
            raise ValueError(
                "ETH3D groundtruth_left.txt must contain 8 columns: "
                "timestamp tx ty tz qx qy qz qw"
            )
        order = np.argsort(data[:, 0])
        data = data[order]
        gt_ts = data[:, 0]
        if np.any(np.diff(gt_ts) <= 0):
            raise ValueError("ETH3D GT timestamps must be strictly increasing")
        if len(gt_ts) < 2:
            raise ValueError("ETH3D ground truth must contain at least two poses")

        poses = []
        gt_valid = []
        for ts in image_timestamps:
            # Match ETH3D's official ComputePosesAtTimestamps(): choose the first
            # GT measurement strictly after the image timestamp and interpolate
            # from the immediately preceding measurement. Start/end extrapolation
            # and intervals exceeding the official maximum interpolation timespan
            # are not valid for trajectory evaluation.
            next_idx = int(np.searchsorted(gt_ts, ts, side="right"))
            valid = False

            if 0 < next_idx < len(gt_ts):
                prev_idx = next_idx - 1
                t0 = float(gt_ts[prev_idx])
                t1 = float(gt_ts[next_idx])
                gap = t1 - t0
                if gap > 0 and gap <= self.max_gt_interpolation_timespan_sec:
                    alpha = float(np.clip((ts - t0) / gap, 0.0, 1.0))
                    trans = (
                        (1.0 - alpha) * data[prev_idx, 1:4]
                        + alpha * data[next_idx, 1:4]
                    )
                    quat = _slerp_xyzw(
                        data[prev_idx, 4:8], data[next_idx, 4:8], alpha
                    )
                    valid = True
                else:
                    # Placeholder GT for MonoGS Camera.R_gt/T_gt only. Those fields
                    # are not used by the tracking/mapping algorithm; this frame is
                    # excluded from trajectory evaluation through gt_valid_mask.
                    nearest_idx = (
                        prev_idx
                        if abs(ts - t0) <= abs(t1 - ts)
                        else next_idx
                    )
                    trans = data[nearest_idx, 1:4]
                    quat = data[nearest_idx, 4:8]
            elif next_idx == 0:
                trans = data[0, 1:4]
                quat = data[0, 4:8]
            else:
                trans = data[-1, 1:4]
                quat = data[-1, 4:8]

            T_w_c = _pose_c2w(trans, quat)
            poses.append(np.linalg.inv(T_w_c))
            gt_valid.append(valid)

        return poses, np.asarray(gt_valid, dtype=bool)


class ETH3DStereoDataset(StereoDataset):
    """MonoGS stereo adapter for rectified ETH3D RGB pairs.

    The released MonoGS stereo depth recipe is preserved: StereoSGBM with
    minDisparity=0, numDisparities=64, blockSize=20 and uniquenessRatio=40 by
    default. SGBM operates on grayscale images. Unlike EuRoC, ETH3D is RGB, so
    the left image supplied to MonoGS tracking/mapping is explicitly converted
    from OpenCV BGR to RGB instead of being discarded to grayscale.
    """

    def __init__(self, args, path, config):
        dataset_cfg = config["Dataset"]
        dataset_path = os.path.abspath(os.path.expanduser(dataset_cfg["dataset_path"]))
        _, K_left, K_right, width, height, baseline = _load_rectified_calibration(
            dataset_path
        )
        dataset_cfg["Calibration"] = _stereo_calibration_dict(
            K_left, K_right, width, height, baseline
        )
        super().__init__(args, path, config)

        stereo_cfg = dataset_cfg.get("StereoMatching", {})
        self.bf = float(baseline) * float(self.fx)
        self.num_disparities = int(stereo_cfg.get("num_disparities", 64))
        self.block_size = int(stereo_cfg.get("block_size", 20))
        self.uniqueness_ratio = int(stereo_cfg.get("uniqueness_ratio", 40))
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError(
                "StereoMatching.num_disparities must be a positive multiple of 16"
            )
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
        )
        self.stereo_matcher.setUniquenessRatio(self.uniqueness_ratio)

        parser = ETH3DRectifiedParser(
            dataset_path,
            start_idx=dataset_cfg.get("start_idx", 0),
            end_idx=dataset_cfg.get("end_idx", -1),
            frame_stride=dataset_cfg.get("frame_stride", 1),
            max_gt_interpolation_timespan_sec=dataset_cfg.get(
                "max_gt_interpolation_timespan_sec",
                ETH3D_OFFICIAL_MAX_GT_INTERPOLATION_TIMESPAN_SEC,
            ),
            require_right=True,
        )
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses
        self.gt_valid_mask = parser.gt_valid_mask
        self.pose_file = parser.pose_file
        self.frame_indices = parser.indices
        self.timestamps = parser.timestamps

        valid_gt = int(np.count_nonzero(self.gt_valid_mask))
        print(
            "MonoGS: loaded ETH3D rectified stereo sequence "
            f"{dataset_path} ({self.num_imgs} frames, {self.width}x{self.height}, "
            f"bf={self.bf:.6f}, ATE_GT={valid_gt}/{self.num_imgs}, "
            f"GT={self.pose_file})"
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]
        pose = self.poses[idx]

        left_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(color_path_r, cv2.IMREAD_COLOR)
        if left_bgr is None or right_bgr is None:
            raise FileNotFoundError(
                f"Failed to read ETH3D stereo pair: {color_path}, {color_path_r}"
            )
        if left_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Unexpected ETH3D left image size {left_bgr.shape[:2]}; "
                f"expected ({self.height}, {self.width})"
            )
        if right_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Unexpected ETH3D right image size {right_bgr.shape[:2]}; "
                f"expected ({self.height}, {self.width})"
            )

        # Rectified ETH3D images are already in the target camera domain. Keep
        # SGBM grayscale-only, but preserve RGB for MonoGS photometric mapping.
        left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        disparity = (
            self.stereo_matcher.compute(left_gray, right_gray).astype(np.float32)
            / 16.0
        )
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > 0.0)
        depth[valid] = self.bf / disparity[valid]

        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        image = (
            torch.from_numpy(left_rgb.copy() / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device, dtype=self.dtype)
        return image, depth, pose


class ETH3DMonocularDataset(MonocularDataset):
    """Left-camera RGB monocular adapter for the same rectified ETH3D sequences."""

    def __init__(self, args, path, config):
        dataset_cfg = config["Dataset"]
        dataset_path = os.path.abspath(os.path.expanduser(dataset_cfg["dataset_path"]))
        _, K_left, _, width, height, _ = _load_rectified_calibration(dataset_path)
        dataset_cfg["Calibration"] = _mono_calibration_dict(K_left, width, height)
        super().__init__(args, path, config)

        parser = ETH3DRectifiedParser(
            dataset_path,
            start_idx=dataset_cfg.get("start_idx", 0),
            end_idx=dataset_cfg.get("end_idx", -1),
            frame_stride=dataset_cfg.get("frame_stride", 1),
            max_gt_interpolation_timespan_sec=dataset_cfg.get(
                "max_gt_interpolation_timespan_sec",
                ETH3D_OFFICIAL_MAX_GT_INTERPOLATION_TIMESPAN_SEC,
            ),
            require_right=False,
        )
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.poses = parser.poses
        self.gt_valid_mask = parser.gt_valid_mask
        self.pose_file = parser.pose_file
        self.frame_indices = parser.indices
        self.timestamps = parser.timestamps

        valid_gt = int(np.count_nonzero(self.gt_valid_mask))
        print(
            "MonoGS: loaded ETH3D rectified monocular sequence "
            f"{dataset_path} ({self.num_imgs} left RGB frames, "
            f"{self.width}x{self.height}, ATE_GT={valid_gt}/{self.num_imgs}, "
            f"GT={self.pose_file})"
        )