import glob
import os

import cv2
import numpy as np
import torch
import trimesh

from utils.dataset import StereoDataset


class TartanAirStereoParser:
    """Parser for the TartanAir v1 CVPR Visual SLAM stereo challenge sequences."""

    def __init__(
        self,
        input_folder,
        pose_file=None,
        start_idx=0,
        end_idx=-1,
        frame_stride=1,
    ):
        self.input_folder = os.path.abspath(os.path.expanduser(input_folder))
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx) if end_idx is not None else -1
        self.frame_stride = int(frame_stride)

        if self.start_idx < 0:
            raise ValueError("TartanAir start_idx must be >= 0")
        if self.frame_stride <= 0:
            raise ValueError("TartanAir frame_stride must be > 0")

        all_left = sorted(glob.glob(os.path.join(self.input_folder, "image_left", "*.png")))
        all_right = sorted(glob.glob(os.path.join(self.input_folder, "image_right", "*.png")))

        if not all_left:
            raise FileNotFoundError(
                f"No left images found under {self.input_folder}/image_left"
            )
        if len(all_left) != len(all_right):
            raise ValueError(
                "TartanAir stereo image count mismatch: "
                f"left={len(all_left)}, right={len(all_right)}"
            )

        # The challenge uses matching names such as 000000_left.png / 000000_right.png.
        for left_path, right_path in zip(all_left, all_right):
            left_id = os.path.basename(left_path).split("_")[0]
            right_id = os.path.basename(right_path).split("_")[0]
            if left_id != right_id:
                raise ValueError(
                    f"TartanAir stereo pair mismatch: {left_path} vs {right_path}"
                )

        total = len(all_left)
        stop = total if self.end_idx < 0 else min(self.end_idx, total)
        if self.start_idx >= stop:
            raise ValueError(
                f"Invalid TartanAir frame range [{self.start_idx}, {self.end_idx}) "
                f"for {total} frames"
            )

        self.indices = list(range(self.start_idx, stop, self.frame_stride))
        self.color_paths = [all_left[i] for i in self.indices]
        self.color_paths_r = [all_right[i] for i in self.indices]
        self.n_img = len(self.indices)

        self.pose_file = self._resolve_pose_file(pose_file)
        self.poses = self._load_poses(self.pose_file, total)

    def _resolve_pose_file(self, pose_file):
        if pose_file:
            path = os.path.abspath(os.path.expanduser(pose_file))
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Configured TartanAir pose_file does not exist: {path}"
                )
            return path

        sequence_name = os.path.basename(self.input_folder.rstrip(os.sep))
        dataset_root = os.path.dirname(os.path.dirname(self.input_folder))
        candidates = [
            os.path.join(self.input_folder, "pose_left.txt"),
            os.path.join(self.input_folder, f"{sequence_name}.txt"),
            os.path.join(dataset_root, "stereo_gt", f"{sequence_name}.txt"),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

        candidate_text = "\n  - ".join(candidates)
        raise FileNotFoundError(
            "TartanAir ground-truth pose file was not found. "
            "Set Dataset.pose_file explicitly, or place the released challenge GT at one "
            f"of these locations:\n  - {candidate_text}"
        )

    def _load_poses(self, pose_file, total_images):
        pose_data = np.atleast_2d(np.loadtxt(pose_file, dtype=np.float64))
        if pose_data.shape[1] != 7:
            raise ValueError(
                "TartanAir pose file must contain 7 values per row: "
                "tx ty tz qx qy qz qw"
            )
        if pose_data.shape[0] != total_images:
            raise ValueError(
                "TartanAir pose/image count mismatch before frame slicing: "
                f"poses={pose_data.shape[0]}, images={total_images}"
            )

        # TartanAir poses use the NED camera convention (x forward, y right, z down).
        # DROID-SLAM's official TartanAir evaluation converts NED to the conventional
        # camera xyz ordering with [y, z, x] for translation and quaternion vector parts.
        pose_data = pose_data[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = []
        for idx in self.indices:
            row = pose_data[idx]
            trans = row[:3]
            quat_xyzw = row[3:7]
            quat_wxyz = np.roll(quat_xyzw, 1)

            T_w_c = trimesh.transformations.quaternion_matrix(quat_wxyz)
            T_w_c[:3, 3] = trans
            poses.append(np.linalg.inv(T_w_c))

        return poses


class TartanAirStereoDataset(StereoDataset):
    """MonoGS stereo adapter for TartanAir v1 challenge data."""

    def __init__(self, args, path, config):
        super().__init__(args, path, config)

        dataset_cfg = config["Dataset"]
        calibration = dataset_cfg["Calibration"]
        stereo_cfg = dataset_cfg.get("StereoMatching", {})

        if "bf" in calibration:
            self.bf = float(calibration["bf"])
        elif "baseline" in calibration:
            self.bf = float(calibration["baseline"]) * float(self.fx)
        else:
            raise KeyError(
                "TartanAir stereo calibration requires either Calibration.bf "
                "or Calibration.baseline"
            )

        self.num_disparities = int(stereo_cfg.get("num_disparities", 64))
        self.block_size = int(stereo_cfg.get("block_size", 20))
        self.uniqueness_ratio = int(stereo_cfg.get("uniqueness_ratio", 40))
        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError("StereoMatching.num_disparities must be a positive multiple of 16")

        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
        )
        self.stereo_matcher.setUniquenessRatio(self.uniqueness_ratio)

        parser = TartanAirStereoParser(
            dataset_cfg["dataset_path"],
            pose_file=dataset_cfg.get("pose_file"),
            start_idx=dataset_cfg.get("start_idx", 0),
            end_idx=dataset_cfg.get("end_idx", -1),
            frame_stride=dataset_cfg.get("frame_stride", 1),
        )
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses
        self.pose_file = parser.pose_file
        self.frame_indices = parser.indices

        print(
            "MonoGS: loaded TartanAir stereo sequence "
            f"{dataset_cfg['dataset_path']} ({self.num_imgs} frames, "
            f"bf={self.bf:.3f}, GT={self.pose_file})"
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]
        pose = self.poses[idx]

        image = cv2.imread(color_path, cv2.IMREAD_GRAYSCALE)
        image_r = cv2.imread(color_path_r, cv2.IMREAD_GRAYSCALE)
        if image is None or image_r is None:
            raise FileNotFoundError(
                f"Failed to read TartanAir stereo pair: {color_path}, {color_path_r}"
            )

        if image.shape != (self.height, self.width):
            raise ValueError(
                f"Unexpected left image size {image.shape}; expected "
                f"({self.height}, {self.width})"
            )
        if image_r.shape != (self.height, self.width):
            raise ValueError(
                f"Unexpected right image size {image_r.shape}; expected "
                f"({self.height}, {self.width})"
            )

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)

        disparity = self.stereo_matcher.compute(image, image_r).astype(np.float32) / 16.0
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > 0.0)
        depth[valid] = self.bf / disparity[valid]

        # Preserve MonoGS's original stereo behavior: SGBM runs on grayscale and the
        # left grayscale image is replicated to three channels for tracking/mapping.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device, dtype=self.dtype)

        return image, depth, pose
