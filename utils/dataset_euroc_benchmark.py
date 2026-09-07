import os

import numpy as np
import yaml

from utils.dataset import EurocDataset


class EuRoCBenchmarkDataset(EurocDataset):
    """EuRoC stereo dataset wrapper for benchmark-side temporal subsampling.

    The underlying image loading, rectification, SGBM depth construction, pose
    conversion, and MonoGS parameters remain the released EuRoC implementation.
    This wrapper only applies an explicit raw-frame stride/range after the
    released parser has associated images with ground-truth poses.
    """

    def __init__(self, args, path, config):
        dataset_cfg = config["Dataset"]
        dataset_path = os.path.abspath(os.path.expanduser(dataset_cfg["dataset_path"]))
        dataset_cfg["dataset_path"] = dataset_path

        self._validate_layout_and_calibration(dataset_cfg)

        start_idx = int(dataset_cfg.get("start_idx", 0))
        end_idx = int(dataset_cfg.get("end_idx", -1))
        frame_stride = int(dataset_cfg.get("frame_stride", 1))

        if start_idx < 0:
            raise ValueError("EuRoC start_idx must be >= 0")
        if frame_stride <= 0:
            raise ValueError("EuRoC frame_stride must be > 0")
        if end_idx >= 0 and end_idx <= start_idx:
            raise ValueError(
                f"Invalid EuRoC range: start_idx={start_idx}, end_idx={end_idx}"
            )

        # Released MonoGS EuRoC parser handles image/GT association and starts at
        # Dataset.start_idx. We keep that path intact, then apply the benchmark
        # stride to the already-associated stereo pairs and poses.
        super().__init__(args, path, config)

        remaining_count = int(self.num_imgs)
        raw_total = start_idx + remaining_count
        stop_abs = raw_total if end_idx < 0 else min(end_idx, raw_total)
        stop_rel = stop_abs - start_idx

        relative_indices = list(range(0, stop_rel, frame_stride))
        if not relative_indices:
            raise RuntimeError(
                "EuRoC benchmark slicing produced no frames: "
                f"start={start_idx}, end={end_idx}, stride={frame_stride}, "
                f"raw_total={raw_total}"
            )

        self.color_paths = [self.color_paths[i] for i in relative_indices]
        self.color_paths_r = [self.color_paths_r[i] for i in relative_indices]
        self.poses = [self.poses[i] for i in relative_indices]
        self.num_imgs = len(relative_indices)

        # frame_indices are the original EuRoC camera-frame indices and are used
        # in trajectory/per-frame output. benchmark_frame_indices are contiguous
        # indices after temporal subsampling and are used for the 8:2 split and
        # MaxMap continuity, i.e. split is applied AFTER stride=5.
        self.frame_indices = [start_idx + i for i in relative_indices]
        self.benchmark_frame_indices = list(range(self.num_imgs))
        self.raw_frame_count = raw_total
        self.frame_stride = frame_stride

        every = int(dataset_cfg.get("holdout_every", 5))
        offset = int(dataset_cfg.get("holdout_offset", 4))
        test_count = sum(
            1 for i in self.benchmark_frame_indices if i % every == offset
        )
        train_count = self.num_imgs - test_count

        gt_path = os.path.join(
            dataset_path, "mav0", "state_groundtruth_estimate0", "data.csv"
        )
        print(
            "MonoGS: loaded EuRoC benchmark sequence "
            f"{dataset_path} | raw_frames={raw_total}, start={start_idx}, "
            f"end={stop_abs}, stride={frame_stride}, retained={self.num_imgs}, "
            f"train/test={train_count}/{test_count}, GT={gt_path}",
            flush=True,
        )

    @staticmethod
    def _validate_layout_and_calibration(dataset_cfg):
        root = os.path.abspath(os.path.expanduser(dataset_cfg["dataset_path"]))
        required = [
            os.path.join(root, "mav0", "cam0", "data"),
            os.path.join(root, "mav0", "cam1", "data"),
            os.path.join(root, "mav0", "cam0", "sensor.yaml"),
            os.path.join(root, "mav0", "cam1", "sensor.yaml"),
            os.path.join(root, "mav0", "state_groundtruth_estimate0", "data.csv"),
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "EuRoC sequence layout is incomplete. Missing:\n  - "
                + "\n  - ".join(missing)
            )

        calibration = dataset_cfg["Calibration"]
        expected_resolution = [int(calibration["width"]), int(calibration["height"])]

        # EuRoC ships calibration per camera. Validate only the raw sensor model;
        # rectified intrinsics/R matrices remain exactly those in MonoGS's official
        # mh02.yaml configuration.
        for camera_name in ("cam0", "cam1"):
            sensor_path = os.path.join(root, "mav0", camera_name, "sensor.yaml")
            with open(sensor_path, "r", encoding="utf-8") as f:
                sensor = yaml.safe_load(f)

            resolution = [int(v) for v in sensor.get("resolution", [])]
            if resolution and resolution != expected_resolution:
                raise ValueError(
                    f"{camera_name} resolution mismatch in {sensor_path}: "
                    f"sensor.yaml={resolution}, MonoGS={expected_resolution}"
                )

            intrinsics = sensor.get("intrinsics")
            if intrinsics is not None:
                raw = calibration[camera_name]["raw"]
                expected_intrinsics = np.array(
                    [raw["fx"], raw["fy"], raw["cx"], raw["cy"]], dtype=np.float64
                )
                actual_intrinsics = np.asarray(intrinsics, dtype=np.float64)
                if actual_intrinsics.shape != (4,) or not np.allclose(
                    actual_intrinsics, expected_intrinsics, rtol=0.0, atol=1e-6
                ):
                    raise ValueError(
                        f"{camera_name} intrinsics in {sensor_path} do not match "
                        "MonoGS's official EuRoC calibration: "
                        f"sensor.yaml={actual_intrinsics.tolist()}, "
                        f"MonoGS={expected_intrinsics.tolist()}"
                    )

            distortion = sensor.get("distortion_coefficients")
            if distortion is not None:
                raw = calibration[camera_name]["raw"]
                expected_distortion = np.array(
                    [raw["k1"], raw["k2"], raw["p1"], raw["p2"]],
                    dtype=np.float64,
                )
                actual_distortion = np.asarray(distortion, dtype=np.float64)
                if actual_distortion.shape != (4,) or not np.allclose(
                    actual_distortion, expected_distortion, rtol=0.0, atol=1e-6
                ):
                    raise ValueError(
                        f"{camera_name} distortion in {sensor_path} does not match "
                        "MonoGS's official EuRoC calibration: "
                        f"sensor.yaml={actual_distortion.tolist()}, "
                        f"MonoGS={expected_distortion.tolist()}"
                    )
