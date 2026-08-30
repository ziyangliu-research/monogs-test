import numpy as np

import slam as monogs_slam
import run_tartanair_holdout_safe as safe_runner
from gaussian_splatting.scene.gaussian_model import GaussianModel


_original_extend_from_pcd_seq = GaussianModel.extend_from_pcd_seq


def _sparse_safe_extend_from_pcd_seq(
    self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
):
    """Skip only geometrically empty/sparse stereo Gaussian insertions.

    MonoGS creates a point cloud from the observed stereo depth and then applies
    Dataset.pcd_downsample before calling simple-knn's distCUDA2 to initialize
    Gaussian scales. distCUDA2 estimates local neighbour distances, so a point
    set with only a handful of retained points is not a meaningful input and can
    trigger an invalid CUDA launch on recent GPUs.

    This guard does not change tracking, keyframe selection, the keyframe window,
    backend mapping, densification, pruning, or the released 3 FPS pacing. It only
    avoids attempting to append Gaussians when the current stereo observation is
    too sparse to produce a valid post-downsample neighbour set.
    """
    sensor_type = self.config.get("Dataset", {}).get("sensor_type")
    if sensor_type == "stereo" and (not init) and depthmap is not None:
        depth = np.asarray(depthmap)
        valid = np.isfinite(depth) & (depth > 0.0) & (depth < 100.0)
        valid_count = int(valid.sum())
        downsample_factor = int(
            self.config["Dataset"].get("pcd_downsample", 1)
        )

        # simple-knn initializes scale from local neighbours. Keep at least four
        # expected post-downsample points (query point + three neighbours).
        min_source_points = max(1, downsample_factor) * 4
        if valid_count < min_source_points:
            expected = valid_count / float(max(1, downsample_factor))
            print(
                "MonoGS: skipping Gaussian insertion for sparse stereo keyframe "
                f"kf_id={kf_id}: valid_depth={valid_count}, "
                f"pcd_downsample={downsample_factor}, "
                f"expected_retained_points={expected:.3f} < 4",
                flush=True,
            )
            return None

    return _original_extend_from_pcd_seq(
        self,
        cam_info,
        kf_id=kf_id,
        init=init,
        scale=scale,
        depthmap=depthmap,
    )


# Apply at import time so multiprocessing spawn children receive the same guard.
GaussianModel.extend_from_pcd_seq = _sparse_safe_extend_from_pcd_seq


if __name__ == "__main__":
    # Benchmark output is written locally; avoid implicit W&B initialization from
    # the generic SLAM eval path.
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    safe_runner.main()
