import torch

import slam as monogs_slam
import run_tartanair_holdout_paired as paired_runner
from run_tartanair_holdout import HoldoutFrontEnd
from run_tartanair_mono_holdout_safe import apply_tartanair_sequence_override
from utils.logging_utils import Log


_original_evaluate_pose_files = monogs_slam.evaluate_pose_files
_original_add_new_keyframe = HoldoutFrontEnd.add_new_keyframe


def _evaluate_pose_files_se3(*args, **kwargs):
    """Force the paper benchmark to use SE(3) alignment for monocular ATE."""
    kwargs["correct_scale"] = False
    return _original_evaluate_pose_files(*args, **kwargs)


def _add_new_keyframe_with_degenerate_depth_guard(
    self, cur_frame_idx, depth=None, opacity=None, init=False
):
    """Use MonoGS's existing no-depth fallback for an undefined depth statistic.

    Released MonoGS estimates monocular keyframe depth statistics from pixels with
    depth > 0, opacity > 0.95, and valid RGB.  When that set contains fewer than
    two pixels, torch.std() is undefined and produces NaN, which can propagate to
    the generated depth map and then to the CUDA point-cloud insertion path.

    This benchmark-side guard changes behavior only for that degenerate case.  It
    reuses MonoGS's existing depth=None fallback rather than introducing a new
    depth estimator or threshold.
    """
    if self.monocular and depth is not None and opacity is not None:
        viewpoint = self.cameras[cur_frame_idx]
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        valid = torch.isfinite(depth) & (depth > 0)
        valid = torch.logical_and(valid, torch.isfinite(opacity) & (opacity > 0.95))
        valid = torch.logical_and(valid, valid_rgb.view(*depth.shape))
        valid_count = int(valid.count_nonzero().item())

        if valid_count < 2:
            Log(
                "Degenerate monocular keyframe depth",
                f"frame={cur_frame_idx}, confident_depth_pixels={valid_count}; "
                "using released MonoGS depth=None fallback",
                tag="Eval",
            )
            return _original_add_new_keyframe(
                self,
                cur_frame_idx,
                depth=None,
                opacity=None,
                init=init,
            )

    return _original_add_new_keyframe(
        self,
        cur_frame_idx,
        depth=depth,
        opacity=opacity,
        init=init,
    )


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_tartanair_sequence_override
    monogs_slam.evaluate_pose_files = _evaluate_pose_files_se3
    monogs_slam.wandb.log = lambda *args, **kwargs: None

    # HoldoutFrontEnd inherits MonoGS's original add_new_keyframe().  Patch only
    # this benchmark class so the released repository logic remains untouched.
    HoldoutFrontEnd.add_new_keyframe = _add_new_keyframe_with_degenerate_depth_guard

    paired_runner.main()
