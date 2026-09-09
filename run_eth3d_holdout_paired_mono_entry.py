import slam as monogs_slam
import run_tartanair_holdout_paired as paired_runner
import run_tartanair_holdout_paired_mono_entry as mono_guard
from run_tartanair_holdout import HoldoutFrontEnd
from utils.dataset_eth3d import apply_eth3d_sequence_override


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_eth3d_sequence_override
    monogs_slam.evaluate_pose_files = mono_guard._evaluate_pose_files_se3
    monogs_slam.wandb.log = lambda *args, **kwargs: None

    # Reuse the same narrow degenerate-depth guard as the TartanAir monocular
    # paired benchmark. It only falls back to released MonoGS depth=None when
    # the confident rendered-depth set is too small for a defined std().
    HoldoutFrontEnd.add_new_keyframe = mono_guard._add_new_keyframe_with_degenerate_depth_guard

    paired_runner.main()
