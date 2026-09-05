import slam as monogs_slam
import run_tartanair_holdout_paired as paired_runner
from run_tartanair_mono_holdout_safe import apply_tartanair_sequence_override


_original_evaluate_pose_files = monogs_slam.evaluate_pose_files


def _evaluate_pose_files_se3(*args, **kwargs):
    """Force the paper benchmark to use SE(3) alignment for monocular ATE."""
    kwargs["correct_scale"] = False
    return _original_evaluate_pose_files(*args, **kwargs)


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_tartanair_sequence_override
    monogs_slam.evaluate_pose_files = _evaluate_pose_files_se3
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    paired_runner.main()
