import slam as monogs_slam

# Reuse the existing benchmark-side sparse stereo insertion guard. It only
# skips geometrically degenerate point-cloud insertions and leaves tracking,
# keyframe selection, mapping, pruning, and CR unchanged.
import run_tartanair_holdout_safe_entry  # noqa: F401
import run_tartanair_holdout_paired as paired_runner
from utils.dataset_eth3d import apply_eth3d_sequence_override


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_eth3d_sequence_override
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    paired_runner.main()
