import slam as monogs_slam
import run_tartanair_holdout_safe as safe_runner
from run_tartanair_mono_holdout_safe import apply_tartanair_sequence_override


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_tartanair_sequence_override
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    safe_runner.main()
