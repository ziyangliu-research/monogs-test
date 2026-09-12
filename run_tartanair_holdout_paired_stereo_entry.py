import slam as monogs_slam

# Import applies the existing benchmark-side sparse stereo insertion guard used
# by the current TartanAir stereo evaluation path.  No model parameter changes.
import run_tartanair_holdout_safe_entry  # noqa: F401
import run_tartanair_holdout_paired as paired_runner


if __name__ == "__main__":
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    paired_runner.main()
