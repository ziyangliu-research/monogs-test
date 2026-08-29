import slam as monogs_slam
import run_tartanair_holdout_safe as safe_runner


if __name__ == "__main__":
    # Benchmark output is written locally; avoid implicit W&B initialization from
    # the generic SLAM eval path.
    monogs_slam.wandb.log = lambda *args, **kwargs: None
    safe_runner.main()
