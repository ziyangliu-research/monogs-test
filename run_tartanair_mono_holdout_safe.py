import os
import re

import slam as monogs_slam
import run_tartanair_holdout_safe as safe_runner


def apply_tartanair_sequence_override(config, sequence):
    if sequence is None:
        return
    if config["Dataset"]["type"] not in {"tartanair_stereo", "tartanair_mono"}:
        raise ValueError(
            "TartanAir --sequence override requires tartanair_stereo or tartanair_mono"
        )

    sequence = sequence.upper()
    if re.fullmatch(r"S[EH]\d{3}", sequence) is None:
        raise ValueError(
            f"Invalid TartanAir sequence '{sequence}'. Expected names such as SE000 or SH003."
        )

    current_path = config["Dataset"]["dataset_path"].rstrip(os.sep)
    config["Dataset"]["dataset_path"] = os.path.join(
        os.path.dirname(current_path), sequence
    )
    config["Dataset"].pop("pose_file", None)


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_tartanair_sequence_override
    safe_runner.main()
