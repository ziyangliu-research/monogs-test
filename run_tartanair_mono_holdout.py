import os
import re

import slam as monogs_slam
import run_tartanair_holdout as holdout_runner


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
    # Reuse the same 80/20 benchmark implementation while allowing the
    # monocular TartanAir dataset type. MonoGS core frontend/backend code is unchanged.
    monogs_slam.apply_sequence_override = apply_tartanair_sequence_override
    holdout_runner.main()
