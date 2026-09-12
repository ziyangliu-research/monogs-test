import os

import slam as monogs_slam

# Reuse the benchmark-side sparse stereo insertion guard already used for the
# TartanAir stereo baseline. It only skips geometrically degenerate stereo point
# clouds before simple-knn and does not alter tracking/mapping parameters.
import run_tartanair_holdout_safe_entry  # noqa: F401
import run_tartanair_holdout_paired as paired_runner
from run_tartanair_holdout import HoldoutFrontEnd
from utils.dataset_euroc_benchmark import EuRoCBenchmarkDataset
from utils.dataset_router import load_dataset as default_load_dataset


EUROC_ROOT = "/home/shiyo/Desktop/Datasets/EuRoC/extracted"
EUROC_SEQUENCE_PATHS = {
    "MH02": os.path.join(
        EUROC_ROOT, "machine_hall", "machine_hall", "MH_02_easy"
    ),
    "V101": os.path.join(
        EUROC_ROOT, "vicon_room1", "vicon_room1", "V1_01_easy"
    ),
    "V201": os.path.join(
        EUROC_ROOT, "vicon_room2", "vicon_room2", "V2_01_easy"
    ),
    "MH05": os.path.join(
        EUROC_ROOT, "machine_hall", "machine_hall", "MH_05_difficult"
    ),
}


class EuRoCHoldoutFrontEnd(HoldoutFrontEnd):
    """Apply the 8:2 split after EuRoC temporal subsampling.

    All retained frames still execute released MonoGS tracking. Every fifth
    retained frame (offset 4) is excluded only from keyframe judgement/mapping.
    """

    def is_holdout_frame(self, local_idx):
        every = int(self.config["Dataset"].get("holdout_every", 0))
        if every <= 0:
            return False
        offset = int(self.config["Dataset"].get("holdout_offset", every - 1))
        benchmark_ids = getattr(self.dataset, "benchmark_frame_indices", None)
        split_id = int(local_idx) if benchmark_ids is None else int(benchmark_ids[local_idx])
        return split_id % every == offset


def euroc_split_rows(per_frame, holdout_every, holdout_offset):
    """Split metrics using retained-sequence local indices, not raw frame IDs."""
    train_rows, test_rows = [], []
    for row in per_frame:
        split_id = int(row["local_idx"])
        if split_id % holdout_every == holdout_offset:
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def euroc_largest_contiguous_map(dataset, cameras):
    """MaxMap continuity in the stride-5 retained sequence."""
    ids = sorted(int(i) for i in cameras.keys())
    if not ids:
        return {"frames": 0, "first_frame": None, "last_frame": None}

    best_start = best_end = ids[0]
    cur_start = cur_end = ids[0]
    for idx in ids[1:]:
        if idx == cur_end + 1:
            cur_end = idx
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = idx
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end

    source_ids = getattr(dataset, "frame_indices", None)
    result = {
        "frames": int(best_end - best_start + 1),
        "first_frame": int(best_start),
        "last_frame": int(best_end),
    }
    if source_ids is not None:
        result["first_source_frame"] = int(source_ids[best_start])
        result["last_source_frame"] = int(source_ids[best_end])
    return result


def apply_euroc_sequence_override(config, sequence):
    if sequence is None:
        return
    if config["Dataset"]["type"] != "euroc":
        raise ValueError(
            "EuRoC benchmark entry requires Dataset.type='euroc'; "
            f"got {config['Dataset']['type']!r}"
        )

    key = sequence.upper().replace("_", "").replace("-", "")
    aliases = {
        "MH02": "MH02",
        "MH2": "MH02",
        "V101": "V101",
        "V1O1": "V101",
        "V201": "V201",
        "V2O1": "V201",
        "MH05": "MH05",
        "MH5": "MH05",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported EuRoC sequence {sequence!r}. "
            "Expected one of: MH02, V101, V201, MH05"
        )

    label = aliases[key]
    config["Dataset"]["dataset_path"] = EUROC_SEQUENCE_PATHS[label]
    config["Dataset"]["sequence_label"] = label


def load_euroc_benchmark_dataset(args, path, config):
    if config["Dataset"]["type"] == "euroc":
        return EuRoCBenchmarkDataset(args, path, config)
    return default_load_dataset(args, path, config)


if __name__ == "__main__":
    monogs_slam.apply_sequence_override = apply_euroc_sequence_override
    monogs_slam.load_dataset = load_euroc_benchmark_dataset
    monogs_slam.wandb.log = lambda *args, **kwargs: None

    # Patch only benchmark protocol helpers. Core MonoGS FrontEnd/BackEnd logic
    # and official EuRoC parameters remain unchanged.
    paired_runner.HoldoutFrontEnd = EuRoCHoldoutFrontEnd
    paired_runner.split_rows = euroc_split_rows
    paired_runner.largest_contiguous_map = euroc_largest_contiguous_map

    paired_runner.main()
