from utils.dataset import load_dataset as load_builtin_dataset
from utils.dataset_tartanair import TartanAirMonocularDataset, TartanAirStereoDataset


def load_dataset(args, path, config):
    dataset_type = config["Dataset"]["type"]
    if dataset_type == "tartanair_stereo":
        return TartanAirStereoDataset(args, path, config)
    if dataset_type == "tartanair_mono":
        return TartanAirMonocularDataset(args, path, config)
    return load_builtin_dataset(args, path, config)
