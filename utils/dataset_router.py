from utils.dataset import load_dataset as load_builtin_dataset
from utils.dataset_tartanair import TartanAirStereoDataset


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tartanair_stereo":
        return TartanAirStereoDataset(args, path, config)
    return load_builtin_dataset(args, path, config)
