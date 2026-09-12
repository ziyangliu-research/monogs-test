from utils.dataset import load_dataset as load_builtin_dataset
from utils.dataset_eth3d import ETH3DMonocularDataset, ETH3DStereoDataset
from utils.dataset_tartanair import TartanAirMonocularDataset, TartanAirStereoDataset


def load_dataset(args, path, config):
    dataset_type = config["Dataset"]["type"]
    if dataset_type == "tartanair_stereo":
        return TartanAirStereoDataset(args, path, config)
    if dataset_type == "tartanair_mono":
        return TartanAirMonocularDataset(args, path, config)
    if dataset_type == "eth3d_stereo":
        return ETH3DStereoDataset(args, path, config)
    if dataset_type == "eth3d_mono":
        return ETH3DMonocularDataset(args, path, config)
    return load_builtin_dataset(args, path, config)
