#####
# Author: Joseph Metcalfe
# Purpose: Count model params
#####

import torch

from models import loadModel
from utils.configIO import yamlRead
import calflops


CONFIGFILE = "source/configs/PAtteRNS-MTLCC-48.yaml"
# CONFIGFILE = "source/configs/PAtteRNSAblation-MTLCC-48.yaml"
# CONFIGFILE = "source/configs/TSViT-MTLCC-48.yaml"
# CONFIGFILE = "source/configs/UNet3D-MTLCC-48.yaml"
# CONFIGFILE = "source/configs/UTAE-MTLCC-48.yaml"


def set_run_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    config = yamlRead(CONFIGFILE)

    device = set_run_device()

    model = loadModel(config, device)

    batchSize = 1

    # PAtteRNS
    T = config["MODEL"]["maxTemporalBands"]
    S = config["MODEL"]["imgSpectralChannels"] + 1
    HW = config["MODEL"]["imgSize"]
    inputShape = (batchSize, T, S, HW, HW)

    # TSViT
    # T = config["MODEL"]["max_seq_len"]
    # S = config["MODEL"]["num_channels"]
    # HW = config["MODEL"]["img_res"]
    # inputShape = (batchSize, T, S, HW, HW)

    # UNet3D
    # T = config["MODEL"]["maxTemporalBands"]
    # S = config["MODEL"]["imgSpectralChannels"] + 1
    # HW = config["MODEL"]["imgSize"]
    # inputShape = (batchSize, T, S, HW, HW)

    print(f"Trainable parameters: {count_parameters(model):,}")

    flops, macs, params = calflops.calculate_flops(model=model, 
                                      input_shape=inputShape,
                                      output_as_string=True,
                                      output_precision=3)
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    exit()

    # UTAE
    T = config["MODEL"]["maxTemporalBands"]
    S = config["MODEL"]["imgSpectralChannels"]
    HW = config["MODEL"]["imgSize"]

    sits_dummy = torch.zeros(batchSize, T, S, HW, HW)
    dates_dummy = torch.zeros(batchSize, T).long()

    flops, macs, params = calflops.calculate_flops(
        model=model,
        kwargs={"input": sits_dummy, "batch_positions": dates_dummy},
        output_as_string=True,
        output_precision=3
    )

    print(f"Trainable parameters: {count_parameters(model):,}")
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

    


if __name__ == "__main__":
    main()