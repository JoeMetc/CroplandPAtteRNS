#####
# Author: Joseph Metcalfe
# Creation Date: 26/02/2025
# Purpose: Dynamic loading of model versions
#####

from models import PAtteRNS
from models import TSViT
from models import UTAE
from models import UNet3D

# Registry maps base names to their modules
MODEL_REGISTRY = {
    "PAtteRNS":   PAtteRNS,
    "UNet3D":        UNet3D,
    "TSViT":         TSViT,
    "UTAE":          UTAE,
}

def loadModel(config, device):
    modelConfig = config["MODEL"]
    base    = modelConfig["architecture"]          # e.g. "PAtteRNS"
    variant = modelConfig.get("variant", "")   # e.g. "Alt", "Temp", ""
    
    fullClassName = f"{base}{variant}"            # e.g. "PAtteRNS"
    module = MODEL_REGISTRY[base] # load module containing that class

    print(f"Using Model {fullClassName}")
    
    modelClass = getattr(module, fullClassName) # grabs the class dynamically
    return modelClass(modelConfig).to(device)