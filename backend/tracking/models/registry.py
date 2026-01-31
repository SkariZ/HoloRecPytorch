from pathlib import Path
from typing import Callable, Dict, Any
import torch.nn as nn

from .unet_base import UNetBase        # your architecture
# from .unet_other import UNetOther    # optional later

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

def build_unet_base() -> nn.Module:
    return UNetBase(in_channels=1, out_channels=1)

UNET_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "UNet Base Model": {
        "builder": build_unet_base,
        "weights": str(WEIGHTS_DIR / "base_unet_model.pth"),
        "size_factor": 8,
    },
    # "UNet Other (v1)": {
    #     "builder": lambda: UNetOther(...),
    #     "weights": str(WEIGHTS_DIR / "unet_other_v1.pth"),
    #     "size_factor": 32,
    # },
}