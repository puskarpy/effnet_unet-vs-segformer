import torch

from api.config import (
    EFFICIENTNET_CHECKPOINT,
    SEGFORMER_CHECKPOINT,
)

from api.models.effnet_unet import build_effnet_unet
from api.models.segformer import build_segformer


def load_efficientnet(device):

    model = build_effnet_unet()

    checkpoint = torch.load(
        EFFICIENTNET_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def load_segformer(device):

    model = build_segformer()

    checkpoint = torch.load(
        SEGFORMER_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

efficientnet = load_efficientnet(device)
segformer = load_segformer(device)