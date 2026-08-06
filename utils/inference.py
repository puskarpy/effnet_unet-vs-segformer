import torch

from models.effnet_unet import build_effnet_unet
from models.segformer import build_segformer

from configs.config import CHECKPOINT_DIR


def load_model(

    model_name,

    device,

    checkpoint=None,

):

    model_name = model_name.lower()

    if model_name == "effnet":

        model = build_effnet_unet()

        if checkpoint is None:

            checkpoint = (
                CHECKPOINT_DIR
                / "best_effnet_unet.pth"
            )

    elif model_name == "segformer":

        model = build_segformer()

        if checkpoint is None:

            checkpoint = (
                CHECKPOINT_DIR
                / "best_segformer.pth"
            )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    state_dict = torch.load(

        checkpoint,

        map_location=device,

    )

    model.load_state_dict(

        state_dict,

    )

    model.to(device)

    model.eval()

    return model


def predict(

    model,

    image,

    device,

    threshold=0.5,

):

    model.eval()

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(image)

    prediction = (

        prediction > threshold

    ).float()

    return prediction.squeeze(0).cpu()