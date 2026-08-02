import torch
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY
)

from data.split import split_patients
from data.dataset import BraTSDataset
from data.transform import train_transform, val_transform

from models.effnet_unet import build_effnet_unet
from models.losses import BCEDiceLoss

from utils.metrics import (
    dice_score,
    iou_score,
    precision,
    recall,
)

from utils.train_utils import train_model


def main():

    # ----------------------------------------
    # Device
    # ----------------------------------------

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print(f"Using device: {DEVICE}")

    # ----------------------------------------
    # Dataset Split
    # ----------------------------------------

    train_patients, val_patients, test_patients = split_patients()

    print(f"Train Patients : {len(train_patients)}")

    print(f"Val Patients   : {len(val_patients)}")

    print(f"Test Patients  : {len(test_patients)}")

    # ----------------------------------------
    # Datasets
    # ----------------------------------------

    train_dataset = BraTSDataset(

        train_patients,

        transform=train_transform,

    )

    val_dataset = BraTSDataset(

        val_patients,

        transform=val_transform,

    )

    # ----------------------------------------
    # DataLoaders
    # ----------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

    )

    # ----------------------------------------
    # Model
    # ----------------------------------------

    model = build_effnet_unet()

    # ----------------------------------------
    # Optimizer
    # ----------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

    )

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    metrics = {

        "Dice": dice_score,

        "IoU": iou_score,

        "Precision": precision,

        "Recall": recall,

    }

    # ----------------------------------------
    # Train
    # ----------------------------------------

    train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        optimizer=optimizer,

        criterion=BCEDiceLoss(),

        metrics=metrics,

        device=DEVICE,

        epochs=EPOCHS,

        model_name="effnet_unet",

    )


if __name__ == "__main__":

    main()