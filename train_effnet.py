import torch
from torch.utils.data import DataLoader

from configs.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY
)

from data.split import split_dataset
from data.dataset import BraTSDataset
from data.transform import get_train_transforms, get_val_transforms

from models.effnet_unet import build_effnet_unet
from models.losses import bce_dice_loss

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

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print(f"Using device: {device}")

    # ----------------------------------------
    # Dataset Split
    # ----------------------------------------

    train_patients, val_patients, test_patients = split_dataset()

    print(f"Train Patients : {len(train_patients)}")

    print(f"Val Patients   : {len(val_patients)}")

    print(f"Test Patients  : {len(test_patients)}")

    # ----------------------------------------
    # Datasets
    # ----------------------------------------

    train_dataset = BraTSDataset(

        train_patients,

        transform=get_train_transforms(),

    )

    val_dataset = BraTSDataset(

        val_patients,

        transform=get_val_transforms(),

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

        criterion=bce_dice_loss,

        metrics=metrics,

        device=device,

        epochs=EPOCHS,

        model_name="effnet_unet",

    )


if __name__ == "__main__":

    main()