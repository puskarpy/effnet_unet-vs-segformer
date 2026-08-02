from torch.utils.data import DataLoader
import torch

from configs.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SHUFFLE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from data.dataset import BraTSDataset
from data.split import split_patients
from data.transform import train_transform, val_transform

from models.segformer import build_segformer
from models.losses import BCEDiceLoss

from utils.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)

from utils.train_utils import train_model


def main():

    print("Splitting dataset...")

    train_patients, val_patients, _ = split_patients()

    train_dataset = BraTSDataset(
        train_patients,
        transform=train_transform,
    )

    val_dataset = BraTSDataset(
        val_patients,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
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

    print("Building SegFormer...")

    model = build_segformer().to(DEVICE)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    metrics = {
        "Dice": dice_score,
        "IoU": iou_score,
        "Precision": precision_score,
        "Recall": recall_score,
    }

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        device=DEVICE,
        epochs=EPOCHS,
        model_name="segformer",
    )


if __name__ == "__main__":
    main()