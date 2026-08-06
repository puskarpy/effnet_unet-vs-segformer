import argparse
import time

import torch
from torch.utils.data import DataLoader

from configs.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from data.dataset import BraTSDataset
from data.split import split_patients
from data.transform import val_transform

from utils.inference import (
    load_model,
    predict,
)

from utils.metrics import (
    dice_score,
    iou_score,
    precision,
    recall,
)

from utils.save_metrics import save_metrics

def evaluate(

    model,

    dataloader,

    device,

    model_name,

):

    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_time = 0.0

    num_images = 0

    for batch_idx, (images, masks) in enumerate(dataloader):

        batch_size = images.size(0)

        for i in range(batch_size):

            image = images[i]
            mask = masks[i]

            start = time.perf_counter()

            prediction = predict(

                model,

                image,

                device,

            )

            end = time.perf_counter()

            total_time += end - start

            prediction = prediction.unsqueeze(0)

            total_dice += dice_score(
                prediction,
                mask,
            )

            total_iou += iou_score(
                prediction,
                mask,
            )

            total_precision += precision(
                prediction,
                mask,
            )

            total_recall += recall(
                prediction,
                mask,
            )

            
            num_images += 1

    avg_dice = total_dice / num_images
    avg_iou = total_iou / num_images
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_time = total_time / num_images

    save_metrics(

        model_name,

        avg_dice,

        avg_iou,

        avg_precision,

        avg_recall,

        avg_time,

    )
    print("\n==============================")
    print(f"{model_name.upper()} Evaluation Results")
    print("==============================\n")

    print(f"Dice Score     : {avg_dice:.4f}")
    print(f"IoU            : {avg_iou:.4f}")
    print(f"Precision      : {avg_precision:.4f}")
    print(f"Recall         : {avg_recall:.4f}")
    print(
        f"Inference Time : {(avg_time) * 1000:.2f} ms/image"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate trained model"
    )

    parser.add_argument(

        "--model",

        required=True,

        choices=[

            "effnet",

            "segformer",

        ],

    )

    parser.add_argument(

        "--save",

        action="store_true",

        help="Save predicted masks",

    )

    args = parser.parse_args()

    model = load_model(

        args.model,

        DEVICE,

    )

    _, _, test_patients = split_patients()

    test_dataset = BraTSDataset(

        patient_folders=test_patients,

        transform=val_transform,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

    )

    evaluate(

        model,

        test_loader,

        DEVICE,

        args.model
    )


if __name__ == "__main__":

    main()