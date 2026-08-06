import argparse

import torch

from pathlib import Path

from data.dataset import BraTSDataset
from data.transform import val_transform

from utils.inference import (
    load_model,
    predict,
)
from configs.config import DATASET_DIR

from utils.visualize import (
    show_prediction,
    save_prediction,
)


def main():

    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation Prediction"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "effnet",
            "segformer",
        ],
        help="Model to use",
    )

    parser.add_argument(
        "--patient",
        required=True,
        help="Patient folder name",
    )

    parser.add_argument(
        "--slice",
        type=int,
        default=60,
        help="Slice index",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save prediction images",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device : {device}")

    print(f"Loading {args.model} model...")

    model = load_model(
        args.model,
        device,
    )

    patient = DATASET_DIR / args.patient

    dataset = BraTSDataset(
        patient_folders=[patient],
        transform=val_transform,
    )

    image, gt = dataset.get_slice(
        patient,
        args.slice,
    )

    prediction = predict(
        model,
        image,
        device,
    )

    show_prediction(
        image=image,
        prediction=prediction,
        ground_truth=gt,
    )

    if args.save:

        save_prediction(
            image=image,
            prediction=prediction,
            save_dir=Path("results") / args.model / patient.name,
            filename=f"{patient.name}_slice_{args.slice}",
        )

        print("\nPrediction saved to results/")

    print("\nDone.")


if __name__ == "__main__":
    main()