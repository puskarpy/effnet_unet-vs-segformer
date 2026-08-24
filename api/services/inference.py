from pathlib import Path

import cv2
import numpy as np
import torch

from api.config import RESULTS_DIR

from .model_loader import (
    efficientnet,
    segformer,
    device,
)

from api.services.preprocessing import (
    load_patient_volumes,
    preprocess_slice,
)

from api.services.visualization import (
    save_flair,
    save_mask,
    save_overlay,
)


def process_patient(patient_dir):
    """
    Run both segmentation models on one representative
    MRI slice and save exactly four images per model:

        flair.png
        ground_truth.png
        prediction.png
        overlay.png

    The representative slice is selected as the slice
    containing the largest ground-truth tumor area.
    """

    patient_dir = Path(patient_dir)
    patient_name = patient_dir.name

    # --------------------------------------------------
    # Load MRI volumes
    # --------------------------------------------------

    volumes = load_patient_volumes(
        patient_dir
    )

    flair_volume = volumes["flair"]
    ground_truth_volume = volumes["seg"]

    depth = flair_volume.shape[2]

    # --------------------------------------------------
    # Find slice with largest tumor area
    # --------------------------------------------------

    tumor_sizes = []

    for slice_idx in range(depth):

        tumor_area = np.sum(
            ground_truth_volume[:, :, slice_idx] > 0
        )

        tumor_sizes.append(tumor_area)

    best_slice_idx = int(
        np.argmax(tumor_sizes)
    )

    print(
        f"Selected slice: {best_slice_idx}"
        f" | Tumor pixels: "
        f"{tumor_sizes[best_slice_idx]}"
    )

    # --------------------------------------------------
    # Preprocess selected slice
    # --------------------------------------------------

    (
        image_tensor,
        flair,
        ground_truth,
    ) = preprocess_slice(
        volumes,
        best_slice_idx,
    )

    image_tensor = image_tensor.to(device)

    # --------------------------------------------------
    # Run both models
    # --------------------------------------------------

    with torch.no_grad():

        efficientnet_prediction = (
            efficientnet(image_tensor)
        )

        segformer_prediction = (
            segformer(image_tensor)
        )

    # --------------------------------------------------
    # Convert predictions to numpy
    # --------------------------------------------------

    efficientnet_prediction = (
        efficientnet_prediction[0, 0]
        .cpu()
        .numpy()
    )

    segformer_prediction = (
        segformer_prediction[0, 0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------
    # Convert probabilities to binary masks
    # --------------------------------------------------

    efficientnet_prediction = (
        efficientnet_prediction >= 0.5
    ).astype(np.uint8)

    segformer_prediction = (
        segformer_prediction >= 0.5
    ).astype(np.uint8)

    # --------------------------------------------------
    # Resize predictions back to original MRI size
    # --------------------------------------------------

    height, width = flair.shape

    efficientnet_prediction = cv2.resize(
        efficientnet_prediction,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    segformer_prediction = cv2.resize(
        segformer_prediction,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Ground truth returned by preprocessing is
    # currently 224x224, so resize it back too.
    ground_truth = cv2.resize(
        ground_truth,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    # --------------------------------------------------
    # Create model result directories
    # --------------------------------------------------

    efficientnet_dir = (
        RESULTS_DIR
        / "efficientnet"
        / patient_name
    )

    segformer_dir = (
        RESULTS_DIR
        / "segformer"
        / patient_name
    )

    efficientnet_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    segformer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save EfficientNet results
    # --------------------------------------------------

    save_flair(
        flair,
        efficientnet_dir / "flair.png",
    )

    save_mask(
        ground_truth,
        efficientnet_dir / "ground_truth.png",
    )

    save_mask(
        efficientnet_prediction,
        efficientnet_dir / "prediction.png",
    )

    save_overlay(
        flair,
        ground_truth,
        efficientnet_prediction,
        efficientnet_dir / "overlay.png",
    )

    # --------------------------------------------------
    # Save SegFormer results
    # --------------------------------------------------

    save_flair(
        flair,
        segformer_dir / "flair.png",
    )

    save_mask(
        ground_truth,
        segformer_dir / "ground_truth.png",
    )

    save_mask(
        segformer_prediction,
        segformer_dir / "prediction.png",
    )

    save_overlay(
        flair,
        ground_truth,
        segformer_prediction,
        segformer_dir / "overlay.png",
    )

    # --------------------------------------------------
    # Return information for FastAPI
    # --------------------------------------------------

    return {
        "patient": patient_name,
        "slice": best_slice_idx,
        "efficientnet": {
            "flair": str(
                efficientnet_dir / "flair.png"
            ),
            "ground_truth": str(
                efficientnet_dir / "ground_truth.png"
            ),
            "prediction": str(
                efficientnet_dir / "prediction.png"
            ),
            "overlay": str(
                efficientnet_dir / "overlay.png"
            ),
        },
        "segformer": {
            "flair": str(
                segformer_dir / "flair.png"
            ),
            "ground_truth": str(
                segformer_dir / "ground_truth.png"
            ),
            "prediction": str(
                segformer_dir / "prediction.png"
            ),
            "overlay": str(
                segformer_dir / "overlay.png"
            ),
        },
    }