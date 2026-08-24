from pathlib import Path

import cv2
import numpy as np


def normalize_to_uint8(image):
    """
    Convert an image to 0-255 uint8.
    """

    image = image.astype(np.float32)

    min_value = image.min()
    max_value = image.max()

    if max_value - min_value > 0:
        image = (
            (image - min_value)
            / (max_value - min_value)
            * 255
        )

    return image.astype(np.uint8)


def save_flair(flair, output_path):
    """
    Save Flair MRI image.
    """

    image = normalize_to_uint8(flair)

    cv2.imwrite(
        str(output_path),
        image,
    )


def save_mask(mask, output_path):
    """
    Save binary segmentation mask.
    """

    mask = (
        mask.astype(np.uint8) * 255
    )

    cv2.imwrite(
        str(output_path),
        mask,
    )


def save_overlay(
    flair,
    ground_truth,
    prediction,
    output_path,
):
    """
    Create an overlay showing:

        Ground Truth  -> green
        Prediction    -> red
        Overlap       -> yellow

    over the Flair MRI.
    """

    background = normalize_to_uint8(flair)

    overlay = cv2.cvtColor(
        background,
        cv2.COLOR_GRAY2BGR,
    )

    gt = ground_truth.astype(bool)
    pred = prediction.astype(bool)

    # Ground truth only -> green
    gt_only = gt & ~pred

    # Prediction only -> red
    pred_only = pred & ~gt

    # Both -> yellow
    overlap = gt & pred

    overlay[gt_only] = (
        0,
        255,
        0,
    )

    overlay[pred_only] = (
        0,
        0,
        255,
    )

    overlay[overlap] = (
        0,
        255,
        255,
    )

    cv2.imwrite(
        str(output_path),
        overlay,
    )