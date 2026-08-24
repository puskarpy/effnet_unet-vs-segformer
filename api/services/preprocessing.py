from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch


IMAGE_SIZE = (224, 224)

MODALITIES = [
    "flair",
    "t1",
    "t1ce",
    "t2",
]


def normalize(image):
    image = image.astype(np.float32)

    mean = image.mean()
    std = image.std()

    if std > 0:
        image = (image - mean) / std

    return image


def load_patient_volumes(patient_dir):
    """
    Load MRI modalities and ground-truth segmentation.
    """

    patient_dir = Path(patient_dir)
    patient_name = patient_dir.name

    volumes = {}

    for modality in MODALITIES:

        path = patient_dir / f"{patient_name}_{modality}.nii"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing MRI file: {path}"
            )

        volumes[modality] = nib.load(path).get_fdata()

    # Ground truth
    seg_path = patient_dir / f"{patient_name}_seg.nii"

    if not seg_path.exists():
        raise FileNotFoundError(
            f"Missing segmentation file: {seg_path}"
        )

    volumes["seg"] = nib.load(seg_path).get_fdata()

    return volumes


def preprocess_slice(volumes, slice_idx):
    """
    Preprocess one MRI slice.

    Returns:
        image_tensor: (1, 4, 224, 224)
        flair: original Flair slice
        ground_truth: original GT slice
    """

    flair = normalize(
        volumes["flair"][:, :, slice_idx]
    )

    t1 = normalize(
        volumes["t1"][:, :, slice_idx]
    )

    t1ce = normalize(
        volumes["t1ce"][:, :, slice_idx]
    )

    t2 = normalize(
        volumes["t2"][:, :, slice_idx]
    )

    ground_truth = volumes["seg"][:, :, slice_idx]

    # Stack modalities
    image = np.stack(
        [
            flair,
            t1,
            t1ce,
            t2,
        ],
        axis=-1,
    )

    # Resize exactly like training
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR,
    )

    # Resize GT using nearest neighbor
    ground_truth = cv2.resize(
        ground_truth,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )

    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    # HWC -> CHW
    image_tensor = torch.from_numpy(
        image
    ).permute(2, 0, 1).float()

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return (
        image_tensor,
        flair,
        ground_truth,
    )