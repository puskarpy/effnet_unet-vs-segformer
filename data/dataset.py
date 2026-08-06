from pathlib import Path
import random

import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import IMAGE_SIZE


class BraTSDataset(Dataset):

    def __init__(
        self,
        patient_folders,
        transform=None,
        empty_slice_prob=0.2,
    ):

        self.transform = transform
        self.samples = []

        for patient in patient_folders:

            flair = nib.load(
                patient / f"{patient.name}_flair.nii"
            ).get_fdata()

            depth = flair.shape[2]

            mask = nib.load(
                patient / f"{patient.name}_seg.nii"
            ).get_fdata()

            for slice_idx in range(depth):

                m = mask[:, :, slice_idx]

                # Skip most empty slices
                if np.sum(m) == 0:
                    if random.random() > empty_slice_prob:
                        continue

                self.samples.append((patient, slice_idx))

        print(f"Total slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def normalize(self, image):

        image = image.astype(np.float32)

        mean = image.mean()
        std = image.std()

        if std > 0:
            image = (image - mean) / std

        return image

    def _load_slice(self, patient, slice_idx):

        flair = nib.load(
            patient / f"{patient.name}_flair.nii"
        ).get_fdata()[:, :, slice_idx]

        t1 = nib.load(
            patient / f"{patient.name}_t1.nii"
        ).get_fdata()[:, :, slice_idx]

        t1ce = nib.load(
            patient / f"{patient.name}_t1ce.nii"
        ).get_fdata()[:, :, slice_idx]

        t2 = nib.load(
            patient / f"{patient.name}_t2.nii"
        ).get_fdata()[:, :, slice_idx]

        mask = nib.load(
            patient / f"{patient.name}_seg.nii"
        ).get_fdata()[:, :, slice_idx]

        image = np.stack(
            [
                self.normalize(flair),
                self.normalize(t1),
                self.normalize(t1ce),
                self.normalize(t2),
            ],
            axis=-1,
        )

        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            IMAGE_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image = self.transform(image)

        return image, mask

    def get_slice(self, patient, slice_idx):

        return self._load_slice(patient, slice_idx)

    def __getitem__(self, idx):

        patient, slice_idx = self.samples[idx]

        return self._load_slice(patient, slice_idx)


    def get_sample_info(self, idx,):

        patient, slice_idx = self.samples[idx]

        return patient.name, slice_idx