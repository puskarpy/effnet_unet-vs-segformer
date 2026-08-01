from pathlib import Path
from sklearn.model_selection import train_test_split

from configs.config import (
    DATASET_PATH,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def get_patient_folders():
    """
    Returns all valid patient folders.
    """

    patients = sorted(
        DATASET_PATH.glob("BraTS20_Training_*")
    )

    valid_patients = []

    for patient in patients:

        required = [
            patient / f"{patient.name}_flair.nii",
            patient / f"{patient.name}_t1.nii",
            patient / f"{patient.name}_t1ce.nii",
            patient / f"{patient.name}_t2.nii",
            patient / f"{patient.name}_seg.nii",
        ]

        if all(file.exists() for file in required):
            valid_patients.append(patient)

        else:
            print(f"Skipping incomplete patient {patient.name}")

    return valid_patients


def split_patients():

    patients = get_patient_folders()

    train_patients, temp_patients = train_test_split(
        patients,
        train_size=TRAIN_SPLIT,
        random_state=SEED,
        shuffle=True,
    )

    val_size = VAL_SPLIT / (1 - TRAIN_SPLIT)

    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_size,
        random_state=SEED,
        shuffle=True,
    )

    return train_patients, val_patients, test_patients