from pathlib import Path

import nibabel as nib
import numpy as np


def save_prediction(
    prediction,
    reference_path,
    output_path,
):
    """
    Save a prediction volume as NIfTI using
    the reference MRI's affine and header.
    """

    reference = nib.load(reference_path)

    prediction = prediction.astype(np.uint8)

    output = nib.Nifti1Image(
        prediction,
        affine=reference.affine,
        header=reference.header,
    )

    nib.save(
        output,
        output_path,
    )

    return Path(output_path)