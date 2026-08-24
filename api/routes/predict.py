import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from api.services.inference import process_patient


router = APIRouter(
    prefix="/api",
    tags=["Prediction"],
)


@router.post("/predict")
async def predict(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    seg: UploadFile = File(...),
):
    """
    Run both EfficientNet and SegFormer
    on one patient.
    """

    patient_id = f"patient_{uuid4().hex[:8]}"

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="brain_tumor_"
        )
    )

    patient_dir = (
        temp_root / patient_id
    )

    patient_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # --------------------------------------------------
        # Save uploaded files
        # --------------------------------------------------

        files = {
            "flair": flair,
            "t1": t1,
            "t1ce": t1ce,
            "t2": t2,
            "seg": seg,
        }

        for modality, upload in files.items():

            destination = (
                patient_dir
                / f"{patient_id}_{modality}.nii"
            )

            with destination.open("wb") as buffer:
                shutil.copyfileobj(
                    upload.file,
                    buffer,
                )

        # --------------------------------------------------
        # Run inference
        # --------------------------------------------------

        result = process_patient(
            patient_dir
        )

        # --------------------------------------------------
        # Return image URLs
        # --------------------------------------------------

        return {
            "success": True,

            "patient": result["patient"],

            "selected_slice": result["slice"],

            "efficientnet": {
                "flair": (
                    "/results/efficientnet/"
                    f"{patient_id}/flair.png"
                ),

                "ground_truth": (
                    "/results/efficientnet/"
                    f"{patient_id}/ground_truth.png"
                ),

                "prediction": (
                    "/results/efficientnet/"
                    f"{patient_id}/prediction.png"
                ),

                "overlay": (
                    "/results/efficientnet/"
                    f"{patient_id}/overlay.png"
                ),
            },

            "segformer": {
                "flair": (
                    "/results/segformer/"
                    f"{patient_id}/flair.png"
                ),

                "ground_truth": (
                    "/results/segformer/"
                    f"{patient_id}/ground_truth.png"
                ),

                "prediction": (
                    "/results/segformer/"
                    f"{patient_id}/prediction.png"
                ),

                "overlay": (
                    "/results/segformer/"
                    f"{patient_id}/overlay.png"
                ),
            },
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}",
        )

    finally:

        # Remove temporary uploaded NIfTI files
        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )