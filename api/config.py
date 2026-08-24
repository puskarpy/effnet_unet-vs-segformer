from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "api" / "results"

EFFICIENTNET_CHECKPOINT = CHECKPOINTS_DIR / "best_effnet_unet.pth"
SEGFORMER_CHECKPOINT = CHECKPOINTS_DIR / "best_segformer.pth"

CORS_ORIGINS = [
    "http://localhost:5173",
]