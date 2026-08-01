from pathlib import Path
import torch

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR = BASE_DIR / "results"

LOG_DIR = BASE_DIR / "logs"

# ==========================================================
# Dataset
# ==========================================================

IMAGE_SIZE = (224, 224)

NUM_CLASSES = 1

MODALITIES = [
    "flair",
    "t1",
    "t1ce",
    "t2",
]

# ==========================================================
# Dataset Split
# ==========================================================

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

SEED = 42

# ==========================================================
# Training
# ==========================================================

BATCH_SIZE = 4

NUM_WORKERS = 4

PIN_MEMORY = True

SHUFFLE = True

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

# ==========================================================
# Scheduler
# ==========================================================

LR_FACTOR = 0.5

LR_PATIENCE = 5

MIN_LR = 1e-6

# ==========================================================
# Early Stopping
# ==========================================================

EARLY_STOPPING_PATIENCE = 10

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# Checkpoints
# ==========================================================

SAVE_BEST_ONLY = True

# ==========================================================
# Visualization
# ==========================================================

NUM_VISUALIZATIONS = 5