from pathlib import Path
import torch


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

IMAGE_DIR = DATASET_DIR / "images" / "val2017"

CAPTION_FILE = (
    DATASET_DIR
    / "annotations"
    / "annotations"
    / "captions_val2017.json"
)

EMBEDDING_FILE = (
    DATASET_DIR
    / "analysis_results"
    / "embeddings"
    / "caption_embeddings.npy"
)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# CGAN Configuration
# ---------------------------------------------------------

# ---------------------------------------------------------
# CGAN Categories
# ---------------------------------------------------------

CATEGORIES = [
    "car",
    "chair",
    "bottle",
    "bird",
    "boat"
]

IMAGE_SIZE = 64

IMAGE_CHANNELS = 3

NOISE_DIM = 100

TEXT_EMBEDDING_DIM = 768

BATCH_SIZE = 8

LEARNING_RATE = 0.0001

BETA1 = 0.5

BETA2 = 0.999

NUM_EPOCHS = 75



# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

MAX_SAMPLES = 2000
