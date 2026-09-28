"""
Text-GAN Configuration
======================

Extends the base CGAN config with text-encoder-specific
constants and COCO annotation paths.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------
# Add the parent cgan/ directory to sys.path so that
# the base config.py can be imported directly.
# ---------------------------------------------------------

_CGAN_DIR = Path(__file__).resolve().parent.parent
if str(_CGAN_DIR) not in sys.path:
    sys.path.insert(0, str(_CGAN_DIR))

from config import (                    # noqa: E402
    CATEGORIES,
    NOISE_DIM,
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    BATCH_SIZE,
    LEARNING_RATE,
    BETA1,
    BETA2,
    NUM_EPOCHS,
)


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

IMAGE_DIR = DATASET_DIR / "images" / "val2017"

INSTANCES_FILE = (
    DATASET_DIR
    / "annotations"
    / "annotations"
    / "instances_val2017.json"
)

CAPTION_FILE = (
    DATASET_DIR
    / "annotations"
    / "annotations"
    / "captions_val2017.json"
)


# ---------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------

TEXT_ENCODER_NAME = "distilbert-base-uncased"

TEXT_EMBED_DIM = 768

MAX_CAPTION_LENGTH = 64


# ---------------------------------------------------------
# Embedding Cache
# ---------------------------------------------------------

TEXT_GAN_DIR = Path(__file__).resolve().parent

CACHE_DIR = TEXT_GAN_DIR / "cache"


# ---------------------------------------------------------
# COCO Category Mapping
#
# Maps our 5 category names to their COCO category IDs
# (from instances_val2017.json).
# ---------------------------------------------------------

COCO_CATEGORY_IDS = {
    "car":     3,
    "chair":   62,
    "bottle":  44,
    "bird":    16,
    "boat":    9,
}

# Reverse mapping: COCO ID → our internal label (0–4)
COCO_ID_TO_LABEL = {
    coco_id: CATEGORIES.index(name)
    for name, coco_id in COCO_CATEGORY_IDS.items()
}

# Our internal label → category name
LABEL_TO_CATEGORY = {
    index: name
    for index, name in enumerate(CATEGORIES)
}
