from pathlib import Path

import torch
from torchvision.utils import save_image

from config import NOISE_DIM, CATEGORIES
from generator import Generator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "final_4_11"
    / "generator.pth"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "samples"
    / "final_4_11"
    / "checkpoint_test.png"
)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

print("=" * 60)
print("PHASE 4.11 CHECKPOINT TEST")
print("=" * 60)

print("Checkpoint:")
print(MODEL_PATH)

generator = Generator().to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

generator.load_state_dict(state_dict)

generator.eval()

print("Checkpoint loaded successfully.")


# ---------------------------------------------------------
# Generate controlled samples
# ---------------------------------------------------------

torch.manual_seed(42)

num_per_category = 5

noise = torch.randn(
    len(CATEGORIES) * num_per_category,
    NOISE_DIM,
    device=DEVICE
)

labels = torch.tensor(
    [
        category_index
        for category_index in range(len(CATEGORIES))
        for _ in range(num_per_category)
    ],
    dtype=torch.long,
    device=DEVICE
)


# ---------------------------------------------------------
# Generate
# ---------------------------------------------------------

with torch.no_grad():

    images = generator(
        noise,
        labels
    )


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print("\nGenerated tensor:")
print("Shape:", tuple(images.shape))

print(
    "Minimum:",
    images.min().item()
)

print(
    "Maximum:",
    images.max().item()
)

print(
    "Mean:",
    images.mean().item()
)

print(
    "Std:",
    images.std().item()
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

save_image(
    images,
    OUTPUT_PATH,
    nrow=num_per_category,
    normalize=True,
    value_range=(-1, 1)
)

print("\nSaved test image to:")
print(OUTPUT_PATH)

print("\nCheckpoint test completed.")