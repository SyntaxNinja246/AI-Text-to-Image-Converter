import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import make_grid

from config import NOISE_DIM, CATEGORIES, DEVICE
from generator import Generator


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "final_cgan"
    / "generator.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "samples"
    / "final_cgan"
)

OUTPUT_PATH = OUTPUT_DIR / "evaluation.png"


# --------------------------------------------------
# Settings
# --------------------------------------------------

SAMPLES_PER_CATEGORY = 10
SEED = 42


# --------------------------------------------------
# Setup
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

generator = Generator().to(DEVICE)

generator.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

generator.eval()


# --------------------------------------------------
# Generate images
# --------------------------------------------------

all_images = []

with torch.no_grad():

    for category_index in range(len(CATEGORIES)):

        noise = torch.randn(
            SAMPLES_PER_CATEGORY,
            NOISE_DIM,
            device=DEVICE
        )

        labels = torch.full(
            (SAMPLES_PER_CATEGORY,),
            category_index,
            dtype=torch.long,
            device=DEVICE
        )

        images = generator(noise, labels)

        all_images.append(images)


# Combine all categories
images = torch.cat(all_images, dim=0)


# --------------------------------------------------
# Create evaluation grid
# --------------------------------------------------

grid = make_grid(
    images,
    nrow=SAMPLES_PER_CATEGORY,
    normalize=True,
    value_range=(-1, 1),
    padding=2
)

grid = grid.cpu().permute(1, 2, 0).numpy()


# --------------------------------------------------
# Save image
# --------------------------------------------------

plt.figure(figsize=(15, 8))
plt.imshow(grid)
plt.axis("off")
plt.title(
    "Final CGAN Evaluation — Category Conditioned Generation"
)
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL CGAN EVALUATION")
print("=" * 55)

print("Checkpoint:")
print(MODEL_PATH)

print("\nCategories:")
for i, category in enumerate(CATEGORIES):
    print(f"{i}: {category}")

print("\nSamples per category:", SAMPLES_PER_CATEGORY)
print("Total images generated:", len(images))
print("Image shape:", tuple(images.shape))

print("\nEvaluation image saved to:")
print(OUTPUT_PATH)

print("\nEvaluation completed successfully.")
print("=" * 55)