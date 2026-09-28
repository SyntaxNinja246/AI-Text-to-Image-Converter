from pathlib import Path

import torch
from torchvision.utils import save_image

from config import NOISE_DIM, CATEGORIES
from generator import Generator
from attention_generator import AttentionGenerator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent

BASELINE_MODEL = (
    PROJECT_DIR / "models" / "final_4_11_diversity"
    / "final_4_11" / "generator.pth"
)

ATTENTION_MODEL = (
    PROJECT_DIR / "models" / "attention_cgan"
    / "final" / "generator.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "attention_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE CGAN vs ATTENTION CGAN")
print("=" * 60)

print("\nBaseline model:")
print(BASELINE_MODEL)

print("\nAttention model:")
print(ATTENTION_MODEL)


if not BASELINE_MODEL.exists():
    raise FileNotFoundError(
        f"Baseline checkpoint not found:\n{BASELINE_MODEL}"
    )

if not ATTENTION_MODEL.exists():
    raise FileNotFoundError(
        f"Attention checkpoint not found:\n{ATTENTION_MODEL}"
    )


# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------

baseline = Generator().to(DEVICE)

attention = AttentionGenerator().to(DEVICE)

baseline.load_state_dict(
    torch.load(
        BASELINE_MODEL,
        map_location=DEVICE
    )
)

attention.load_state_dict(
    torch.load(
        ATTENTION_MODEL,
        map_location=DEVICE
    )
)

baseline.eval()
attention.eval()

print("\nBoth generators loaded successfully.")


# ---------------------------------------------------------
# SAME Noise + SAME Labels
# ---------------------------------------------------------

torch.manual_seed(42)

noise = torch.randn(
    len(CATEGORIES),
    NOISE_DIM,
    device=DEVICE
)

labels = torch.arange(
    len(CATEGORIES),
    device=DEVICE
)


# ---------------------------------------------------------
# Generate Images
# ---------------------------------------------------------

with torch.no_grad():

    baseline_images = baseline(
        noise,
        labels
    )

    attention_images = attention(
        noise,
        labels
    )


# ---------------------------------------------------------
# Save Individual Results
# ---------------------------------------------------------

baseline_path = (
    OUTPUT_DIR / "baseline_cgan.png"
)

attention_path = (
    OUTPUT_DIR / "attention_cgan.png"
)

save_image(
    baseline_images,
    baseline_path,
    nrow=len(CATEGORIES),
    normalize=True,
    value_range=(-1, 1)
)

save_image(
    attention_images,
    attention_path,
    nrow=len(CATEGORIES),
    normalize=True,
    value_range=(-1, 1)
)


# ---------------------------------------------------------
# Create Side-by-Side Comparison
# ---------------------------------------------------------

# Arrange:
#
# Row 1 = Baseline
# Row 2 = Attention
#
# Columns = categories

comparison = torch.cat(
    [
        baseline_images,
        attention_images
    ],
    dim=0
)

comparison_path = (
    OUTPUT_DIR / "baseline_vs_attention.png"
)

save_image(
    comparison,
    comparison_path,
    nrow=len(CATEGORIES),
    normalize=True,
    value_range=(-1, 1)
)


# ---------------------------------------------------------
# Print Statistics
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("GENERATION STATISTICS")
print("-" * 60)

print(
    "\nBaseline CGAN:"
)

print(
    "Mean:",
    baseline_images.mean().item()
)

print(
    "Std:",
    baseline_images.std().item()
)

print(
    "Min:",
    baseline_images.min().item()
)

print(
    "Max:",
    baseline_images.max().item()
)


print(
    "\nAttention CGAN:"
)

print(
    "Mean:",
    attention_images.mean().item()
)

print(
    "Std:",
    attention_images.std().item()
)

print(
    "Min:",
    attention_images.min().item()
)

print(
    "Max:",
    attention_images.max().item()
)


# ---------------------------------------------------------
# Final Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("COMPARISON COMPLETED")
print("=" * 60)

print("\nCategories:")

for index, category in enumerate(CATEGORIES):
    print(
        f"{index}: {category}"
    )

print(
    "\nBaseline image saved to:"
)

print(
    baseline_path
)

print(
    "\nAttention image saved to:"
)

print(
    attention_path
)

print(
    "\nSide-by-side comparison saved to:"
)

print(
    comparison_path
)

print("\nUse the side-by-side image for visual evaluation.")