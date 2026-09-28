from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import (
    NOISE_DIM,
    BATCH_SIZE,
    LEARNING_RATE,
    BETA1,
    BETA2,
    CATEGORIES
)

from dataset import CGANDataset
from attention_generator import AttentionGenerator
from discriminator import Discriminator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models" / "attention_cgan"
SAMPLE_DIR = PROJECT_DIR / "samples" / "attention_cgan"
FINAL_MODEL_DIR = MODEL_DIR / "final"

GENERATOR_CHECKPOINT = (
    FINAL_MODEL_DIR / "generator.pth"
)

DISCRIMINATOR_CHECKPOINT = (
    FINAL_MODEL_DIR / "discriminator.pth"
)

# We already trained epochs 1-2.
START_EPOCH = 2

# Train four additional epochs.
# Total = 6 epochs.
ADDITIONAL_EPOCHS = 4

FINAL_EPOCH = START_EPOCH + ADDITIONAL_EPOCHS


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CONTINUING ATTENTION-ENHANCED CGAN TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Categories:", CATEGORIES)
print("Starting from epoch:", START_EPOCH)
print("Additional epochs:", ADDITIONAL_EPOCHS)
print("Final epoch:", FINAL_EPOCH)


# ---------------------------------------------------------
# Checkpoint Verification
# ---------------------------------------------------------

if not GENERATOR_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Generator checkpoint not found:\n"
        f"{GENERATOR_CHECKPOINT}"
    )

if not DISCRIMINATOR_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Discriminator checkpoint not found:\n"
        f"{DISCRIMINATOR_CHECKPOINT}"
    )

print("\nExisting attention checkpoints found.")


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = CGANDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print("Dataset samples:", len(dataset))
print("Batches per epoch:", len(loader))


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

generator = AttentionGenerator().to(DEVICE)

discriminator = Discriminator().to(DEVICE)


# ---------------------------------------------------------
# Load Existing Checkpoints
# ---------------------------------------------------------

generator.load_state_dict(
    torch.load(
        GENERATOR_CHECKPOINT,
        map_location=DEVICE
    )
)

discriminator.load_state_dict(
    torch.load(
        DISCRIMINATOR_CHECKPOINT,
        map_location=DEVICE
    )
)

print("\nExisting generator loaded.")
print("Existing discriminator loaded.")


# ---------------------------------------------------------
# Loss
# ---------------------------------------------------------

criterion = nn.BCELoss()


# ---------------------------------------------------------
# Optimizers
# ---------------------------------------------------------

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)


# ---------------------------------------------------------
# Fixed Evaluation Samples
# ---------------------------------------------------------

torch.manual_seed(42)

fixed_noise = torch.randn(
    len(CATEGORIES),
    NOISE_DIM,
    device=DEVICE
)

fixed_labels = torch.arange(
    len(CATEGORIES),
    device=DEVICE
)


# ---------------------------------------------------------
# Continued Training
# ---------------------------------------------------------

for epoch in range(
    START_EPOCH + 1,
    FINAL_EPOCH + 1
):

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    print(
        f"\nEpoch {epoch}/{FINAL_EPOCH}"
    )

    for batch_index, (real_images, labels) in enumerate(loader):

        real_images = real_images.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_size = real_images.size(0)


        # =================================================
        # Train Discriminator
        # =================================================

        optimizer_D.zero_grad()

        real_targets = torch.full(
            (batch_size, 1),
            0.9,
            device=DEVICE
        )

        real_output = discriminator(
            real_images,
            labels
        )

        real_loss = criterion(
            real_output,
            real_targets
        )


        # Generate fake images
        noise = torch.randn(
            batch_size,
            NOISE_DIM,
            device=DEVICE
        )

        fake_images = generator(
            noise,
            labels
        )

        fake_targets = torch.zeros(
            batch_size,
            1,
            device=DEVICE
        )

        fake_output = discriminator(
            fake_images.detach(),
            labels
        )

        fake_loss = criterion(
            fake_output,
            fake_targets
        )


        d_loss = (
            real_loss +
            fake_loss
        ) / 2

        d_loss.backward()

        optimizer_D.step()


        # =================================================
        # Train Generator
        # =================================================

        optimizer_G.zero_grad()

        noise = torch.randn(
            batch_size,
            NOISE_DIM,
            device=DEVICE
        )

        fake_images = generator(
            noise,
            labels
        )

        fake_output = discriminator(
            fake_images,
            labels
        )

        g_loss = criterion(
            fake_output,
            real_targets
        )

        g_loss.backward()

        optimizer_G.step()


        # =================================================
        # Record Losses
        # =================================================

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()


        # =================================================
        # Progress
        # =================================================

        if (batch_index + 1) % 25 == 0:

            print(
                f"Batch {batch_index + 1}/{len(loader)} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f}"
            )


    # -----------------------------------------------------
    # Average Losses
    # -----------------------------------------------------

    average_g_loss = (
        total_g_loss / len(loader)
    )

    average_d_loss = (
        total_d_loss / len(loader)
    )

    print(
        f"\nEpoch {epoch} completed"
    )

    print(
        f"Average D Loss: {average_d_loss:.4f}"
    )

    print(
        f"Average G Loss: {average_g_loss:.4f}"
    )


    # -----------------------------------------------------
    # Monitoring Samples
    # -----------------------------------------------------

    generator.eval()

    with torch.no_grad():

        samples = generator(
            fixed_noise,
            fixed_labels
        )

    sample_path = (
        SAMPLE_DIR /
        f"epoch_{epoch}.png"
    )

    save_image(
        samples,
        sample_path,
        nrow=len(CATEGORIES),
        normalize=True,
        value_range=(-1, 1)
    )

    print(
        f"Sample images saved to:\n{sample_path}"
    )


# ---------------------------------------------------------
# Save Updated Checkpoints
# ---------------------------------------------------------

torch.save(
    generator.state_dict(),
    GENERATOR_CHECKPOINT
)

torch.save(
    discriminator.state_dict(),
    DISCRIMINATOR_CHECKPOINT
)


# ---------------------------------------------------------
# Final Check
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL ATTENTION CGAN CHECK")
print("=" * 60)

generator.eval()

with torch.no_grad():

    test_images = generator(
        fixed_noise,
        fixed_labels
    )

print(
    "Generator parameters:",
    sum(
        p.numel()
        for p in generator.parameters()
    )
)

print(
    "Final image range:",
    test_images.min().item(),
    "to",
    test_images.max().item()
)

print(
    "Final image mean:",
    test_images.mean().item()
)

print(
    "Final image std:",
    test_images.std().item()
)


# ---------------------------------------------------------
# Final Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ATTENTION CGAN CONTINUATION COMPLETED")
print("=" * 60)

print(
    "Training completed through epoch:",
    FINAL_EPOCH
)

print(
    f"\nGenerator saved to:\n"
    f"{GENERATOR_CHECKPOINT}"
)

print(
    f"\nDiscriminator saved to:\n"
    f"{DISCRIMINATOR_CHECKPOINT}"
)

print(
    f"\nSamples saved to:\n"
    f"{SAMPLE_DIR}"
)

print("\nTraining finished successfully.")