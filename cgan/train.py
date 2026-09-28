import os
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
    NUM_EPOCHS,
    CATEGORIES
)

from dataset import CGANDataset
from generator import Generator
from discriminator import Discriminator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"/"final_4_11_diversity"
SAMPLE_DIR = PROJECT_DIR / "samples"/"final_4_11_diversity"
FINAL_MODEL_DIR = MODEL_DIR/"final_4_11"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Training Setup
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CGAN TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Categories:", CATEGORIES)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)


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


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

generator = Generator().to(DEVICE)

discriminator = Discriminator().to(DEVICE)


# ---------------------------------------------------------
# Loss Function
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
# Fixed samples for monitoring
# ---------------------------------------------------------

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
# Training Loop
# ---------------------------------------------------------

for epoch in range(NUM_EPOCHS):

    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
    )

    for batch_index, (real_images, labels) in enumerate(loader):

        real_images = real_images.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_size = real_images.size(0)


        # =================================================
        # Train Discriminator
        # =================================================

        optimizer_D.zero_grad()

        # Real images
        real_targets = torch.full(
            (batch_size,1),
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


        # Fake images
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


        # Total discriminator loss
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

        # Generator wants discriminator
        # to classify fake images as real
        g_loss = criterion(
            fake_output,
            real_targets
        )

        g_loss.backward()

        optimizer_G.step()


        # =================================================
        # Record losses
        # =================================================

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (batch_index + 1) % 25 == 0:

            print(
                f"Batch {batch_index + 1}/{len(loader)} | "
                f"D Loss: {d_loss.item():.4f} | "
                f"G Loss: {g_loss.item():.4f}"
            )


    # -----------------------------------------------------
    # Average losses
    # -----------------------------------------------------

    average_g_loss = (
        total_g_loss / len(loader)
    )

    average_d_loss = (
        total_d_loss / len(loader)
    )

    print(
        f"\nEpoch {epoch + 1} completed"
    )

    print(
        f"Average D Loss: {average_d_loss:.4f}"
    )

    print(
        f"Average G Loss: {average_g_loss:.4f}"
    )


    # -----------------------------------------------------
    # Generate monitoring samples
    # -----------------------------------------------------

    generator.eval()

    with torch.no_grad():

        samples = generator(
            fixed_noise,
            fixed_labels
        )

    sample_path = (
        SAMPLE_DIR /
        f"epoch_{epoch + 1}.png"
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
# Save Models
# ---------------------------------------------------------

generator_path = (
    FINAL_MODEL_DIR /
    "generator.pth"
)

discriminator_path = (
    FINAL_MODEL_DIR /
    "discriminator.pth"
)

torch.save(
    generator.state_dict(),
    generator_path
)

print("\nFINAL GENERATOR CHECK")
print(
    "Generator parameters:",
    sum(p.numel() for p in generator.parameters())
)

generator.eval()

with torch.no_grad():
    test_images = generator(
        fixed_noise,
        fixed_labels
    )

print(
    "Final test image range:",
    test_images.min().item(),
    "to",
    test_images.max().item()
)

print(
    "Final test image mean:",
    test_images.mean().item()
)

print(
    "Final test image std:",
    test_images.std().item()
)

torch.save(
    discriminator.state_dict(),
    discriminator_path
)


# ---------------------------------------------------------
# Final Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CGAN TRAINING COMPLETED")
print("=" * 60)

print(
    f"Generator saved to:\n{generator_path}"
)

print(
    f"Discriminator saved to:\n{discriminator_path}"
)

print(
    f"Samples saved to:\n{SAMPLE_DIR}"
)

print("\nTraining finished successfully.")