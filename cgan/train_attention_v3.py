from pathlib import Path

import torch
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
from discriminator_hinge import HingeDiscriminator


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models" / "attention_cgan_v3"
SAMPLE_DIR = PROJECT_DIR / "samples" / "attention_cgan_v3"
FINAL_MODEL_DIR = MODEL_DIR / "final"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 6


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ATTENTION CGAN V3 TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Categories:", CATEGORIES)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Loss: Hinge")
print("Discriminator: Spectral Normalization")


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

discriminator = HingeDiscriminator().to(DEVICE)


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

        real_scores = discriminator(
            real_images,
            labels
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


        fake_scores = discriminator(
            fake_images.detach(),
            labels
        )


        # Hinge discriminator loss:
        #
        # max(0, 1 - D(real))
        # max(0, 1 + D(fake))

        real_loss = torch.relu(
            1.0 - real_scores
        ).mean()

        fake_loss = torch.relu(
            1.0 + fake_scores
        ).mean()

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

        fake_scores = discriminator(
            fake_images,
            labels
        )


        # Generator wants high discriminator scores.

        g_loss = -fake_scores.mean()


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
        f"\nEpoch {epoch + 1} completed"
    )

    print(
        f"Average D Loss: {average_d_loss:.4f}"
    )

    print(
        f"Average G Loss: {average_g_loss:.4f}"
    )


    # -----------------------------------------------------
    # Generate Monitoring Samples
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
# Save Final Models
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

torch.save(
    discriminator.state_dict(),
    discriminator_path
)


# ---------------------------------------------------------
# Final Generator Check
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL V3 GENERATOR CHECK")
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
print("ATTENTION CGAN V3 TRAINING COMPLETED")
print("=" * 60)

print(
    "Training epochs:",
    NUM_EPOCHS
)

print(
    f"\nGenerator saved to:\n{generator_path}"
)

print(
    f"\nDiscriminator saved to:\n{discriminator_path}"
)

print(
    f"\nSamples saved to:\n{SAMPLE_DIR}"
)

print("\nTraining finished successfully.")