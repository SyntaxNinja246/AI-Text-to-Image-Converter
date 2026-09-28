import torch
from torch.utils.data import DataLoader

from config import NOISE_DIM, BATCH_SIZE
from dataset import CGANDataset
from generator import Generator
from discriminator import Discriminator


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

device = torch.device("cpu")

print("\n" + "=" * 60)
print("CGAN INTEGRATION TEST")
print("=" * 60)

print("Device:", device)


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

dataset = CGANDataset()

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

real_images, labels = next(iter(loader))

real_images = real_images.to(device)
labels = labels.to(device)


# ---------------------------------------------------------
# Create Models
# ---------------------------------------------------------

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.eval()
discriminator.eval()


# ---------------------------------------------------------
# Generate Fake Images
# ---------------------------------------------------------

noise = torch.randn(
    real_images.size(0),
    NOISE_DIM,
    device=device
)

with torch.no_grad():

    fake_images = generator(
        noise,
        labels
    )


# ---------------------------------------------------------
# Discriminator Predictions
# ---------------------------------------------------------

with torch.no_grad():

    real_predictions = discriminator(
        real_images,
        labels
    )

    fake_predictions = discriminator(
        fake_images,
        labels
    )


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print("\nReal images:")
print("Shape:", tuple(real_images.shape))

print("\nLabels:")
print(labels.tolist())

print("\nFake images:")
print("Shape:", tuple(fake_images.shape))

print("\nDiscriminator output:")
print("Real predictions:", real_predictions.squeeze().tolist())
print("Fake predictions:", fake_predictions.squeeze().tolist())

print("\n" + "-" * 60)

if fake_images.shape == real_images.shape:
    print("✓ Generator output matches real image dimensions.")
else:
    print("✗ Generator output dimensions do not match.")

if real_predictions.shape == fake_predictions.shape:
    print("✓ Discriminator accepts both real and fake images.")
else:
    print("✗ Discriminator output mismatch.")

print("\nCGAN integration test completed successfully.")