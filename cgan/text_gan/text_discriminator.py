"""
Text-Conditioned Projection Discriminator
==========================================

Spectral-normalized discriminator that conditions on text
using the projection discriminator approach (Miyato & Koyama,
2018).  Also retains category label conditioning.

    score = unconditional(img_feats, label_emb)
          + dot(text_projection(pooled_emb), img_feats)

No Sigmoid — designed for hinge loss.
Same discriminator used for both cross-attention and
no-cross-attention generator variants.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from text_config import (
    IMAGE_CHANNELS,
    CATEGORIES,
    TEXT_EMBED_DIM,
)


# ---------------------------------------------------------
# Projection Discriminator
# ---------------------------------------------------------

class TextDiscriminator(nn.Module):
    """
    Spectral-normalized discriminator with:
      - CNN image encoder  (64×64 → 512×4×4 → 512-dim GAP)
      - Category embedding (32-dim)
      - Unconditional head (512 + 32 → 1)
      - Text projection    (768 → 512, then dot with img_feats)
    """

    def __init__(self):

        super().__init__()

        num_classes = len(CATEGORIES)

        # -------------------------------------------------
        # Category embedding
        # -------------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes, 32,
        )


        # -------------------------------------------------
        # Image encoder (spectral-normalized, no BatchNorm)
        # -------------------------------------------------

        self.image_model = nn.Sequential(

            # 64×64 → 32×32
            spectral_norm(
                nn.Conv2d(
                    IMAGE_CHANNELS, 64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 32×32 → 16×16
            spectral_norm(
                nn.Conv2d(
                    64, 128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 16×16 → 8×8
            spectral_norm(
                nn.Conv2d(
                    128, 256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 8×8 → 4×4
            spectral_norm(
                nn.Conv2d(
                    256, 512,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )


        # -------------------------------------------------
        # Global Average Pooling → 512-dim feature vector
        # -------------------------------------------------

        self.gap = nn.AdaptiveAvgPool2d(1)


        # -------------------------------------------------
        # Unconditional head
        #
        # image_features (512) + label_embedding (32) → 1
        # -------------------------------------------------

        self.unconditional_head = nn.Sequential(

            spectral_norm(
                nn.Linear(512 + 32, 256)
            ),

            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(
                nn.Linear(256, 1)
            ),
        )


        # -------------------------------------------------
        # Text projection for dot-product conditioning
        #
        # pooled_embedding (768) → 512
        # -------------------------------------------------

        self.text_projection = spectral_norm(
            nn.Linear(TEXT_EMBED_DIM, 512)
        )


    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        pooled_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images:           (B, 3, 64, 64)
            pooled_embedding: (B, 768) sentence vector
            labels:           (B,) category indices

        Returns:
            (B, 1) scores (logits, no Sigmoid)
        """

        # CNN features → (B, 512, 4, 4)
        features = self.image_model(images)

        # GAP → (B, 512)
        features = self.gap(features).flatten(1)

        # Category embedding → (B, 32)
        label_emb = self.label_embedding(labels)

        # Unconditional score
        unconditional_input = torch.cat(
            (features, label_emb),
            dim=1,
        )

        unconditional_score = self.unconditional_head(
            unconditional_input
        )

        # Text-conditioned score (projection)
        text_proj = self.text_projection(
            pooled_embedding
        )

        # Dot product: (B, 512) · (B, 512) → (B,)
        text_score = (features * text_proj).sum(
            dim=1, keepdim=True,
        )

        # Combined score
        return unconditional_score + text_score


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TEXT DISCRIMINATOR TEST")
    print("=" * 60)

    device = torch.device("cpu")

    discriminator = TextDiscriminator().to(device)

    images = torch.randn(
        4, IMAGE_CHANNELS, 64, 64,
        device=device,
    )

    pooled = torch.randn(
        4, TEXT_EMBED_DIM,
        device=device,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
        device=device,
    )

    scores = discriminator(images, pooled, labels)

    print(f"Image shape  : {tuple(images.shape)}")
    print(f"Pooled shape : {tuple(pooled.shape)}")
    print(f"Labels       : {labels.tolist()}")
    print(f"Output shape : {tuple(scores.shape)}")
    print(
        f"Score range  : "
        f"{scores.min().item():.4f} to "
        f"{scores.max().item():.4f}"
    )

    print(
        f"\nParameters   : "
        f"{sum(p.numel() for p in discriminator.parameters()):,}"
    )

    print("\nText discriminator test successful.")
