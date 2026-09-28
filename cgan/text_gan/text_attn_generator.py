"""
Text-Conditioned Attention Generator
=====================================

Architecture: noise + pooled sentence vector → 4×4 → 8×8 → 16×16
→ 32×32 → 64×64 with:

  - Self-attention at 16×16 and 32×32 (gamma-gated residual)
  - Cross-attention at 16×16 and 32×32 where image features
    are the queries and projected word embeddings are the
    keys/values (with padding mask applied)

The `use_cross_attention` flag allows disabling cross-attention
for ablation studies while keeping the same state_dict keys.
"""

import torch
import torch.nn as nn

from text_config import (
    NOISE_DIM,
    IMAGE_CHANNELS,
    TEXT_EMBED_DIM,
)


# ---------------------------------------------------------
# Residual Block
# ---------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.block(x))


# ---------------------------------------------------------
# Self-Attention (gamma-gated)
# ---------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Standard multi-head self-attention over spatial
    feature maps with a learnable gamma gate.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True,
        )

        # Learnable gate, initialized to 0 so the
        # attention contribution starts at zero.
        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch, channels, height, width = x.shape

        # (B, C, H, W) → (B, H*W, C)
        features = x.flatten(2).transpose(1, 2)

        normalized = self.norm(features)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
        )

        # Gamma-gated residual
        features = features + self.gamma * attended

        # (B, H*W, C) → (B, C, H, W)
        return (
            features
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )


# ---------------------------------------------------------
# Cross-Attention (gamma-gated)
# ---------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross-attention where image spatial features are the
    queries and projected word embeddings are keys/values.

    Image input:  (B, C, H, W)
    Text input:   (B, T, text_dim)
    Mask input:   (B, T)  — 1 for real tokens, 0 for padding

    Output: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        text_dim: int = TEXT_EMBED_DIM,
        heads: int = 4,
    ):
        super().__init__()

        # Normalize image queries
        self.query_norm = nn.LayerNorm(channels)

        # Project text from text_dim → channels
        self.text_projection = nn.Linear(
            text_dim, channels
        )

        # Normalize projected text keys/values
        self.key_norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True,
        )

        # Learnable gate, initialized to 0
        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:               (B, C, H, W) image features
            text_embeddings: (B, T, text_dim) per-token embeddings
            text_mask:       (B, T) attention mask (1=real, 0=pad)

        Returns:
            (B, C, H, W) cross-attended image features
        """

        batch, channels, height, width = x.shape

        # (B, C, H, W) → (B, H*W, C)
        features = x.flatten(2).transpose(1, 2)

        # Normalize queries
        queries = self.query_norm(features)

        # Project and normalize text keys/values
        text_projected = self.text_projection(
            text_embeddings
        )
        text_projected = self.key_norm(text_projected)

        # PyTorch MHA expects key_padding_mask where
        # True = IGNORE (opposite of HuggingFace convention)
        key_padding_mask = ~text_mask.bool()

        attended, _ = self.attention(
            query=queries,
            key=text_projected,
            value=text_projected,
            key_padding_mask=key_padding_mask,
        )

        # Gamma-gated residual
        features = features + self.gamma * attended

        # (B, H*W, C) → (B, C, H, W)
        return (
            features
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )


# ---------------------------------------------------------
# Text-Conditioned Attention Generator
# ---------------------------------------------------------

class TextAttentionGenerator(nn.Module):
    """
    Generator conditioned on:
      - noise vector (NOISE_DIM)
      - pooled sentence embedding (TEXT_EMBED_DIM)
      - per-token word embeddings for cross-attention

    Architecture: 4×4 → 8×8 → 16×16 → 32×32 → 64×64

    Self-attention at 16×16 and 32×32.
    Cross-attention at 16×16 and 32×32 (togglable).
    """

    def __init__(
        self,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention


        # -------------------------------------------------
        # Initial projection: noise + pooled text → 4×4
        # -------------------------------------------------

        self.initial = nn.Sequential(

            nn.Linear(
                NOISE_DIM + TEXT_EMBED_DIM,
                256 * 4 * 4,
            ),

            nn.BatchNorm1d(256 * 4 * 4),

            nn.ReLU(True),

            nn.Unflatten(1, (256, 4, 4)),
        )


        # -------------------------------------------------
        # 4×4 → 8×8
        # -------------------------------------------------

        self.up1 = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(True),
        )

        self.residual_8 = ResidualBlock(128)


        # -------------------------------------------------
        # 8×8 → 16×16
        # -------------------------------------------------

        self.up2 = nn.Sequential(

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(True),
        )

        self.residual_16 = ResidualBlock(64)


        # -------------------------------------------------
        # Self-Attention at 16×16
        # -------------------------------------------------

        self.self_attention_16 = SelfAttention(
            channels=64,
            heads=4,
        )


        # -------------------------------------------------
        # Cross-Attention at 16×16
        # -------------------------------------------------

        self.cross_attention_16 = CrossAttention(
            channels=64,
            text_dim=TEXT_EMBED_DIM,
            heads=4,
        )


        # -------------------------------------------------
        # 16×16 → 32×32
        # -------------------------------------------------

        self.up3 = nn.Sequential(

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(True),
        )

        self.residual_32 = ResidualBlock(32)


        # -------------------------------------------------
        # Self-Attention at 32×32
        # -------------------------------------------------

        self.self_attention_32 = SelfAttention(
            channels=32,
            heads=4,
        )


        # -------------------------------------------------
        # Cross-Attention at 32×32
        # -------------------------------------------------

        self.cross_attention_32 = CrossAttention(
            channels=32,
            text_dim=TEXT_EMBED_DIM,
            heads=4,
        )


        # -------------------------------------------------
        # 32×32 → 64×64
        # -------------------------------------------------

        self.up4 = nn.Sequential(

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(True),
        )

        self.residual_64 = ResidualBlock(16)


        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        self.output = nn.Sequential(

            nn.Conv2d(
                16,
                IMAGE_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.Tanh(),
        )


        # -------------------------------------------------
        # Weight initialization
        # -------------------------------------------------

        self.apply(self._initialize_weights)


    # -----------------------------------------------------
    # Weight Initialization
    # -----------------------------------------------------

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:

        if isinstance(
            module,
            (nn.Conv2d, nn.ConvTranspose2d),
        ):
            nn.init.normal_(module.weight, 0.0, 0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):

            nn.init.normal_(module.weight, 0.0, 0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(
            module,
            (nn.BatchNorm2d, nn.BatchNorm1d),
        ):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)


    # -----------------------------------------------------
    # Gamma Logging
    # -----------------------------------------------------

    def get_gamma_values(self) -> dict[str, float]:
        """
        Return the current gamma values for all
        attention layers (useful for monitoring).
        """

        gammas = {
            "self_attn_16": (
                self.self_attention_16.gamma.item()
            ),
            "self_attn_32": (
                self.self_attention_32.gamma.item()
            ),
            "cross_attn_16": (
                self.cross_attention_16.gamma.item()
            ),
            "cross_attn_32": (
                self.cross_attention_32.gamma.item()
            ),
        }

        return gammas


    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def forward(
        self,
        noise: torch.Tensor,
        pooled_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate images conditioned on text.

        Args:
            noise:            (B, NOISE_DIM)
            pooled_embedding: (B, TEXT_EMBED_DIM) sentence vector
            token_embeddings: (B, T, TEXT_EMBED_DIM) word-level
            text_mask:        (B, T) attention mask (1=real, 0=pad)

        Returns:
            (B, 3, 64, 64) generated images in [-1, 1]
        """

        # Concatenate noise with pooled sentence vector
        generator_input = torch.cat(
            (noise, pooled_embedding),
            dim=1,
        )

        # Initial projection → (B, 256, 4, 4)
        x = self.initial(generator_input)


        # 4×4 → 8×8
        x = self.up1(x)
        x = self.residual_8(x)


        # 8×8 → 16×16
        x = self.up2(x)
        x = self.residual_16(x)


        # Self-attention at 16×16
        x = self.self_attention_16(x)

        # Cross-attention at 16×16 (conditional)
        if self.use_cross_attention:
            x = self.cross_attention_16(
                x, token_embeddings, text_mask,
            )


        # 16×16 → 32×32
        x = self.up3(x)
        x = self.residual_32(x)


        # Self-attention at 32×32
        x = self.self_attention_32(x)

        # Cross-attention at 32×32 (conditional)
        if self.use_cross_attention:
            x = self.cross_attention_32(
                x, token_embeddings, text_mask,
            )


        # 32×32 → 64×64
        x = self.up4(x)
        x = self.residual_64(x)


        return self.output(x)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TEXT ATTENTION GENERATOR TEST")
    print("=" * 60)

    device = torch.device("cpu")

    # -------------------------------------------------
    # Test WITH cross-attention
    # -------------------------------------------------

    print("\n--- With cross-attention ---")

    generator_on = TextAttentionGenerator(
        use_cross_attention=True,
    ).to(device)

    noise = torch.randn(2, NOISE_DIM, device=device)
    pooled = torch.randn(2, TEXT_EMBED_DIM, device=device)
    tokens = torch.randn(2, 64, TEXT_EMBED_DIM, device=device)
    mask = torch.ones(2, 64, dtype=torch.long, device=device)

    with torch.no_grad():
        images = generator_on(noise, pooled, tokens, mask)

    print(f"Output shape : {tuple(images.shape)}")
    print(
        f"Output range : "
        f"{images.min().item():.3f} to "
        f"{images.max().item():.3f}"
    )
    print(f"Gamma values : {generator_on.get_gamma_values()}")
    print(
        f"Parameters   : "
        f"{sum(p.numel() for p in generator_on.parameters()):,}"
    )

    # -------------------------------------------------
    # Test WITHOUT cross-attention
    # -------------------------------------------------

    print("\n--- Without cross-attention ---")

    generator_off = TextAttentionGenerator(
        use_cross_attention=False,
    ).to(device)

    with torch.no_grad():
        images_off = generator_off(noise, pooled, tokens, mask)

    print(f"Output shape : {tuple(images_off.shape)}")
    print(
        f"Output range : "
        f"{images_off.min().item():.3f} to "
        f"{images_off.max().item():.3f}"
    )

    # Verify state_dict keys match
    keys_on = set(generator_on.state_dict().keys())
    keys_off = set(generator_off.state_dict().keys())

    assert keys_on == keys_off, (
        "State dict keys differ between cross-attention "
        "on/off variants!"
    )

    print("\nState dict keys match ✓")

    print("\nText attention generator test successful.")
