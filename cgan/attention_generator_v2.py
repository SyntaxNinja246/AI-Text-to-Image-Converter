import torch
import torch.nn as nn

from config import NOISE_DIM, IMAGE_CHANNELS, CATEGORIES


# ---------------------------------------------------------
# Residual Block
# ---------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(channels)
        )

    def forward(self, x):

        return torch.relu(
            x + self.block(x)
        )


# ---------------------------------------------------------
# Self Attention
# ---------------------------------------------------------

class SelfAttention(nn.Module):

    def __init__(
        self,
        channels,
        heads=4
    ):

        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):

        batch, channels, height, width = x.shape

        # Convert:
        # B x C x H x W
        #
        # to:
        # B x (H*W) x C

        features = x.flatten(2).transpose(1, 2)

        normalized = self.norm(features)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized
        )

        features = (
            features +
            self.gamma * attended
        )

        return (
            features
            .transpose(1, 2)
            .reshape(
                batch,
                channels,
                height,
                width
            )
        )


# ---------------------------------------------------------
# Attention CGAN Generator V2
# ---------------------------------------------------------

class AttentionGeneratorV2(nn.Module):

    def __init__(self):

        super().__init__()

        num_classes = len(CATEGORIES)

        # ---------------------------------------------
        # Category embedding
        # ---------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            32
        )


        # ---------------------------------------------
        # Initial projection
        # ---------------------------------------------

        self.initial = nn.Sequential(

            nn.Linear(
                NOISE_DIM + 32,
                256 * 4 * 4
            ),

            nn.BatchNorm1d(
                256 * 4 * 4
            ),

            nn.ReLU(True),

            nn.Unflatten(
                1,
                (256, 4, 4)
            )
        )


        # ---------------------------------------------
        # 4×4 → 8×8
        # ---------------------------------------------

        self.up1 = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(True)
        )

        self.residual_8 = ResidualBlock(128)


        # ---------------------------------------------
        # 8×8 → 16×16
        # ---------------------------------------------

        self.up2 = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(True)
        )

        self.residual_16 = ResidualBlock(64)


        # ---------------------------------------------
        # SELF-ATTENTION AT 16×16
        # ---------------------------------------------

        self.self_attention = SelfAttention(
            channels=64,
            heads=4
        )


        # ---------------------------------------------
        # 16×16 → 32×32
        # ---------------------------------------------

        self.up3 = nn.Sequential(

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(True)
        )

        self.residual_32 = ResidualBlock(32)


        # ---------------------------------------------
        # 32×32 → 64×64
        # ---------------------------------------------

        self.up4 = nn.Sequential(

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(True)
        )

        self.residual_64 = ResidualBlock(16)


        # ---------------------------------------------
        # Output
        # ---------------------------------------------

        self.output = nn.Sequential(

            nn.Conv2d(
                16,
                IMAGE_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.Tanh()
        )


        self.apply(
            self._initialize_weights
        )


    # -------------------------------------------------
    # Weight Initialization
    # -------------------------------------------------

    @staticmethod
    def _initialize_weights(module):

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d
            )
        ):

            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )


        elif isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )


        elif isinstance(
            module,
            (
                nn.BatchNorm2d,
                nn.BatchNorm1d
            )
        ):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias,
                0
            )


    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    def forward(
        self,
        noise,
        labels
    ):

        label_embedding = (
            self.label_embedding(labels)
        )

        generator_input = torch.cat(
            (
                noise,
                label_embedding
            ),
            dim=1
        )

        x = self.initial(
            generator_input
        )


        # 4 → 8

        x = self.up1(x)

        x = self.residual_8(x)


        # 8 → 16

        x = self.up2(x)

        x = self.residual_16(x)


        # Global spatial attention

        x = self.self_attention(x)


        # 16 → 32

        x = self.up3(x)

        x = self.residual_32(x)


        # 32 → 64

        x = self.up4(x)

        x = self.residual_64(x)


        return self.output(x)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ATTENTION CGAN GENERATOR V2 TEST")
    print("=" * 60)

    device = torch.device("cpu")

    model = AttentionGeneratorV2().to(device)

    noise = torch.randn(
        4,
        NOISE_DIM,
        device=device
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        device=device
    )

    with torch.no_grad():

        images = model(
            noise,
            labels
        )

    print(
        "Noise shape:",
        noise.shape
    )

    print(
        "Labels:",
        labels.tolist()
    )

    print(
        "Generated image shape:",
        images.shape
    )

    print(
        "Image range:",
        images.min().item(),
        "to",
        images.max().item()
    )

    print(
        "Generator parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    print(
        "\nV2 attention generator test successful."
    )