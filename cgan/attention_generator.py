import torch
import torch.nn as nn

from config import NOISE_DIM, IMAGE_CHANNELS, CATEGORIES


class SelfAttention(nn.Module):

    def __init__(self, channels, heads=4):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        batch, channels, height, width = x.shape

        # Convert feature map:
        # (B, C, H, W) -> (B, H*W, C)

        features = x.flatten(2).transpose(1, 2)

        normalized = self.norm(features)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized
        )

        features = features + self.gamma * attended

        # Convert back:
        # (B, H*W, C) -> (B, C, H, W)

        return features.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width
        )


class AttentionGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = len(CATEGORIES)

        self.label_embedding = nn.Embedding(
            num_classes,
            32
        )

        self.initial = nn.Sequential(
            nn.Linear(
                NOISE_DIM + 32,
                256 * 4 * 4
            ),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 4, 4))
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.noise_8 = nn.Linear(
            NOISE_DIM,
            16 * 8 * 8
        )

        self.label_8 = nn.Linear(
            32,
            16 * 8 * 8
        )

        self.refine_8 = nn.Sequential(
            nn.Conv2d(
                128 + 16 + 16,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.noise_16 = nn.Linear(
            NOISE_DIM,
            8 * 16 * 16
        )

        self.label_16 = nn.Linear(
            32,
            8 * 16 * 16
        )

        self.refine_16 = nn.Sequential(
            nn.Conv2d(
                64 + 8 + 8,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.noise_32 = nn.Linear(
            NOISE_DIM,
            4 * 32 * 32
        )

        self.refine_32 = nn.Sequential(
            nn.Conv2d(
                32 + 4,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # -------------------------------------------------
        # SELF-ATTENTION
        # -------------------------------------------------

        self.self_attention = SelfAttention(
            channels=32,
            heads=4
        )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(
                32,
                IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):

        if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )
            nn.init.constant_(module.bias, 0)

    def forward(self, noise, labels):

        label_embedding = self.label_embedding(labels)

        generator_input = torch.cat(
            (noise, label_embedding),
            dim=1
        )

        x = self.initial(generator_input)

        # 4 -> 8

        x = self.up1(x)

        noise_8 = self.noise_8(noise).view(
            noise.size(0), 16, 8, 8
        )

        label_8 = self.label_8(label_embedding).view(
            noise.size(0), 16, 8, 8
        )

        x = torch.cat(
            (x, noise_8, label_8),
            dim=1
        )

        x = self.refine_8(x)

        # 8 -> 16

        x = self.up2(x)

        noise_16 = self.noise_16(noise).view(
            noise.size(0), 8, 16, 16
        )

        label_16 = self.label_16(label_embedding).view(
            noise.size(0), 8, 16, 16
        )

        x = torch.cat(
            (x, noise_16, label_16),
            dim=1
        )

        x = self.refine_16(x)

        # 16 -> 32

        x = self.up3(x)

        noise_32 = self.noise_32(noise).view(
            noise.size(0), 4, 32, 32
        )

        x = torch.cat(
            (x, noise_32),
            dim=1
        )

        x = self.refine_32(x)

        # -------------------------------------------------
        # APPLY SELF-ATTENTION AT 32x32
        # -------------------------------------------------

        x = self.self_attention(x)

        # 32 -> 64

        return self.output(x)


if __name__ == "__main__":

    device = torch.device("cpu")

    generator = AttentionGenerator().to(device)

    noise = torch.randn(
        4,
        NOISE_DIM,
        device=device
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
        device=device
    )

    images = generator(noise, labels)

    print("Attention CGAN Generator Test")
    print("-----------------------------")
    print("Noise shape:", noise.shape)
    print("Labels:", labels.tolist())
    print("Generated image shape:", images.shape)
    print(
        "Image range:",
        images.min().item(),
        "to",
        images.max().item()
    )

    print("Self-attention:", generator.self_attention)
    print("Attention generator test successful.")