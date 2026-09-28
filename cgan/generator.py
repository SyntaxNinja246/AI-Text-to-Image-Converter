import torch
import torch.nn as nn

from config import NOISE_DIM, IMAGE_CHANNELS, CATEGORIES


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = len(CATEGORIES)

        # -------------------------------------------------
        # Category embedding
        # -------------------------------------------------

        self.label_embedding = nn.Embedding(
            num_classes,
            32
        )

        # -------------------------------------------------
        # Initial noise + category
        # -------------------------------------------------

        self.initial = nn.Sequential(

            nn.Linear(
                NOISE_DIM + 32,
                256 * 4 * 4
            ),

            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),

            nn.Unflatten(
                1,
                (256, 4, 4)
            )
        )

        # -------------------------------------------------
        # 4x4 -> 8x8
        # -------------------------------------------------

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

        # Extra noise injection at 8x8
        self.noise_8 = nn.Linear(
            NOISE_DIM,
            16 * 8 * 8
        )

        # Extra category injection at 8x8
        self.label_8 = nn.Linear(
            32,
            16 * 8 * 8
        )

        self.refine_8 = nn.Sequential(

            nn.Conv2d(
                128 + 16 + 16,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        # -------------------------------------------------
        # 8x8 -> 16x16
        # -------------------------------------------------

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

        # Extra noise injection at 16x16
        self.noise_16 = nn.Linear(
            NOISE_DIM,
            8 * 16 * 16
        )

        # Extra category injection at 16x16
        self.label_16 = nn.Linear(
            32,
            8 * 16 * 16
        )

        self.refine_16 = nn.Sequential(

            nn.Conv2d(
                64 + 8 + 8,
                64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # -------------------------------------------------
        # 16x16 -> 32x32
        # -------------------------------------------------

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

        # Extra noise injection at 32x32
        self.noise_32 = nn.Linear(
            NOISE_DIM,
            4 * 32 * 32
        )

        self.refine_32 = nn.Sequential(

            nn.Conv2d(
                32 + 4,
                32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # -------------------------------------------------
        # 32x32 -> 64x64
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Weight initialization
        # -------------------------------------------------

        self.apply(self._initialize_weights)


    @staticmethod
    def _initialize_weights(module):

        if isinstance(module, nn.ConvTranspose2d):

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

        elif isinstance(module, nn.Conv2d):

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

        elif isinstance(module, nn.Linear):

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

        elif isinstance(module, nn.BatchNorm2d):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias,
                0
            )

        elif isinstance(module, nn.BatchNorm1d):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )

            nn.init.constant_(
                module.bias,
                0
            )


    def forward(self, noise, labels):

        # -------------------------------------------------
        # Initial conditioning
        # -------------------------------------------------

        label_embedding = self.label_embedding(labels)

        generator_input = torch.cat(
            (
                noise,
                label_embedding
            ),
            dim=1
        )

        x = self.initial(generator_input)

        # -------------------------------------------------
        # 4x4 -> 8x8
        # -------------------------------------------------

        x = self.up1(x)

        noise_8 = self.noise_8(noise)
        noise_8 = noise_8.view(
            noise.size(0),
            16,
            8,
            8
        )

        label_8 = self.label_8(label_embedding)
        label_8 = label_8.view(
            noise.size(0),
            16,
            8,
            8
        )

        x = torch.cat(
            (
                x,
                noise_8,
                label_8
            ),
            dim=1
        )

        x = self.refine_8(x)

        # -------------------------------------------------
        # 8x8 -> 16x16
        # -------------------------------------------------

        x = self.up2(x)

        noise_16 = self.noise_16(noise)
        noise_16 = noise_16.view(
            noise.size(0),
            8,
            16,
            16
        )

        label_16 = self.label_16(label_embedding)
        label_16 = label_16.view(
            noise.size(0),
            8,
            16,
            16
        )

        x = torch.cat(
            (
                x,
                noise_16,
                label_16
            ),
            dim=1
        )

        x = self.refine_16(x)

        # -------------------------------------------------
        # 16x16 -> 32x32
        # -------------------------------------------------

        x = self.up3(x)

        noise_32 = self.noise_32(noise)
        noise_32 = noise_32.view(
            noise.size(0),
            4,
            32,
            32
        )

        x = torch.cat(
            (
                x,
                noise_32
            ),
            dim=1
        )

        x = self.refine_32(x)

        # -------------------------------------------------
        # 32x32 -> 64x64
        # -------------------------------------------------

        return self.output(x)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cpu")

    generator = Generator().to(device)

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

    images = generator(
        noise,
        labels
    )

    print("Noise shape:", noise.shape)
    print("Labels:", labels.tolist())
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

    print("Generator test successful.")