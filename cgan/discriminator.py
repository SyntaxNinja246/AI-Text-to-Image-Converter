import torch
import torch.nn as nn

from config import IMAGE_CHANNELS, CATEGORIES


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = len(CATEGORIES)

        # Strong but stable category embedding
        self.label_embedding = nn.Embedding(
            num_classes,
            32
        )

        self.image_model = nn.Sequential(

            # 64x64 -> 32x32
            nn.Conv2d(
                IMAGE_CHANNELS,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Image features = 512 * 4 * 4
        image_feature_dim = 512 * 4 * 4

        # Image features + 32-D category embedding
        self.classifier = nn.Sequential(

            nn.Linear(
                image_feature_dim + 32,
                256
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Linear(
                256,
                1
            ),

            nn.Sigmoid()
        )

        self.apply(self._initialize_weights)


    @staticmethod
    def _initialize_weights(module):

        if isinstance(module, nn.Conv2d):

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


    def forward(self, images, labels):

        image_features = self.image_model(images)

        image_features = image_features.view(
            image_features.size(0),
            -1
        )

        label_features = self.label_embedding(
            labels
        )

        combined = torch.cat(
            (
                image_features,
                label_features
            ),
            dim=1
        )

        return self.classifier(combined)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cpu")

    discriminator = Discriminator().to(device)

    images = torch.randn(
        4,
        IMAGE_CHANNELS,
        64,
        64,
        device=device
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
        device=device
    )

    predictions = discriminator(
        images,
        labels
    )

    print("Image shape:", images.shape)
    print("Labels:", labels.tolist())
    print(
        "Discriminator output shape:",
        predictions.shape
    )

    print(
        "Prediction range:",
        predictions.min().item(),
        "to",
        predictions.max().item()
    )

    print("Discriminator test successful.")