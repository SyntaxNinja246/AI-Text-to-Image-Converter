import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from config import IMAGE_CHANNELS, CATEGORIES


class HingeDiscriminator(nn.Module):

    def __init__(self):

        super().__init__()

        num_classes = len(CATEGORIES)

        self.label_embedding = nn.Embedding(
            num_classes,
            32
        )

        # No BatchNorm in discriminator.
        # Spectral normalization stabilizes the discriminator.

        self.image_model = nn.Sequential(

            # 64 -> 32
            spectral_norm(
                nn.Conv2d(
                    IMAGE_CHANNELS,
                    64,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # 32 -> 16
            spectral_norm(
                nn.Conv2d(
                    64,
                    128,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # 16 -> 8
            spectral_norm(
                nn.Conv2d(
                    128,
                    256,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),


            # 8 -> 4
            spectral_norm(
                nn.Conv2d(
                    256,
                    512,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )


        image_feature_dim = 512 * 4 * 4


        self.classifier = nn.Sequential(

            spectral_norm(
                nn.Linear(
                    image_feature_dim + 32,
                    256
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            spectral_norm(
                nn.Linear(
                    256,
                    1
                )
            )
        )


    def forward(
        self,
        images,
        labels
    ):

        image_features = self.image_model(
            images
        )

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

        # IMPORTANT:
        # No Sigmoid here.
        # Hinge loss works directly with logits/scores.

        return self.classifier(
            combined
        )


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cpu")

    discriminator = HingeDiscriminator().to(device)

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

    print("\n" + "=" * 60)
    print("HINGE DISCRIMINATOR TEST")
    print("=" * 60)

    print(
        "Image shape:",
        images.shape
    )

    print(
        "Labels:",
        labels.tolist()
    )

    print(
        "Output shape:",
        predictions.shape
    )

    print(
        "Output range:",
        predictions.min().item(),
        "to",
        predictions.max().item()
    )

    print(
        "\nHinge discriminator test successful."
    )