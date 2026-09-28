from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "cgan" / "data"

CATEGORIES = [
    "car",
    "chair",
    "bottle",
    "bird",
    "boat"
]

CATEGORY_TO_LABEL = {
    category: index
    for index, category in enumerate(CATEGORIES)
}


# ---------------------------------------------------------
# CGAN Dataset
# ---------------------------------------------------------

class CGANDataset(Dataset):

    def __init__(self, data_dir=DATA_DIR):

        self.data_dir = Path(data_dir)

        self.samples = []

        # ---------------------------------------------
        # Collect image paths and labels
        # ---------------------------------------------

        for category in CATEGORIES:

            category_dir = self.data_dir / category

            if not category_dir.exists():
                raise FileNotFoundError(
                    f"Category directory not found:\n{category_dir}"
                )

            label = CATEGORY_TO_LABEL[category]

            for image_path in category_dir.glob("*.jpg"):

                self.samples.append(
                    (image_path, label)
                )

        # ---------------------------------------------
        # Image preprocessing
        # ---------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),

                transforms.ToTensor(),

                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]
                )
            ]
        )

        print("\n" + "=" * 60)
        print("CGAN DATASET LOADER")
        print("=" * 60)

        print(
            f"Total samples : {len(self.samples)}"
        )

        print("\nCategories:")

        for category in CATEGORIES:

            count = sum(
                1
                for _, label in self.samples
                if label == CATEGORY_TO_LABEL[category]
            )

            print(
                f"{CATEGORY_TO_LABEL[category]} "
                f"→ {category:<10} : {count}"
            )


    # -----------------------------------------------------
    # Dataset length
    # -----------------------------------------------------

    def __len__(self):

        return len(self.samples)


    # -----------------------------------------------------
    # Get sample
    # -----------------------------------------------------

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ---------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------

if __name__ == "__main__":

    dataset = CGANDataset()

    print("\nDataset length:", len(dataset))

    image, label = dataset[0]

    print(
        "Image shape:",
        tuple(image.shape)
    )

    print(
        "Label:",
        label.item()
    )

    print(
        "Category:",
        CATEGORIES[label.item()]
    )

    print(
        "Image range:",
        image.min().item(),
        "to",
        image.max().item()
    )

    print("\nDataset test completed successfully.")