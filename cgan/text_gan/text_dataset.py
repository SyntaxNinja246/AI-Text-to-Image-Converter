"""
Text-Conditioned COCO Dataset
==============================

Loads COCO val2017 images filtered to the 5 target categories,
pairs each image with a caption, and returns:

    (image, caption_token_ids, attention_mask, category_label)

Filtering uses instances_val2017.json; captions come from
captions_val2017.json. Images are resized with an
aspect-preserving center crop to 64×64 and normalized to [-1, 1].
"""

import json
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from text_config import (
    IMAGE_DIR,
    INSTANCES_FILE,
    CAPTION_FILE,
    CATEGORIES,
    COCO_CATEGORY_IDS,
    COCO_ID_TO_LABEL,
    IMAGE_SIZE,
    TEXT_ENCODER_NAME,
    MAX_CAPTION_LENGTH,
)


# ---------------------------------------------------------
# Aspect-Preserving Center Crop
#
# Resize so the shorter side equals IMAGE_SIZE, then crop
# the center IMAGE_SIZE×IMAGE_SIZE region.
# ---------------------------------------------------------

def _build_transform(image_size: int) -> transforms.Compose:

    return transforms.Compose([

        # Resize shorter side to image_size
        transforms.Resize(
            image_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),

        # Center crop to square
        transforms.CenterCrop(image_size),

        transforms.ToTensor(),

        # Normalize to [-1, 1]
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class TextCOCODataset(Dataset):
    """
    COCO val2017 dataset filtered to the 5 target categories.

    Each sample is a whole image (not a bbox crop) paired with
    one of its COCO captions, tokenized for DistilBERT.
    """

    def __init__(
        self,
        image_dir: Path = IMAGE_DIR,
        instances_file: Path = INSTANCES_FILE,
        caption_file: Path = CAPTION_FILE,
        image_size: int = IMAGE_SIZE,
        max_caption_length: int = MAX_CAPTION_LENGTH,
    ):
        super().__init__()

        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.transform = _build_transform(image_size)

        # Load the tokenizer (fast, CPU-only, no model weights)
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ENCODER_NAME
        )

        # -------------------------------------------------
        # Step 1: Parse instances to find qualifying images
        # -------------------------------------------------

        target_coco_ids = set(COCO_CATEGORY_IDS.values())

        with open(instances_file, "r") as f:
            instances = json.load(f)

        # For each image, find the annotation with the
        # largest area among our target categories.
        # image_id → (coco_category_id, area)
        image_best_category: dict[int, tuple[int, float]] = {}

        for annotation in instances["annotations"]:

            coco_cat_id = annotation["category_id"]

            if coco_cat_id not in target_coco_ids:
                continue

            image_id = annotation["image_id"]
            area = annotation["area"]

            if image_id not in image_best_category:
                image_best_category[image_id] = (
                    coco_cat_id, area
                )
            else:
                _, current_best_area = (
                    image_best_category[image_id]
                )

                if area > current_best_area:
                    image_best_category[image_id] = (
                        coco_cat_id, area
                    )

        # -------------------------------------------------
        # Step 2: Parse captions for qualifying images
        # -------------------------------------------------

        with open(caption_file, "r") as f:
            captions_data = json.load(f)

        # image_id → list of caption strings
        image_captions: dict[int, list[str]] = {}

        for annotation in captions_data["annotations"]:

            image_id = annotation["image_id"]

            if image_id in image_best_category:

                if image_id not in image_captions:
                    image_captions[image_id] = []

                image_captions[image_id].append(
                    annotation["caption"]
                )

        # -------------------------------------------------
        # Step 3: Build the final sample list
        # -------------------------------------------------

        # Build image_id → filename lookup from instances
        id_to_filename: dict[int, str] = {}

        for img_info in instances["images"]:
            id_to_filename[img_info["id"]] = (
                img_info["file_name"]
            )

        self.samples: list[
            tuple[Path, str, int, int]
        ] = []
        # Each entry: (image_path, caption, label, image_id)

        for image_id, (coco_cat_id, _) in (
            image_best_category.items()
        ):
            # Skip if no captions found
            if image_id not in image_captions:
                continue

            filename = id_to_filename.get(image_id)
            if filename is None:
                continue

            image_path = self.image_dir / filename

            if not image_path.exists():
                continue

            # Pick one random caption
            caption = random.choice(
                image_captions[image_id]
            )

            label = COCO_ID_TO_LABEL[coco_cat_id]

            self.samples.append(
                (image_path, caption, label, image_id)
            )

        # -------------------------------------------------
        # Report
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("TEXT COCO DATASET")
        print("=" * 60)

        print(
            f"Total samples : {len(self.samples)}"
        )

        print("\nPer-category counts:")

        for cat_index, cat_name in enumerate(CATEGORIES):

            count = sum(
                1
                for _, _, label, _ in self.samples
                if label == cat_index
            )

            print(
                f"  {cat_index} → {cat_name:<10} : {count}"
            )

        print(f"Image size    : {image_size}×{image_size}")
        print(f"Max tokens    : {max_caption_length}")


    # -----------------------------------------------------
    # Length
    # -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)


    # -----------------------------------------------------
    # Get Item
    # -----------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        image_path, caption, label, image_id = (
            self.samples[index]
        )

        # -------------------------------------------------
        # Load and transform image
        # -------------------------------------------------

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # -------------------------------------------------
        # Tokenize caption
        # -------------------------------------------------

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )

        # Squeeze batch dimension added by return_tensors
        token_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # -------------------------------------------------
        # Category label
        # -------------------------------------------------

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, token_ids, attention_mask, label_tensor


    # -----------------------------------------------------
    # Get image ID (for embedding cache lookup)
    # -----------------------------------------------------

    def get_image_id(self, index: int) -> int:
        """Return the COCO image_id for the given index."""
        return self.samples[index][3]


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    dataset = TextCOCODataset()

    print(f"\nDataset length: {len(dataset)}")

    image, token_ids, attention_mask, label = dataset[0]

    print(f"Image shape      : {tuple(image.shape)}")
    print(f"Token IDs shape  : {tuple(token_ids.shape)}")
    print(f"Attention mask   : {tuple(attention_mask.shape)}")
    print(f"Label            : {label.item()}")
    print(f"Category         : {CATEGORIES[label.item()]}")

    print(
        f"Image range      : "
        f"{image.min().item():.3f} to "
        f"{image.max().item():.3f}"
    )

    # Decode tokens back to text
    decoded = dataset.tokenizer.decode(
        token_ids[attention_mask.bool()],
        skip_special_tokens=True,
    )

    print(f"Caption          : {decoded}")

    print("\nDataset test completed successfully.")
