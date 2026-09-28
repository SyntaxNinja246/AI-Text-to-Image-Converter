import json
from pathlib import Path
from collections import Counter

from PIL import Image


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = (
    PROJECT_DIR
    / "dataset"
    / "images"
    / "val2017"
)

ANNOTATION_FILE = (
    PROJECT_DIR
    / "dataset"
    / "annotations"
    / "annotations"
    / "instances_val2017.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "cgan"
    / "data"
)


# ---------------------------------------------------------
# CGAN Categories
# ---------------------------------------------------------

SELECTED_CATEGORIES = [
    "car",
    "chair",
    "bottle",
    "bird",
    "boat"
]

MAX_SAMPLES_PER_CATEGORY = 500


# ---------------------------------------------------------
# Load COCO annotations
# ---------------------------------------------------------

with open(
    ANNOTATION_FILE,
    "r",
    encoding="utf-8"
) as file:

    coco = json.load(file)


# ---------------------------------------------------------
# Category lookup
# ---------------------------------------------------------

category_lookup = {
    category["id"]: category["name"]
    for category in coco["categories"]
}


selected_ids = {
    category_id
    for category_id, name in category_lookup.items()
    if name in SELECTED_CATEGORIES
}


# ---------------------------------------------------------
# Image lookup
# ---------------------------------------------------------

image_lookup = {
    image["id"]: image["file_name"]
    for image in coco["images"]
}


# ---------------------------------------------------------
# Prepare output directories
# ---------------------------------------------------------

for category in SELECTED_CATEGORIES:

    (
        OUTPUT_DIR / category
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ---------------------------------------------------------
# Counters
# ---------------------------------------------------------

counts = Counter()


# ---------------------------------------------------------
# Create object crops
# ---------------------------------------------------------

for annotation in coco["annotations"]:

    category_id = annotation["category_id"]

    if category_id not in selected_ids:
        continue

    category_name = category_lookup[
        category_id
    ]

    if counts[category_name] >= MAX_SAMPLES_PER_CATEGORY:
        continue

    image_id = annotation["image_id"]

    image_name = image_lookup[image_id]

    image_path = IMAGE_DIR / image_name

    if not image_path.exists():
        continue

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        x, y, width, height = annotation["bbox"]

        # Convert COCO coordinates to integers
        left = max(0, int(x))
        top = max(0, int(y))

        right = min(
            image.width,
            int(x + width)
        )

        bottom = min(
            image.height,
            int(y + height)
        )

        # Ignore invalid bounding boxes
        if right <= left or bottom <= top:
            continue

        crop = image.crop(
            (left, top, right, bottom)
        )

        # Ignore extremely small crops
        if crop.width < 20 or crop.height < 20:
            continue

        output_name = (
            f"{category_name}_"
            f"{image_id}_"
            f"{counts[category_name]:04d}.jpg"
        )

        output_path = (
            OUTPUT_DIR
            / category_name
            / output_name
        )

        crop.save(
            output_path,
            quality=95
        )

        counts[category_name] += 1

    except Exception as error:

        print(
            f"Skipping image {image_id}: {error}"
        )


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CGAN TRAINING DATA PREPARATION")
print("=" * 60)

for category in SELECTED_CATEGORIES:

    print(
        f"{category:<15} : "
        f"{counts[category]} crops"
    )

print(
    f"\nTotal crops : {sum(counts.values())}"
)

print(
    f"Output directory : {OUTPUT_DIR}"
)

print("\nDataset preparation completed successfully.")