import json
import random
from pathlib import Path

# ============================================================
# PixelDream AI - Task 1 Dataset Preparation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "val2017"

CAPTION_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "annotations"
    / "annotations"
    / "captions_val2017.json"
)

OUTPUT_DIR = PROJECT_ROOT / "finetune" / "data"
OUTPUT_FILE = OUTPUT_DIR / "metadata.jsonl"

# Number of UNIQUE images used for the CPU fine-tuning experiment
NUM_SAMPLES = 500

SEED = 42


def main():

    print("=" * 60)
    print("PixelDream AI - Task 1 Dataset Preparation")
    print("=" * 60)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    if not CAPTION_FILE.exists():
        raise FileNotFoundError(
            f"Caption file not found:\n{CAPTION_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nImage directory: {IMAGE_DIR}")
    print(f"Caption file:    {CAPTION_FILE}")

    # --------------------------------------------------------
    # Load COCO captions
    # --------------------------------------------------------

    with open(
        CAPTION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        coco = json.load(f)

    images = {
        item["id"]: item["file_name"]
        for item in coco["images"]
    }

    annotations = coco["annotations"]

    print(
        f"\nCOCO images available: {len(images)}"
    )

    print(
        f"COCO captions available: {len(annotations)}"
    )

    # --------------------------------------------------------
    # Build valid image-caption records
    # --------------------------------------------------------

    records = []

    for annotation in annotations:

        image_id = annotation["image_id"]

        caption = annotation["caption"].strip()

        if image_id not in images:
            continue

        image_path = IMAGE_DIR / images[image_id]

        if not image_path.exists():
            continue

        if not caption:
            continue

        records.append(
            {
                "file_name": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "caption": caption
            }
        )

    print(
        f"Valid image-caption pairs: {len(records)}"
    )

    # --------------------------------------------------------
    # Randomize records reproducibly
    # --------------------------------------------------------

    random.seed(SEED)

    random.shuffle(records)

    # --------------------------------------------------------
    # Select ONE caption for each UNIQUE image
    # --------------------------------------------------------

    selected = []

    seen_images = set()

    for record in records:

        image_name = record["file_name"]

        if image_name in seen_images:
            continue

        seen_images.add(image_name)

        selected.append(record)

        if len(selected) >= NUM_SAMPLES:
            break

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(selected) < NUM_SAMPLES:

        raise RuntimeError(
            f"Only {len(selected)} unique images were found, "
            f"but {NUM_SAMPLES} are required."
        )

    # --------------------------------------------------------
    # Save JSONL metadata
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for record in selected:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    unique_images = len(
        set(
            record["file_name"]
            for record in selected
        )
    )

    empty_captions = sum(
        not record["caption"].strip()
        for record in selected
    )

    print(
        f"\nSelected records: {len(selected)}"
    )

    print(
        f"Unique images:    {unique_images}"
    )

    print(
        f"Empty captions:   {empty_captions}"
    )

    print(
        f"Saved metadata:   {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\nSample records:")

    for record in selected[:5]:

        print(
            f"  Image:   {record['file_name']}"
        )

        print(
            f"  Caption: {record['caption']}"
        )

        print()

    print("=" * 60)
    print(
        "Dataset preparation completed successfully."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()