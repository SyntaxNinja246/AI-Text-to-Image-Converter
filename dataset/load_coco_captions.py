import json
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent

CAPTIONS_FILE = (
    DATASET_DIR
    / "annotations"
    / "annotations"
    / "captions_val2017.json"
)


# ---------------------------------------------------------
# COCO Caption Loader
# ---------------------------------------------------------

def load_coco_captions(captions_file):
    """Load captions from a COCO captions JSON file."""

    if not captions_file.exists():
        raise FileNotFoundError(
            f"COCO captions file not found:\n{captions_file}"
        )

    with captions_file.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    if "annotations" not in coco_data:
        raise ValueError(
            "The JSON file does not contain an 'annotations' field."
        )

    caption_records = []

    for annotation in coco_data["annotations"]:
        image_id = annotation.get("image_id")
        caption = annotation.get("caption")

        if image_id is None or not caption:
            continue

        caption_records.append(
            {
                "image_id": image_id,
                "caption": caption.strip()
            }
        )

    return caption_records


# ---------------------------------------------------------
# Basic Dataset Inspection
# ---------------------------------------------------------

def display_summary(caption_records):
    """Display a small summary of the loaded captions."""

    print("\n" + "=" * 50)
    print("COCO CAPTION DATASET — PHASE 3.1")
    print("=" * 50)

    print(f"Caption records loaded : {len(caption_records)}")

    unique_image_ids = {
        record["image_id"]
        for record in caption_records
    }

    print(f"Unique image IDs       : {len(unique_image_ids)}")

    print("\nSample captions:")
    print("-" * 50)

    for record in caption_records[:5]:
        print(f"Image ID : {record['image_id']}")
        print(f"Caption  : {record['caption']}")
        print()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    captions = load_coco_captions(CAPTIONS_FILE)

    display_summary(captions)