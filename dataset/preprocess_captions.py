import json
import re
from pathlib import Path

from load_coco_captions import (
    load_coco_captions,
    CAPTIONS_FILE
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = DATASET_DIR / "analysis_results"
OUTPUT_FILE = OUTPUT_DIR / "preprocessed_captions.json"


# ---------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------

def preprocess_caption(caption):
    """
    Clean a single COCO caption while preserving
    its original meaning.
    """

    # Remove leading and trailing whitespace
    caption = caption.strip()

    # Convert text to lowercase
    caption = caption.lower()

    # Normalize multiple whitespace characters
    caption = re.sub(r"\s+", " ", caption)

    # Remove punctuation while keeping letters,
    # numbers and spaces.
    caption = re.sub(r"[^\w\s]", "", caption)

    # Normalize whitespace once more after punctuation removal
    caption = re.sub(r"\s+", " ", caption).strip()

    return caption


# ---------------------------------------------------------
# Process Caption Records
# ---------------------------------------------------------

def preprocess_captions(caption_records):
    """Preprocess all caption records."""

    processed_records = []

    for record in caption_records:

        original_caption = record["caption"]
        cleaned_caption = preprocess_caption(original_caption)

        if not cleaned_caption:
            continue

        processed_records.append(
            {
                "image_id": record["image_id"],
                "original_caption": original_caption,
                "processed_caption": cleaned_caption
            }
        )

    return processed_records


# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

def save_processed_captions(records, output_file):
    """Save processed captions as JSON."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# Display Verification Samples
# ---------------------------------------------------------

def display_samples(records, number_of_samples=5):

    print("\n" + "=" * 60)
    print("COCO CAPTION PREPROCESSING — PHASE 3.2")
    print("=" * 60)

    print(f"Processed caption records : {len(records)}")

    print("\nBefore → After")
    print("-" * 60)

    for record in records[:number_of_samples]:

        print(f"Image ID : {record['image_id']}")
        print(f"Original : {record['original_caption']}")
        print(f"Processed: {record['processed_caption']}")
        print("-" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # Load the captions produced by Phase 3.1
    caption_records = load_coco_captions(CAPTIONS_FILE)

    # Preprocess the captions
    processed_records = preprocess_captions(
        caption_records
    )

    # Save the processed dataset
    save_processed_captions(
        processed_records,
        OUTPUT_FILE
    )

    # Display verification information
    display_samples(processed_records)

    print(f"\nSaved processed captions to:")
    print(OUTPUT_FILE)