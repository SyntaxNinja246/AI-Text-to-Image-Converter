import json
from pathlib import Path

from transformers import CLIPTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    DATASET_DIR
    / "analysis_results"
    / "preprocessed_captions.json"
)

OUTPUT_FILE = (
    DATASET_DIR
    / "analysis_results"
    / "tokenized_captions.json"
)

# Use the tokenizer associated with PixelDream's Tiny-SD model.
MODEL_ID = "segmind/tiny-sd"


# ---------------------------------------------------------
# Load Preprocessed Captions
# ---------------------------------------------------------

def load_preprocessed_captions(input_file):
    """Load captions produced by Phase 3.2."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Preprocessed caption file not found:\n{input_file}"
        )

    with input_file.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Load CLIP Tokenizer
# ---------------------------------------------------------

def load_tokenizer():
    """Load the tokenizer used by PixelDream's Tiny-SD model."""

    print("\nLoading CLIP tokenizer...")
    print(f"Model: {MODEL_ID}")

    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer"
    )

    return tokenizer


# ---------------------------------------------------------
# Tokenize Captions
# ---------------------------------------------------------

def tokenize_captions(records, tokenizer):
    """Convert processed captions into token IDs and masks."""

    captions = [
        record["processed_caption"]
        for record in records
    ]

    encoded = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_attention_mask=True
    )

    tokenized_records = []

    for index, record in enumerate(records):

        tokenized_records.append(
            {
                "image_id": record["image_id"],
                "original_caption": record["original_caption"],
                "processed_caption": record["processed_caption"],
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index]
            }
        )

    return tokenized_records


# ---------------------------------------------------------
# Save Tokenized Data
# ---------------------------------------------------------

def save_tokenized_captions(records, output_file):
    """Save tokenized captions as JSON."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=2
        )


# ---------------------------------------------------------
# Display Verification
# ---------------------------------------------------------

def display_samples(records, tokenizer, number_of_samples=3):

    print("\n" + "=" * 65)
    print("COCO CAPTION TOKENIZATION — PHASE 3.3")
    print("=" * 65)

    print(f"Tokenized caption records : {len(records)}")
    print(f"Maximum sequence length   : {tokenizer.model_max_length}")

    print("\nSample tokenization")
    print("-" * 65)

    for record in records[:number_of_samples]:

        tokens = tokenizer.convert_ids_to_tokens(
            record["input_ids"]
        )

        non_padding_tokens = sum(
            record["attention_mask"]
        )

        print(f"Image ID       : {record['image_id']}")
        print(f"Processed text : {record['processed_caption']}")
        print(f"Token count    : {non_padding_tokens}")
        print(f"Token IDs      : {record['input_ids'][:15]}")
        print(f"Tokens         : {tokens[:15]}")
        print("-" * 65)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # Load Phase 3.2 output
    records = load_preprocessed_captions(
        INPUT_FILE
    )

    # Load PixelDream's tokenizer
    tokenizer = load_tokenizer()

    # Tokenize captions
    tokenized_records = tokenize_captions(
        records,
        tokenizer
    )

    # Save results
    save_tokenized_captions(
        tokenized_records,
        OUTPUT_FILE
    )

    # Verify results
    display_samples(
        tokenized_records,
        tokenizer
    )

    print("\nTokenization completed successfully.")
    print(f"Saved tokenized data to:\n{OUTPUT_FILE}")