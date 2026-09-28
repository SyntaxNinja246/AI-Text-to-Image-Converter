import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTextModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    DATASET_DIR
    / "analysis_results"
    / "tokenized_captions.json"
)

OUTPUT_DIR = DATASET_DIR / "analysis_results" / "embeddings"

MODEL_ID = "segmind/tiny-sd"

DEVICE = torch.device("cpu")

BATCH_SIZE = 16


# ---------------------------------------------------------
# Load Tokenized Captions
# ---------------------------------------------------------

def load_tokenized_captions(input_file):
    """Load tokenized captions generated in Phase 3.3."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Tokenized caption file not found:\n{input_file}"
        )

    with input_file.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Load CLIP Text Encoder
# ---------------------------------------------------------

def load_text_encoder():
    """Load Tiny-SD's CLIP text encoder."""

    print("\nLoading CLIP text encoder...")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder"
    )

    text_encoder.to(DEVICE)
    text_encoder.eval()

    return text_encoder


# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------

def generate_embeddings(records, text_encoder):
    """
    Generate sequence-level CLIP embeddings for
    all tokenized captions.
    """

    all_embeddings = []

    for start in range(0, len(records), BATCH_SIZE):

        batch_records = records[
            start:start + BATCH_SIZE
        ]

        input_ids = torch.tensor(
            [
                record["input_ids"]
                for record in batch_records
            ],
            dtype=torch.long,
            device=DEVICE
        )

        attention_mask = torch.tensor(
            [
                record["attention_mask"]
                for record in batch_records
            ],
            dtype=torch.long,
            device=DEVICE
        )

        with torch.no_grad():

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        embeddings = outputs.last_hidden_state

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        processed = min(
            start + BATCH_SIZE,
            len(records)
        )

        print(
            f"Processed {processed}/{len(records)} captions"
        )

    return np.concatenate(
        all_embeddings,
        axis=0
    )


# ---------------------------------------------------------
# Save Embeddings
# ---------------------------------------------------------

def save_embeddings(embeddings, output_file):
    """Save embeddings as a NumPy array."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_file,
        embeddings
    )


# ---------------------------------------------------------
# Save Metadata
# ---------------------------------------------------------

def save_metadata(records, embeddings, output_file):
    """Save information describing the embedding dataset."""

    metadata = {
        "model": MODEL_ID,
        "device": str(DEVICE),
        "number_of_records": len(records),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "sequence_length": embeddings.shape[1],
        "embedding_dimension": embeddings.shape[2]
    }

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    # Load Phase 3.3 output
    records = load_tokenized_captions(
        INPUT_FILE
    )

    # Load Tiny-SD's CLIP text encoder
    text_encoder = load_text_encoder()

    # Generate embeddings
    embeddings = generate_embeddings(
        records,
        text_encoder
    )

    # Output locations
    embedding_file = (
        OUTPUT_DIR
        / "caption_embeddings.npy"
    )

    metadata_file = (
        OUTPUT_DIR
        / "embedding_metadata.json"
    )

    # Save results
    save_embeddings(
        embeddings,
        embedding_file
    )

    save_metadata(
        records,
        embeddings,
        metadata_file
    )

    # Final verification
    print("\n" + "=" * 65)
    print("TEXT EMBEDDINGS — PHASE 3.4")
    print("=" * 65)

    print(
        f"Number of captions : {embeddings.shape[0]}"
    )

    print(
        f"Sequence length    : {embeddings.shape[1]}"
    )

    print(
        f"Embedding dimension: {embeddings.shape[2]}"
    )

    print(
        f"Data type          : {embeddings.dtype}"
    )

    print(
        f"\nSaved embeddings to:\n{embedding_file}"
    )

    print(
        f"Saved metadata to:\n{metadata_file}"
    )

    print("\nEmbedding generation completed successfully.")