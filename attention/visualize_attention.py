import os
import numpy as np
import matplotlib.pyplot as plt


INPUT_FILE = "attention/outputs/token_attention_maps.npz"
OUTPUT_DIR = "attention/outputs"


# Token positions discovered from our CLIP tokenizer
TOKENS = {
    "red": 2,
    "car": 3,
    "blue": 6,
    "house": 7
}


def main():

    print("=" * 60)
    print("PIXELDREAM AI - TASK 5")
    print("WORD-LEVEL CROSS-ATTENTION VISUALIZATION")
    print("=" * 60)

    print("\nLoading attention maps...")

    data = np.load(INPUT_FILE)

    attention_maps = data["attention_maps"]

    print("Original attention shape:")
    print(attention_maps.shape)

    # Shape:
    # steps × batch × heads × image_positions × text_tokens

    # Use the first conditioning batch.
    maps = attention_maps[:, 1]

    # Average across diffusion steps and attention heads.
    averaged_attention = maps.mean(axis=(0, 1))

    print("\nAveraged attention shape:")
    print(averaged_attention.shape)

    # 1024 = 32 × 32 spatial positions
    spatial_size = int(
        np.sqrt(averaged_attention.shape[0])
    )

    attention_spatial = averaged_attention.reshape(
        spatial_size,
        spatial_size,
        averaged_attention.shape[-1]
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("\nGenerating word-level maps...")

    for word, token_position in TOKENS.items():

        attention_map = attention_spatial[
            :, :,
            token_position
        ]

        plt.figure(figsize=(6, 6))

        plt.imshow(
            attention_map,
            interpolation="nearest"
        )

        plt.colorbar(
            label="Attention"
        )

        plt.title(
            f"Cross-Attention: '{word}'"
        )

        plt.xlabel("Spatial X")
        plt.ylabel("Spatial Y")

        output_file = os.path.join(
            OUTPUT_DIR,
            f"attention_{word}.png"
        )

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )

    # -------------------------------------------------
    # Create a combined figure
    # -------------------------------------------------

    print("\nCreating combined attention visualization...")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10)
    )

    for ax, (word, token_position) in zip(
        axes.flatten(),
        TOKENS.items()
    ):

        attention_map = attention_spatial[
            :, :,
            token_position
        ]

        image = ax.imshow(
            attention_map,
            interpolation="nearest"
        )

        ax.set_title(
            f"'{word}'"
        )

        ax.set_xlabel("Spatial X")
        ax.set_ylabel("Spatial Y")

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

    fig.suptitle(
        "PixelDream AI - Word-Level Cross-Attention",
        fontsize=16
    )

    combined_file = os.path.join(
        OUTPUT_DIR,
        "word_level_attention.png"
    )

    plt.tight_layout()

    plt.savefig(
        combined_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nCombined visualization saved:")
    print(combined_file)

    # -------------------------------------------------
    # Save metadata
    # -------------------------------------------------

    metadata_file = os.path.join(
        OUTPUT_DIR,
        "word_attention_metadata.json"
    )

    metadata = {
        "prompt": "a red car beside a blue house",
        "token_positions": TOKENS,
        "attention_shape": list(
            attention_maps.shape
        ),
        "averaging": {
            "diffusion_steps": True,
            "attention_heads": True
        }
    }

    import json

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nMetadata saved:")
    print(metadata_file)

    print("\n" + "=" * 60)
    print("WORD-LEVEL ATTENTION VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()