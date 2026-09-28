"""
Cross-Attention Ablation Script
================================

Loads generators trained WITH and WITHOUT cross-attention,
generates images from the same fixed captions and noise,
saves a side-by-side comparison grid, and computes FID
between the two sets of generated images.

Usage:
    python -m cgan.text_gan.ablation \
      --cross_attn_checkpoint cgan/text_gan/outputs/cross_attn/checkpoints/checkpoint_epoch_075.pt \
      --no_cross_attn_checkpoint cgan/text_gan/outputs/no_cross_attn/checkpoints/checkpoint_epoch_075.pt
"""

import argparse
import importlib
import math
import sys
from pathlib import Path

# Ensure text_gan directory and project root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torchvision.utils import save_image

try:
    from cgan.text_gan.text_config import (
        NOISE_DIM,
        TEXT_EMBED_DIM,
        CATEGORIES,
        TEXT_ENCODER_NAME,
        MAX_CAPTION_LENGTH,
    )
    from cgan.text_gan.text_attn_generator import TextAttentionGenerator
except ImportError:
    from text_config import (
        NOISE_DIM,
        TEXT_EMBED_DIM,
        CATEGORIES,
        TEXT_ENCODER_NAME,
        MAX_CAPTION_LENGTH,
    )
    from text_attn_generator import TextAttentionGenerator


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Cross-attention ablation comparison",
    )

    parser.add_argument(
        "--cross_attn_checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint trained WITH cross-attention",
    )

    parser.add_argument(
        "--no_cross_attn_checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint trained WITHOUT cross-attention",
    )

    parser.add_argument(
        "--num_fid_samples",
        type=int,
        default=200,
        help="Number of images to generate for FID (default: 200)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Fixed Comparison Captions (2 per category = 10 total)
# ---------------------------------------------------------

COMPARISON_CAPTIONS = {
    "car": [
        "A red car parked on a city street.",
        "A silver car driving along the highway.",
    ],
    "chair": [
        "A wooden chair beside a dining table.",
        "An office chair in a modern workspace.",
    ],
    "bottle": [
        "A glass bottle on a kitchen counter.",
        "A plastic water bottle on a wooden desk.",
    ],
    "bird": [
        "A small bird sitting on a tree branch.",
        "A colorful bird flying across the sky.",
    ],
    "boat": [
        "A white sailboat on calm blue water.",
        "A fishing boat docked at a wooden pier.",
    ],
}


# ---------------------------------------------------------
# Encode Captions
# ---------------------------------------------------------

def encode_captions(
    captions: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize and encode a list of captions.

    Returns:
        pooled_embs:  (N, 768)
        token_embs:   (N, T, 768)
        masks:        (N, T)
    """

    from transformers import AutoTokenizer
    try:
        from cgan.text_gan.text_encoder import TextEncoder
    except ImportError:
        from text_encoder import TextEncoder

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_ENCODER_NAME
    )

    encodings = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=MAX_CAPTION_LENGTH,
        return_tensors="pt",
    )

    encoder = TextEncoder().to(device)
    encoder.eval()

    with torch.no_grad():
        token_embs, pooled_embs = encoder(
            encodings["input_ids"].to(device),
            encodings["attention_mask"].to(device),
        )

    return (
        pooled_embs,
        token_embs,
        encodings["attention_mask"].to(device),
    )


# ---------------------------------------------------------
# Load Generator
# ---------------------------------------------------------

def load_generator(
    checkpoint_path: str,
    use_cross_attention: bool,
    device: torch.device,
) -> TextAttentionGenerator:
    """Load a generator from a checkpoint file."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    generator = TextAttentionGenerator(
        use_cross_attention=use_cross_attention,
    ).to(device)

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.eval()

    epoch = checkpoint.get("epoch", "?")
    print(
        f"  Loaded epoch {epoch} "
        f"(cross_attn={use_cross_attention})"
    )

    return generator


# ---------------------------------------------------------
# Generate Images
# ---------------------------------------------------------

@torch.no_grad()
def generate_images(
    generator: TextAttentionGenerator,
    noise: torch.Tensor,
    pooled_embs: torch.Tensor,
    token_embs: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Generate images from fixed inputs."""

    return generator(
        noise,
        pooled_embs,
        token_embs,
        masks,
    )


# ---------------------------------------------------------
# Compute FID
# ---------------------------------------------------------

def compute_fid(
    images_a: torch.Tensor,
    images_b: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute FID between two sets of generated images
    using torchmetrics.

    Both inputs should be (N, 3, H, W) in [-1, 1].
    """

    FrechetInceptionDistance = None
    try:
        tm_fid = importlib.import_module("torchmetrics.image.fid")
        FrechetInceptionDistance = getattr(tm_fid, "FrechetInceptionDistance", None)
    except (ImportError, ModuleNotFoundError, Exception):
        pass

    if FrechetInceptionDistance is None:
        print(
            "\n[WARNING] torchmetrics not installed. "
            "Skipping FID computation."
        )
        print(
            "Install with: pip install torchmetrics[image]"
        )
        return float("nan")

    # Rescale from [-1, 1] to [0, 1] and clamp to avoid numerical issues
    images_a_norm = torch.clamp((images_a.to(device) + 1.0) / 2.0, 0.0, 1.0)
    images_b_norm = torch.clamp((images_b.to(device) + 1.0) / 2.0, 0.0, 1.0)

    try:
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=True,
        ).to(device)
        fid.update(images_a_norm, real=True)
        fid.update(images_b_norm, real=False)
    except (TypeError, ValueError):
        # Compatibility fallback for torchmetrics versions requiring uint8 [0, 255]
        fid = FrechetInceptionDistance(
            feature=2048,
            normalize=False,
        ).to(device)
        fid.update((images_a_norm * 255.0).clamp(0, 255).to(torch.uint8), real=True)
        fid.update((images_b_norm * 255.0).clamp(0, 255).to(torch.uint8), real=False)

    score = fid.compute().item()

    return score


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Output directory
    output_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / "ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)


    print("\n" + "=" * 60)
    print("CROSS-ATTENTION ABLATION")
    print("=" * 60)

    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")


    # -------------------------------------------------
    # Load Models
    # -------------------------------------------------

    print("\nLoading generators...")

    gen_cross = load_generator(
        args.cross_attn_checkpoint,
        use_cross_attention=True,
        device=device,
    )

    gen_no_cross = load_generator(
        args.no_cross_attn_checkpoint,
        use_cross_attention=False,
        device=device,
    )


    # -------------------------------------------------
    # Prepare Comparison Captions
    # -------------------------------------------------

    print("\nEncoding comparison captions...")

    captions = []
    labels = []

    for cat_index, cat_name in enumerate(CATEGORIES):
        for caption in COMPARISON_CAPTIONS[cat_name]:
            captions.append(caption)
            labels.append(cat_index)

    pooled, tokens, masks = encode_captions(
        captions, device,
    )

    # Fixed noise for fair comparison
    torch.manual_seed(args.seed)
    noise = torch.randn(
        len(captions), NOISE_DIM,
        device=device,
    )

    print(f"Comparison captions: {len(captions)}")


    # -------------------------------------------------
    # Generate Comparison Images
    # -------------------------------------------------

    print("\nGenerating comparison images...")

    images_cross = generate_images(
        gen_cross, noise, pooled, tokens, masks,
    )

    images_no_cross = generate_images(
        gen_no_cross, noise, pooled, tokens, masks,
    )


    # -------------------------------------------------
    # Save Side-by-Side Grid
    #
    # Row 1: With cross-attention
    # Row 2: Without cross-attention
    # Columns: captions (2 per category × 5 = 10)
    # -------------------------------------------------

    comparison = torch.cat(
        [images_cross, images_no_cross],
        dim=0,
    )

    comparison_path = output_dir / "side_by_side.png"

    save_image(
        comparison,
        comparison_path,
        nrow=len(captions),
        normalize=True,
        value_range=(-1, 1),
    )

    print(f"\nSide-by-side grid saved to:\n{comparison_path}")


    # Save individual grids too
    cross_path = output_dir / "with_cross_attention.png"
    no_cross_path = output_dir / "without_cross_attention.png"

    save_image(
        images_cross,
        cross_path,
        nrow=len(CATEGORIES),
        normalize=True,
        value_range=(-1, 1),
    )

    save_image(
        images_no_cross,
        no_cross_path,
        nrow=len(CATEGORIES),
        normalize=True,
        value_range=(-1, 1),
    )


    # -------------------------------------------------
    # Gamma Values
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("GAMMA VALUES")
    print("-" * 60)

    print("\nWith cross-attention:")
    for name, value in gen_cross.get_gamma_values().items():
        print(f"  {name}: {value:.6f}")

    print("\nWithout cross-attention:")
    for name, value in gen_no_cross.get_gamma_values().items():
        print(f"  {name}: {value:.6f}")


    # -------------------------------------------------
    # Compute FID
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("FID COMPUTATION")
    print("-" * 60)

    # Generate more images for FID (need enough samples)
    num_fid = args.num_fid_samples
    print(f"\nGenerating {num_fid} images per variant for FID...")

    torch.manual_seed(args.seed + 1000)

    # Cycle through captions for FID generation
    fid_images_cross = []
    fid_images_no_cross = []

    num_captions = len(captions)

    for start in range(0, num_fid, num_captions):

        count = min(num_captions, num_fid - start)

        fid_noise = torch.randn(
            count, NOISE_DIM,
            device=device,
        )

        fid_cross = generate_images(
            gen_cross,
            fid_noise,
            pooled[:count],
            tokens[:count],
            masks[:count],
        )

        fid_no_cross = generate_images(
            gen_no_cross,
            fid_noise,
            pooled[:count],
            tokens[:count],
            masks[:count],
        )

        fid_images_cross.append(fid_cross)
        fid_images_no_cross.append(fid_no_cross)

    fid_images_cross = torch.cat(
        fid_images_cross, dim=0,
    )[:num_fid]

    fid_images_no_cross = torch.cat(
        fid_images_no_cross, dim=0,
    )[:num_fid]

    fid_score = compute_fid(
        fid_images_cross,
        fid_images_no_cross,
        device,
    )

    if math.isnan(fid_score):
        print("\nFID (cross-attn vs no-cross-attn): N/A (torchmetrics not installed)")
    else:
        print(f"\nFID (cross-attn vs no-cross-attn): {fid_score:.4f}")
        print(
            "  Lower FID = more similar distributions"
        )


    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("GENERATION STATISTICS")
    print("-" * 60)

    print("\nWith cross-attention:")
    print(f"  Mean : {images_cross.mean().item():.4f}")
    print(f"  Std  : {images_cross.std().item():.4f}")
    print(f"  Min  : {images_cross.min().item():.4f}")
    print(f"  Max  : {images_cross.max().item():.4f}")

    print("\nWithout cross-attention:")
    print(f"  Mean : {images_no_cross.mean().item():.4f}")
    print(f"  Std  : {images_no_cross.std().item():.4f}")
    print(f"  Min  : {images_no_cross.min().item():.4f}")
    print(f"  Max  : {images_no_cross.max().item():.4f}")


    # -------------------------------------------------
    # Caption Labels
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("CAPTIONS USED")
    print("-" * 60)

    for i, (caption, label) in enumerate(
        zip(captions, labels)
    ):
        cat = CATEGORIES[label]
        print(f"  {i}: [{cat}] {caption}")


    # -------------------------------------------------
    # Final Report
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("ABLATION COMPLETED")
    print("=" * 60)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Side-by-side   : {comparison_path.name}")
    print(f"  Cross-attn     : {cross_path.name}")
    print(f"  No cross-attn  : {no_cross_path.name}")
    fid_str = "N/A (install torchmetrics[image] to compute)" if math.isnan(fid_score) else f"{fid_score:.4f}"
    print(f"  FID score      : {fid_str}")

    print("\nAblation finished successfully.")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
