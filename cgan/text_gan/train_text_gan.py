"""
Text-Conditioned GAN Training Script
======================================

Full training loop with argparse, hinge loss, text embedding
caching, checkpoint saving (including optimizer state for
resume), sample grid generation, and gamma value logging.

Usage:
    # Train with cross-attention
    python -m cgan.text_gan.train_text_gan --use_cross_attention --epochs 75 --batch_size 8 --seed 42

    # Train without cross-attention (ablation baseline)
    python -m cgan.text_gan.train_text_gan --epochs 75 --batch_size 8 --seed 42

    # Resume from checkpoint
    python -m cgan.text_gan.train_text_gan --use_cross_attention --resume cgan/text_gan/outputs/cross_attn/checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import sys
from pathlib import Path

# Ensure text_gan directory and project root are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

for p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

try:
    from cgan.text_gan.text_config import (
        NOISE_DIM,
        TEXT_EMBED_DIM,
        CATEGORIES,
        LEARNING_RATE,
        BETA1,
        BETA2,
    )
    from cgan.text_gan.text_dataset import TextCOCODataset
    from cgan.text_gan.text_encoder import encode_and_cache
    from cgan.text_gan.text_attn_generator import TextAttentionGenerator
    from cgan.text_gan.text_discriminator import TextDiscriminator
except ImportError:
    from text_config import (
        NOISE_DIM,
        TEXT_EMBED_DIM,
        CATEGORIES,
        LEARNING_RATE,
        BETA1,
        BETA2,
    )
    from text_dataset import TextCOCODataset
    from text_encoder import encode_and_cache
    from text_attn_generator import TextAttentionGenerator
    from text_discriminator import TextDiscriminator


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train text-conditioned GAN",
    )

    parser.add_argument(
        "--use_cross_attention",
        action="store_true",
        default=False,
        help="Enable cross-attention blocks in generator",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="Number of training epochs (default: 75)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Output Directories
# ---------------------------------------------------------

TEXT_GAN_DIR = Path(__file__).resolve().parent


def setup_directories(
    use_cross_attention: bool,
) -> tuple[Path, Path, Path]:
    """Create and return (run_dir, checkpoint_dir, sample_dir)."""

    run_name = (
        "cross_attn" if use_cross_attention
        else "no_cross_attn"
    )

    run_dir = TEXT_GAN_DIR / "outputs" / run_name

    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, checkpoint_dir, sample_dir


# ---------------------------------------------------------
# Fixed Evaluation Captions (2 per category)
# ---------------------------------------------------------

EVAL_CAPTIONS = {
    "car": [
        "A red car parked on a street.",
        "A silver car driving on the highway.",
    ],
    "chair": [
        "A wooden chair next to a table.",
        "An office chair in a room.",
    ],
    "bottle": [
        "A glass bottle on a kitchen counter.",
        "A water bottle sitting on a desk.",
    ],
    "bird": [
        "A small bird perched on a branch.",
        "A colorful bird flying in the sky.",
    ],
    "boat": [
        "A sailboat on calm blue water.",
        "A small boat docked at a pier.",
    ],
}


# ---------------------------------------------------------
# Prepare Fixed Evaluation Tensors
# ---------------------------------------------------------

def prepare_eval_tensors(
    embedding_cache: dict,
    dataset: TextCOCODataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare fixed noise, text embeddings, and labels for
    evaluation sample generation.

    Returns fixed_noise, fixed_pooled, fixed_tokens,
    fixed_masks, fixed_labels — all on `device`.
    """

    from transformers import AutoTokenizer
    from text_encoder import TextEncoder
    from text_config import TEXT_ENCODER_NAME, MAX_CAPTION_LENGTH

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_ENCODER_NAME
    )

    # Collect all eval captions with labels
    captions = []
    labels = []

    for cat_index, cat_name in enumerate(CATEGORIES):
        for caption in EVAL_CAPTIONS[cat_name]:
            captions.append(caption)
            labels.append(cat_index)

    # Tokenize
    encodings = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=MAX_CAPTION_LENGTH,
        return_tensors="pt",
    )

    # Encode with the text encoder
    encoder = TextEncoder().to(device)
    encoder.eval()

    with torch.no_grad():
        token_embs, pooled_embs = encoder(
            encodings["input_ids"].to(device),
            encodings["attention_mask"].to(device),
        )

    # Fixed noise (deterministic)
    num_samples = len(captions)
    fixed_noise = torch.randn(
        num_samples, NOISE_DIM,
        device=device,
    )

    fixed_labels = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )

    return (
        fixed_noise,
        pooled_embs,
        token_embs,
        encodings["attention_mask"].to(device),
        fixed_labels,
    )


# ---------------------------------------------------------
# Collate Function
# ---------------------------------------------------------

def collate_with_cache(
    batch: list,
    embedding_cache: dict,
    indices: list[int],
) -> tuple:
    """
    Collate a batch and look up cached text embeddings.

    Returns:
        images, token_embs, pooled_embs, text_masks, labels
    """

    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[3] for b in batch])

    token_embs = []
    pooled_embs = []
    text_masks = []

    for idx in indices:
        tok, pool, mask = embedding_cache[idx]
        token_embs.append(tok)
        pooled_embs.append(pool)
        text_masks.append(mask)

    token_embs = torch.stack(token_embs)
    pooled_embs = torch.stack(pooled_embs)
    text_masks = torch.stack(text_masks)

    return images, token_embs, pooled_embs, text_masks, labels


# ---------------------------------------------------------
# Custom Sampler-aware DataLoader
#
# We need dataset indices alongside the batches to look
# up cached embeddings. Using a manual batch loop.
# ---------------------------------------------------------

class IndexedDataLoader:
    """
    Wraps a DataLoader to also yield the dataset indices
    for each batch (needed for cache lookups).
    """

    def __init__(
        self,
        dataset: TextCOCODataset,
        embedding_cache: dict,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.embedding_cache = embedding_cache
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Yield (images, token_embs, pooled_embs, text_masks, labels)
        for each batch.
        """

        indices = list(range(len(self.dataset)))

        if self.shuffle:
            import random
            random.shuffle(indices)

        for start in range(
            0, len(indices), self.batch_size
        ):
            batch_indices = indices[
                start : start + self.batch_size
            ]

            if (
                self.drop_last
                and len(batch_indices) < self.batch_size
            ):
                break

            # Load dataset items
            batch_items = [
                self.dataset[i] for i in batch_indices
            ]

            # Collate with cache
            yield collate_with_cache(
                batch_items,
                self.embedding_cache,
                batch_indices,
            )


# ---------------------------------------------------------
# Log Gamma Values
# ---------------------------------------------------------

def log_gammas(
    generator: TextAttentionGenerator,
    epoch: int,
    log_file: Path,
) -> None:
    """Log gamma values to console and a log file."""

    gammas = generator.get_gamma_values()

    gamma_str = " | ".join(
        f"{name}: {value:.6f}"
        for name, value in gammas.items()
    )

    print(f"  Gamma values: {gamma_str}")

    # Append to log file
    with open(log_file, "a") as f:
        f.write(
            f"Epoch {epoch:03d} | {gamma_str}\n"
        )


# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------

def train() -> None:

    args = parse_args()

    # -------------------------------------------------
    # Setup
    # -------------------------------------------------

    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    run_dir, checkpoint_dir, sample_dir = setup_directories(
        args.use_cross_attention
    )

    gamma_log = run_dir / "gamma_log.txt"

    # -------------------------------------------------
    # Print Configuration
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("TEXT-CONDITIONED GAN TRAINING")
    print("=" * 60)

    print(f"Device            : {device}")
    print(f"Cross-attention   : {args.use_cross_attention}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Learning rate     : {args.lr}")
    print(f"Seed              : {args.seed}")
    print(f"Categories        : {CATEGORIES}")
    print(f"Output directory  : {run_dir}")

    if args.resume:
        print(f"Resume from       : {args.resume}")

    # -------------------------------------------------
    # Dataset & Embedding Cache
    # -------------------------------------------------

    dataset = TextCOCODataset()

    embedding_cache = encode_and_cache(dataset)

    loader = IndexedDataLoader(
        dataset=dataset,
        embedding_cache=embedding_cache,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    print(f"\nDataset samples   : {len(dataset)}")
    print(f"Batches per epoch : {len(loader)}")

    # -------------------------------------------------
    # Models
    # -------------------------------------------------

    generator = TextAttentionGenerator(
        use_cross_attention=args.use_cross_attention,
    ).to(device)

    discriminator = TextDiscriminator().to(device)

    print(
        f"\nGenerator params  : "
        f"{sum(p.numel() for p in generator.parameters()):,}"
    )
    print(
        f"Discriminator params: "
        f"{sum(p.numel() for p in discriminator.parameters()):,}"
    )

    # -------------------------------------------------
    # Optimizers
    # -------------------------------------------------

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(BETA1, BETA2),
    )

    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(BETA1, BETA2),
    )

    # -------------------------------------------------
    # Fixed Evaluation Tensors
    # -------------------------------------------------

    torch.manual_seed(args.seed)

    fixed_noise, fixed_pooled, fixed_tokens, fixed_masks, fixed_labels = (
        prepare_eval_tensors(
            embedding_cache, dataset, device,
        )
    )

    # -------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------

    start_epoch = 0

    if args.resume:

        checkpoint_path = Path(args.resume)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print(f"\nLoading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        generator.load_state_dict(
            checkpoint["generator_state_dict"]
        )

        discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )

        optimizer_G.load_state_dict(
            checkpoint["optimizer_G_state_dict"]
        )

        optimizer_D.load_state_dict(
            checkpoint["optimizer_D_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        print(f"Resumed from epoch {start_epoch}")

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(start_epoch, args.epochs):

        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        for batch_index, batch_data in enumerate(loader):

            (
                real_images,
                token_embs,
                pooled_embs,
                text_masks,
                labels,
            ) = batch_data

            real_images = real_images.to(device)
            token_embs = token_embs.to(device)
            pooled_embs = pooled_embs.to(device)
            text_masks = text_masks.to(device)
            labels = labels.to(device)

            batch_size = real_images.size(0)


            # =============================================
            # Train Discriminator
            # =============================================

            optimizer_D.zero_grad()

            # Real images
            real_scores = discriminator(
                real_images,
                pooled_embs,
                labels,
            )

            # Fake images
            noise = torch.randn(
                batch_size, NOISE_DIM,
                device=device,
            )

            fake_images = generator(
                noise,
                pooled_embs,
                token_embs,
                text_masks,
            )

            fake_scores = discriminator(
                fake_images.detach(),
                pooled_embs,
                labels,
            )

            # Hinge loss
            real_loss = torch.relu(
                1.0 - real_scores
            ).mean()

            fake_loss = torch.relu(
                1.0 + fake_scores
            ).mean()

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()


            # =============================================
            # Train Generator
            # =============================================

            optimizer_G.zero_grad()

            noise = torch.randn(
                batch_size, NOISE_DIM,
                device=device,
            )

            fake_images = generator(
                noise,
                pooled_embs,
                token_embs,
                text_masks,
            )

            fake_scores = discriminator(
                fake_images,
                pooled_embs,
                labels,
            )

            # Generator wants high scores
            g_loss = -fake_scores.mean()

            g_loss.backward()
            optimizer_G.step()


            # =============================================
            # Record Losses
            # =============================================

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()


            # =============================================
            # Progress
            # =============================================

            if (batch_index + 1) % 25 == 0:
                print(
                    f"  Batch {batch_index + 1}/{len(loader)} | "
                    f"D: {d_loss.item():.4f} | "
                    f"G: {g_loss.item():.4f}"
                )


        # -------------------------------------------------
        # Epoch Summary
        # -------------------------------------------------

        avg_g_loss = total_g_loss / len(loader)
        avg_d_loss = total_d_loss / len(loader)

        print(
            f"\n  Epoch {epoch + 1} completed"
        )
        print(
            f"  Avg D Loss: {avg_d_loss:.4f}"
        )
        print(
            f"  Avg G Loss: {avg_g_loss:.4f}"
        )


        # -------------------------------------------------
        # Log Gamma Values
        # -------------------------------------------------

        log_gammas(generator, epoch + 1, gamma_log)


        # -------------------------------------------------
        # Generate Sample Grid
        # -------------------------------------------------

        generator.eval()

        with torch.no_grad():

            samples = generator(
                fixed_noise,
                fixed_pooled,
                fixed_tokens,
                fixed_masks,
            )

        sample_path = (
            sample_dir / f"epoch_{epoch + 1:03d}.png"
        )

        # 2 samples per category, 5 categories = 10 images
        # Arrange as 2 rows × 5 columns
        save_image(
            samples,
            sample_path,
            nrow=len(CATEGORIES),
            normalize=True,
            value_range=(-1, 1),
        )

        print(
            f"  Samples saved: {sample_path.name}"
        )


        # -------------------------------------------------
        # Save Checkpoint
        # -------------------------------------------------

        checkpoint = {
            "epoch": epoch + 1,
            "use_cross_attention": args.use_cross_attention,
            "generator_state_dict": (
                generator.state_dict()
            ),
            "discriminator_state_dict": (
                discriminator.state_dict()
            ),
            "optimizer_G_state_dict": (
                optimizer_G.state_dict()
            ),
            "optimizer_D_state_dict": (
                optimizer_D.state_dict()
            ),
            "fixed_noise": fixed_noise.cpu(),
            "gamma_values": generator.get_gamma_values(),
            "avg_g_loss": avg_g_loss,
            "avg_d_loss": avg_d_loss,
        }

        checkpoint_path = (
            checkpoint_dir
            / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        )

        torch.save(checkpoint, checkpoint_path)

        print(
            f"  Checkpoint saved: {checkpoint_path.name}"
        )


    # -------------------------------------------------
    # Final Report
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Total epochs      : {args.epochs}")
    print(f"Cross-attention   : {args.use_cross_attention}")
    print(f"Checkpoints       : {checkpoint_dir}")
    print(f"Sample grids      : {sample_dir}")
    print(f"Gamma log         : {gamma_log}")

    print("\nTraining finished successfully.")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    train()
