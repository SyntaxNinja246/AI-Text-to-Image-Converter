"""
Frozen Text Encoder with Disk Caching
======================================

Wraps DistilBERT-base-uncased as a frozen feature extractor.

    forward(input_ids, attention_mask)
        → token_embeddings  (B, T, 768)
        → pooled_embedding  (B, 768)

The encode_and_cache() function runs the encoder once over the
entire dataset and saves the outputs to a .pt file.  Subsequent
training loads from disk and never instantiates the encoder.
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from text_config import (
    TEXT_ENCODER_NAME,
    TEXT_EMBED_DIM,
    CACHE_DIR,
)


# ---------------------------------------------------------
# Frozen Text Encoder
# ---------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Frozen DistilBERT text encoder.

    All parameters are frozen — this module is used only
    for inference (no gradients flow through it).
    """

    def __init__(
        self,
        model_name: str = TEXT_ENCODER_NAME,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = TEXT_EMBED_DIM

        print(f"\nText encoder: {model_name}")
        print(f"Embed dim   : {self.embed_dim}")
        print(f"Parameters  : {sum(p.numel() for p in self.model.parameters()):,} (all frozen)")


    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of tokenized captions.

        Args:
            input_ids:      (B, T) token IDs
            attention_mask: (B, T) 1 for real tokens, 0 for padding

        Returns:
            token_embeddings: (B, T, 768) per-token hidden states
            pooled_embedding: (B, 768)    mean-pooled sentence vector
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Last hidden state: (B, T, 768)
        token_embeddings = outputs.last_hidden_state

        # Mean-pool over non-padding tokens
        # Expand mask to (B, T, 1) for broadcasting
        mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .float()
        )

        sum_embeddings = (
            token_embeddings * mask_expanded
        ).sum(dim=1)

        # Count of real tokens per sample
        token_counts = (
            mask_expanded.sum(dim=1).clamp(min=1)
        )

        pooled_embedding = sum_embeddings / token_counts

        return token_embeddings, pooled_embedding


# ---------------------------------------------------------
# Disk Caching
# ---------------------------------------------------------

def encode_and_cache(
    dataset,
    cache_path: Path | None = None,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Encode all captions in the dataset and save to disk.

    The cache is a dict keyed by dataset index:
        index → (token_embeddings, pooled_embedding, attention_mask)

    All tensors are stored on CPU as float32.

    Args:
        dataset:    TextCOCODataset instance
        cache_path: Where to save the .pt file
        batch_size: Encoding batch size
        device:     Device for the text encoder

    Returns:
        The cache dict (also saved to disk).
    """

    if cache_path is None:
        cache_path = CACHE_DIR / "text_embeddings.pt"

    cache_path = Path(cache_path)

    # If cache already exists, load and return
    if cache_path.exists():

        print(f"\nLoading cached embeddings from:\n{cache_path}")

        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        print(f"Cached entries: {len(cache)}")
        return cache

    # -------------------------------------------------
    # Build cache from scratch
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("ENCODING CAPTIONS (one-time)")
    print("=" * 60)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = TextEncoder().to(device)
    encoder.eval()

    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    total = len(dataset)

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        # Collect a batch
        token_ids_batch = []
        masks_batch = []

        for i in range(start, end):
            _, token_ids, attention_mask, _ = dataset[i]
            token_ids_batch.append(token_ids)
            masks_batch.append(attention_mask)

        token_ids_batch = torch.stack(
            token_ids_batch
        ).to(device)

        masks_batch = torch.stack(
            masks_batch
        ).to(device)

        # Encode
        token_embs, pooled_emb = encoder(
            token_ids_batch,
            masks_batch,
        )

        # Store each sample individually (on CPU)
        for i, global_idx in enumerate(
            range(start, end)
        ):
            cache[global_idx] = (
                token_embs[i].cpu(),
                pooled_emb[i].cpu(),
                masks_batch[i].cpu(),
            )

        if (end % (batch_size * 10) == 0) or (end == total):
            print(
                f"  Encoded {end}/{total} captions"
            )

    # Save
    torch.save(cache, cache_path)

    print(f"\nCache saved to:\n{cache_path}")
    print(f"Cache size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")

    return cache


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    import sys
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in sys.path:
        sys.path.insert(0, str(_this_dir))

    from text_dataset import TextCOCODataset

    print("\n" + "=" * 60)
    print("TEXT ENCODER TEST")
    print("=" * 60)

    # Test 1: Direct forward pass
    encoder = TextEncoder()
    encoder.eval()

    dummy_ids = torch.randint(0, 1000, (2, 64))
    dummy_mask = torch.ones(2, 64, dtype=torch.long)

    token_embs, pooled_emb = encoder(
        dummy_ids, dummy_mask
    )

    print(f"\nToken embeddings : {tuple(token_embs.shape)}")
    print(f"Pooled embedding : {tuple(pooled_emb.shape)}")

    # Test 2: Cache encoding
    print("\nBuilding cache from dataset...")

    dataset = TextCOCODataset()

    cache = encode_and_cache(dataset)

    sample = cache[0]

    print(f"\nCached token embs : {tuple(sample[0].shape)}")
    print(f"Cached pooled emb : {tuple(sample[1].shape)}")
    print(f"Cached mask       : {tuple(sample[2].shape)}")

    print("\nText encoder test completed successfully.")
