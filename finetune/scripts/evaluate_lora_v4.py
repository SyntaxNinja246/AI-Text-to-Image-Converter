"""
PixelDream AI - Task 1 LoRA Evaluation v4
============================================

Evaluates the Task 1 V4 LoRA adapter across 3 scales (0.5, 0.75, 1.0) on the
5 standard benchmark prompts and fixed seeds (101-105).

Safety and Verification:
  - Uses the proven custom PEFT key-mapping mechanism from evaluate_lora_v2.py.
  - Strictly verifies all 144 LoRA tensors before generation.
  - Stops with an error if missing keys > 0 or unexpected keys > 0.
  - Supports --verify_only flag for fast pre-flight verification without generating images.

Output Structure:
  finetune/outputs/lora_evaluation_v4/
      scale_0.5/
      scale_0.75/
      scale_1.0/
      evaluation_metadata.json
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "outputs" / "lora_evaluation_v4"

DEVICE = torch.device("cpu")
BASE_MODEL = "segmind/tiny-sd"

EVAL_PROMPTS = [
    "a motorcycle parked outdoors",
    "a cozy room with chairs",
    "a person standing in a kitchen",
    "a desk with electronic equipment",
    "a group of people outdoors",
]
EVAL_SEEDS = [101, 102, 103, 104, 105]

SCALES_TO_TEST = [0.5, 0.75, 1.0]

# V4 LoRA Architecture (Rank 4, Alpha 4)
LORA_RANK = 4
LORA_ALPHA = 4
LORA_DROPOUT = 0.0
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelDream AI - Task 1 LoRA Evaluation v4"
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=str(DEFAULT_LORA_DIR),
        help=f"Path to LoRA weights directory (default: {DEFAULT_LORA_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Path to output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=SCALES_TO_TEST,
        help=f"LoRA scales to test (default: {SCALES_TO_TEST})",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of denoising steps (default: 10)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Evaluation image resolution (default: 256)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify weight loading and mapping without generating images",
    )
    return parser.parse_args()


def load_and_verify_pipeline(lora_dir: Path) -> StableDiffusionPipeline:
    """
    Loads base Tiny-SD, attaches the PEFT LoRA structure, maps saved keys to PEFT format,
    loads the state dict, and strictly verifies all 144 LoRA tensors.
    """
    adapter_file = lora_dir / "pytorch_lora_weights.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"LoRA adapter weights not found at: {adapter_file}\n"
            f"Please ensure training has completed and weights are saved."
        )

    # 1. Load the original Tiny-SD base model
    print("Step 1: Loading pretrained Tiny-SD base model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(DEVICE)
    print("Base model loaded successfully.\n")

    # 2. Attach PEFT LoRA structure to UNet
    print("Step 2: Attaching PEFT LoRA structure to UNet (Rank 4, Alpha 4)...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=TARGET_MODULES,
        init_lora_weights=False,
    )

    pipeline.unet = get_peft_model(
        pipeline.unet,
        lora_config,
    )
    print("LoRA structure created successfully.\n")

    # 3. Correctly map and load the saved LoRA tensors
    print("Step 3: Mapping and loading saved LoRA weights...")
    saved_state = load_file(str(adapter_file))

    # Convert training-time names to PEFT names
    peft_state = {}
    for key, value in saved_state.items():
        new_key = "base_model.model." + key
        new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
        peft_state[new_key] = value

    missing, unexpected = pipeline.unet.load_state_dict(
        peft_state,
        strict=False,
    )

    expected_lora_keys = {
        key for key in pipeline.unet.state_dict().keys()
        if ".lora_A." in key or ".lora_B." in key
    }

    loaded_lora_keys = {
        key for key in peft_state.keys()
        if key in pipeline.unet.state_dict()
    }

    missing_lora_keys = expected_lora_keys - loaded_lora_keys

    # 4. Strict Verification
    num_saved = len(saved_state)
    num_mapped = len(peft_state)
    num_expected = len(expected_lora_keys)
    num_loaded = len(loaded_lora_keys)
    num_missing = len(missing_lora_keys)
    num_unexpected = len(unexpected)

    print()
    print("=" * 70)
    print("Step 4: LoRA Tensor Verification (v4)")
    print("=" * 70)
    print(f"Saved LoRA tensors    : {num_saved}  (expected: 144)")
    print(f"Mapped LoRA tensors   : {num_mapped}  (expected: 144)")
    print(f"Expected LoRA keys    : {num_expected}  (expected: 144)")
    print(f"Loaded LoRA keys      : {num_loaded}  (expected: 144)")
    print(f"Missing LoRA keys     : {num_missing}  (expected: 0)")
    print(f"Unexpected keys       : {num_unexpected}  (expected: 0)")
    print(f"Base-model keys unmod.: {len(missing)} (normal - LoRA adapter only)")
    print("=" * 70)

    if missing_lora_keys:
        print("ERROR - Missing LoRA keys:")
        for key in sorted(missing_lora_keys):
            print(f"  - {key}")
        raise RuntimeError(f"LoRA loading failed: Missing {num_missing} LoRA weights.")

    if num_unexpected != 0:
        print("ERROR - Unexpected keys:")
        for key in sorted(unexpected):
            print(f"  - {key}")
        raise RuntimeError(f"LoRA loading failed: Found {num_unexpected} unexpected keys.")

    assert num_saved == 144, f"Verification failed: Saved LoRA tensors = {num_saved}, expected 144"
    assert num_mapped == 144, f"Verification failed: Mapped LoRA tensors = {num_mapped}, expected 144"
    assert num_expected == 144, f"Verification failed: Expected LoRA keys = {num_expected}, expected 144"
    assert num_loaded == 144, f"Verification failed: Loaded LoRA keys = {num_loaded}, expected 144"
    assert num_missing == 0, f"Verification failed: Missing LoRA keys = {num_missing}, expected 0"
    assert num_unexpected == 0, f"Verification failed: Unexpected LoRA keys = {num_unexpected}, expected 0"

    print("VERIFICATION PASSED: All 144 LoRA v4 tensors verified and mapped without error.\n")
    return pipeline


def apply_lora_scale(pipeline: StableDiffusionPipeline, scale: float) -> None:
    """Sets the LoRA scaling factor on all LoRA layers."""
    lora_layers = [m for m in pipeline.unet.modules() if isinstance(m, LoraLayer)]
    for layer in lora_layers:
        layer.set_scale("default", scale)

    applied = {layer.scaling["default"] for layer in lora_layers}
    if applied != {scale}:
        raise RuntimeError(f"Scale mismatch: expected {{{scale}}}, but got {applied}")


def main():
    args = parse_args()

    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PixelDream AI - Task 1 LoRA Evaluation v4")
    print("=" * 70)
    print(f"Device          : {DEVICE}")
    print(f"Base model      : {BASE_MODEL}")
    print(f"LoRA directory  : {lora_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scales to test  : {args.scales}")
    print(f"Inference steps : {args.num_inference_steps}")
    print(f"Guidance scale  : {args.guidance_scale}")
    print(f"Resolution      : {args.image_size}x{args.image_size}")
    if args.verify_only:
        print("Mode            : VERIFY-ONLY (no image generation)")
    print("=" * 70)
    print()

    # Load and strictly verify pipeline
    pipeline = load_and_verify_pipeline(lora_dir)

    if args.verify_only:
        print("Verification-only mode completed successfully.")
        return

    # Generation across each scale
    all_metadata = []

    for scale in args.scales:
        scale_dir = output_dir / f"scale_{scale}"
        scale_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f"Generating Benchmark Images for LoRA Scale: {scale}")
        print("=" * 70)

        apply_lora_scale(pipeline, scale)

        for i, (prompt, seed) in enumerate(zip(EVAL_PROMPTS, EVAL_SEEDS), start=1):
            print(f"  [{i}/{len(EVAL_PROMPTS)}] Scale: {scale} | Seed: {seed} | Prompt: \"{prompt}\"")

            generator = torch.Generator(device="cpu").manual_seed(seed)

            result = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.image_size,
                height=args.image_size,
                generator=generator,
            )
            image = result.images[0]

            filename = f"eval_{i:02d}_seed{seed}.png"
            output_file = scale_dir / filename
            image.save(output_file)

            print(f"    -> Saved: {output_file}")

            all_metadata.append({
                "index": i,
                "scale": scale,
                "prompt": prompt,
                "seed": seed,
                "output_file": str(output_file),
                "resolution": f"{args.image_size}x{args.image_size}",
                "inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
            })

            del image
            del result
            gc.collect()

        print()

    # Save metadata JSON
    metadata_file = output_dir / "evaluation_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print("=" * 70)
    print("V4 LoRA Evaluation Completed Successfully")
    print("=" * 70)
    print(f"Total images generated : {len(all_metadata)}")
    print(f"Output directory       : {output_dir}")
    print(f"Metadata file          : {metadata_file}")
    for scale in args.scales:
        scale_dir = output_dir / f"scale_{scale}"
        n = len(list(scale_dir.glob("*.png")))
        print(f"  scale_{scale}/ : {n} images")
    print("=" * 70)


if __name__ == "__main__":
    main()
