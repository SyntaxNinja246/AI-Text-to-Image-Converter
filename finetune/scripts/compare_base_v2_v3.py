"""
PixelDream AI - Task 1 Model Comparison Script
================================================

Compares model variants on the same 5 benchmark prompts
and seeds used by evaluate_lora_v2.py and evaluate_lora_v3.py:

  1. Base: segmind/tiny-sd (NO LoRA attached)
  2. V2 LoRA (rank 4, alpha 4): tested at scales 0.5, 0.75, 1.0
  3. V3 LoRA (rank 8, alpha 8): tested at scale 1.0

Outputs are saved under:
    finetune/outputs/model_comparison/
        base/
        v2_scale_0.5/
        v2_scale_0.75/
        v2_scale_1.0/
        v3_scale_1.0/
        comparison_metadata.json

Usage:
    .\\venv\\Scripts\\python.exe finetune/scripts/compare_base_v2_v3.py

Optional flags:
    --num_inference_steps  (default: 10)
    --guidance_scale       (default: 7.5)
    --image_size           (default: 256)
    --output_dir           (default: finetune/outputs/model_comparison)
    --verify_only          (only verify loading/mapping without generating images)
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


# ============================================================
# Constants — mirror evaluate_lora_v2.py / evaluate_lora_v3.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "segmind/tiny-sd"
DEVICE = torch.device("cpu")

V2_LORA_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v2"
V3_LORA_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "outputs" / "model_comparison"

# Exactly the same 5 prompts and seeds as evaluate_lora_v2.py / evaluate_lora_v3.py
EVAL_PROMPTS = [
    "a motorcycle parked outdoors",
    "a cozy room with chairs",
    "a person standing in a kitchen",
    "a desk with electronic equipment",
    "a group of people outdoors",
]
EVAL_SEEDS = [101, 102, 103, 104, 105]

# Comparison variants with exact architecture configurations
VARIANTS = [
    {
        "label": "base",
        "lora_version": "Base",
        "lora_dir": None,
        "lora_scale": None,
        "lora_rank": None,
        "lora_alpha": None,
        "lora_dropout": None,
    },
    {
        "label": "v2_scale_0.5",
        "lora_version": "V2",
        "lora_dir": V2_LORA_DIR,
        "lora_scale": 0.5,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
    },
    {
        "label": "v2_scale_0.75",
        "lora_version": "V2",
        "lora_dir": V2_LORA_DIR,
        "lora_scale": 0.75,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
    },
    {
        "label": "v2_scale_1.0",
        "lora_version": "V2",
        "lora_dir": V2_LORA_DIR,
        "lora_scale": 1.0,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
    },
    {
        "label": "v3_scale_1.0",
        "lora_version": "V3",
        "lora_dir": V3_LORA_DIR,
        "lora_scale": 1.0,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelDream AI - Task 1 Model Comparison"
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
        help="Output resolution in pixels (default: 256)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify weight loading, mapping, and scaling without generating images",
    )
    return parser.parse_args()


def verify_prerequisites(output_dir: Path) -> None:
    """Verify all required adapter paths exist and print a pre-flight report."""

    print("=" * 70)
    print("PRE-FLIGHT VERIFICATION")
    print("=" * 70)

    # Verify adapter files
    adapter_checks = [
        ("V2 LoRA adapter", V2_LORA_DIR / "pytorch_lora_weights.safetensors"),
        ("V3 LoRA adapter", V3_LORA_DIR / "pytorch_lora_weights.safetensors"),
    ]
    all_ok = True
    for name, path in adapter_checks:
        exists = path.exists()
        status = "OK   " if exists else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(
            "\nOne or more LoRA adapter files are missing.\n"
            "Please run training before running comparison."
        )

    # Print prompt/seed verification
    print()
    print("Benchmark prompts and seeds (identical to evaluate_lora_v2.py / evaluate_lora_v3.py):")
    for i, (prompt, seed) in enumerate(zip(EVAL_PROMPTS, EVAL_SEEDS), start=1):
        print(f"  [{i}] seed={seed}  \"{prompt}\"")

    # Print output structure
    print()
    print("Output directory structure:")
    for variant in VARIANTS:
        print(f"  {output_dir.name}/{variant['label']}/")

    print()
    print("All prerequisite checks passed.\n")


def prepare_pipeline_for_variant(variant: dict) -> StableDiffusionPipeline:
    """
    Prepares a pipeline for the given variant:
      - For 'base': pure pretrained Tiny-SD base model (NO LoRA attached).
      - For LoRA variants: attaches PEFT LoRA structure, maps and loads the 144
        saved weights, performs strict verification (halting if any tensor is missing),
        and applies the requested LoRA scale using PEFT's layer.set_scale().
    """
    label = variant["label"]
    lora_dir = variant["lora_dir"]
    lora_scale = variant["lora_scale"]
    lora_version = variant["lora_version"]

    print(f"\n{'=' * 70}")
    print(f"PREPARING PIPELINE: {label}")
    print(f"{'=' * 70}")

    # Load fresh base pipeline
    print(f"Loading pretrained Tiny-SD base pipeline for '{label}'...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(DEVICE)
    pipeline.set_progress_bar_config(disable=False)

    # Base variant: pure base model, NO LoRA
    if lora_dir is None:
        has_lora = any("lora" in name.lower() for name, _ in pipeline.unet.named_parameters())
        if has_lora:
            raise RuntimeError("CRITICAL ERROR: Base model UNet unexpectedly contains LoRA parameters!")
        print("Base model loaded successfully with NO LoRA attached.\n")
        return pipeline

    # LoRA variant setup
    adapter_file = Path(lora_dir) / "pytorch_lora_weights.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(f"LoRA adapter file not found: {adapter_file}")

    lora_rank = variant["lora_rank"]
    lora_alpha = variant["lora_alpha"]
    lora_dropout = variant["lora_dropout"]

    print(f"Attaching PEFT LoRA structure ({lora_version}: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
        init_lora_weights=False,
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    # Map saved training keys to PEFT format (exact evaluate_lora_v2/v3 logic)
    print("Mapping and loading saved LoRA weights...")
    saved_state = load_file(str(adapter_file))
    peft_state = {}
    for key, value in saved_state.items():
        new_key = "base_model.model." + key
        new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
        peft_state[new_key] = value

    missing, unexpected = pipeline.unet.load_state_dict(peft_state, strict=False)

    expected_lora_keys = {
        key for key in pipeline.unet.state_dict().keys()
        if ".lora_A." in key or ".lora_B." in key
    }
    loaded_lora_keys = {
        key for key in peft_state.keys()
        if key in pipeline.unet.state_dict()
    }
    missing_lora_keys = expected_lora_keys - loaded_lora_keys

    num_saved = len(saved_state)
    num_mapped = len(peft_state)
    num_expected = len(expected_lora_keys)
    num_loaded = len(loaded_lora_keys)
    num_missing = len(missing_lora_keys)
    num_unexpected = len(unexpected)

    # Print verification report
    print()
    print("=" * 70)
    print(f"{lora_version}:")
    print(f"Saved LoRA tensors : {num_saved}")
    print(f"Mapped LoRA tensors: {num_mapped}")
    print(f"Expected LoRA keys : {num_expected}")
    print(f"Loaded LoRA keys   : {num_loaded}")
    print(f"Missing LoRA keys  : {num_missing}")
    print(f"Unexpected keys    : {num_unexpected}")
    print("=" * 70)

    # Strict error halt if fewer than all expected LoRA tensors loaded
    if missing_lora_keys:
        print("ERROR - Missing LoRA keys:")
        for key in sorted(missing_lora_keys):
            print(f"  - {key}")
        raise RuntimeError(
            f"LoRA loading failed for {label}: Missing {num_missing} LoRA keys!"
        )

    if num_unexpected != 0:
        print("ERROR - Unexpected keys:")
        for key in sorted(unexpected):
            print(f"  - {key}")
        raise RuntimeError(
            f"LoRA loading failed for {label}: Found {num_unexpected} unexpected keys!"
        )

    if num_saved != 144 or num_mapped != 144 or num_expected != 144 or num_loaded != 144:
        raise RuntimeError(
            f"LoRA verification failed for {label}: expected 144 tensors, but got "
            f"(saved={num_saved}, mapped={num_mapped}, expected={num_expected}, loaded={num_loaded})."
        )

    # Apply LoRA scale across all LoraLayers using PEFT native set_scale
    lora_layers = [m for m in pipeline.unet.modules() if isinstance(m, LoraLayer)]
    for layer in lora_layers:
        layer.set_scale("default", lora_scale)

    # Verify that every LoRA layer now carries the requested scale
    applied_scales = {layer.scaling["default"] for layer in lora_layers}
    if applied_scales != {lora_scale}:
        raise RuntimeError(
            f"Scale mismatch for {label}: expected {{{lora_scale}}}, but found {applied_scales}"
        )

    print(f"VERIFICATION PASSED: All 144 LoRA tensors verified and mapped without error.")
    print(f"LoRA scale {lora_scale} successfully applied to all {len(lora_layers)} LoRA layers.\n")

    return pipeline


def generate_images_for_variant(
    pipeline: StableDiffusionPipeline,
    variant: dict,
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    image_size: int,
) -> list[dict]:
    """
    Generate all 5 benchmark images for a single variant.
    Returns a list of metadata records.
    """
    label = variant["label"]
    lora_dir = variant["lora_dir"]
    lora_scale = variant["lora_scale"]
    lora_version = variant["lora_version"]

    variant_dir = output_dir / label
    variant_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print(f"Generating 5 benchmark images for variant '{label}'...")

    for i, (prompt, seed) in enumerate(zip(EVAL_PROMPTS, EVAL_SEEDS), start=1):
        print(f"\n  [{i}/{len(EVAL_PROMPTS)}] seed={seed} | \"{prompt}\"")

        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=image_size,
            height=image_size,
            generator=generator,
        )
        image = result.images[0]

        filename = f"img_{i:02d}_seed{seed}.png"
        filepath = variant_dir / filename
        image.save(filepath)

        print(f"    -> Saved: {filepath}")

        records.append({
            "model": BASE_MODEL,
            "variant": label,
            "lora_version": lora_version,
            "lora_path": str(lora_dir) if lora_dir is not None else "none",
            "lora_scale": lora_scale if lora_scale is not None else "none",
            "lora_rank": variant["lora_rank"] if variant["lora_rank"] is not None else "none",
            "prompt": prompt,
            "seed": seed,
            "resolution": f"{image_size}x{image_size}",
            "inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generated_filename": str(filepath),
        })

        del image
        del result
        gc.collect()

    return records


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PixelDream AI - Task 1 Model Comparison")
    print("=" * 70)
    print(f"Base model         : {BASE_MODEL}")
    print(f"Device             : {DEVICE}")
    print(f"Resolution         : {args.image_size}x{args.image_size}")
    print(f"Inference steps    : {args.num_inference_steps}")
    print(f"Guidance scale     : {args.guidance_scale}")
    print(f"Output directory   : {output_dir}")
    print(f"Variants           : {len(VARIANTS)}")
    print(f"Images per variant : {len(EVAL_PROMPTS)}")
    print(f"Total images       : {len(VARIANTS) * len(EVAL_PROMPTS)}")
    if args.verify_only:
        print("Mode               : VERIFY-ONLY (no images will be generated)")
    print()

    verify_prerequisites(output_dir)

    all_metadata = []

    for variant_index, variant in enumerate(VARIANTS, start=1):
        print(f"\n{'#' * 70}")
        print(f"# VARIANT {variant_index}/{len(VARIANTS)}: {variant['label']}")
        print(f"{'#' * 70}")

        # Prepare pipeline specifically for this variant (clean base or verified PEFT LoRA)
        pipeline = prepare_pipeline_for_variant(variant)

        if args.verify_only:
            print(f"Verification-only mode: successfully validated '{variant['label']}'.")
            del pipeline
            gc.collect()
            continue

        records = generate_images_for_variant(
            pipeline=pipeline,
            variant=variant,
            output_dir=output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            image_size=args.image_size,
        )

        all_metadata.extend(records)
        print(f"\n  Variant '{variant['label']}' completed ({len(records)} images).")

        # Explicitly clean up pipeline to ensure absolute isolation between variants
        del pipeline
        gc.collect()

    if args.verify_only:
        print()
        print("=" * 70)
        print("ALL VARIANTS VERIFIED SUCCESSFULLY (VERIFY-ONLY)")
        print("=" * 70)
        return

    # Save combined metadata
    metadata_file = output_dir / "comparison_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nTotal images generated : {len(all_metadata)}")
    print(f"Output directory       : {output_dir}")
    print(f"Metadata file          : {metadata_file}")
    print()
    print("Output structure:")
    for variant in VARIANTS:
        variant_dir = output_dir / variant["label"]
        n = len(list(variant_dir.glob("*.png")))
        print(f"  {output_dir.name}/{variant['label']}/  ({n} images)")
    print()


if __name__ == "__main__":
    main()
