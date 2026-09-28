"""
PixelDream AI - Task 1 Model Comparison: V2 vs V4
===================================================

Compares model variants on the identical 5 benchmark prompts and seeds:
  1. V2 @ scale 0.75 (finetune/models/tiny-sd-lora-v2)
  2. V4 @ scale 0.50 (finetune/models/tiny-sd-lora-v4)
  3. V4 @ scale 0.75 (finetune/models/tiny-sd-lora-v4)
  4. V4 @ scale 1.00 (finetune/models/tiny-sd-lora-v4)

Outputs are saved under:
  finetune/outputs/model_comparison_v2_v4/
      v2_scale_0.75/
      v4_scale_0.5/
      v4_scale_0.75/
      v4_scale_1.0/
      comparison_metadata.json

Safety and Verification:
  - Proven PEFT key-mapping mechanism for both V2 and V4.
  - Strict verification of 144 LoRA tensors before generation.
  - Zero tolerance for missing or unexpected keys.
  - Complete isolation between pipeline loads.
  - Supports --verify_only flag.
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
BASE_MODEL = "segmind/tiny-sd"
DEVICE = torch.device("cpu")

V2_LORA_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v2"
V4_LORA_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "outputs" / "model_comparison_v2_v4"

EVAL_PROMPTS = [
    "a motorcycle parked outdoors",
    "a cozy room with chairs",
    "a person standing in a kitchen",
    "a desk with electronic equipment",
    "a group of people outdoors",
]
EVAL_SEEDS = [101, 102, 103, 104, 105]

VARIANTS = [
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
        "label": "v4_scale_0.5",
        "lora_version": "V4",
        "lora_dir": V4_LORA_DIR,
        "lora_scale": 0.5,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
    },
    {
        "label": "v4_scale_0.75",
        "lora_version": "V4",
        "lora_dir": V4_LORA_DIR,
        "lora_scale": 0.75,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
    },
    {
        "label": "v4_scale_1.0",
        "lora_version": "V4",
        "lora_dir": V4_LORA_DIR,
        "lora_scale": 1.0,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelDream AI - Task 1 Model Comparison (V2 vs V4)"
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
        "--v2_lora_dir",
        type=str,
        default=str(V2_LORA_DIR),
        help=f"Path to V2 LoRA directory (default: {V2_LORA_DIR})",
    )
    parser.add_argument(
        "--v4_lora_dir",
        type=str,
        default=str(V4_LORA_DIR),
        help=f"Path to V4 LoRA directory (default: {V4_LORA_DIR})",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify weight loading and mapping without generating images",
    )
    return parser.parse_args()


def prepare_pipeline_for_variant(variant: dict) -> StableDiffusionPipeline:
    """
    Attaches PEFT LoRA, loads and maps all 144 weights, verifies them strictly,
    and sets the specified scale.
    """
    label = variant["label"]
    lora_dir = Path(variant["lora_dir"])
    lora_scale = variant["lora_scale"]
    lora_version = variant["lora_version"]

    adapter_file = lora_dir / "pytorch_lora_weights.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"LoRA adapter file not found for {label}: {adapter_file}\n"
            f"Please ensure {lora_version} training has completed."
        )

    print(f"\nLoading fresh base pipeline for '{label}'...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(DEVICE)

    lora_config = LoraConfig(
        r=variant["lora_rank"],
        lora_alpha=variant["lora_alpha"],
        lora_dropout=variant["lora_dropout"],
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights=False,
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    # Key mapping
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

    print(f"[{label}] LoRA Verification:")
    print(f"  Saved tensors   : {num_saved} (expected: 144)")
    print(f"  Mapped tensors  : {num_mapped} (expected: 144)")
    print(f"  Expected keys   : {num_expected} (expected: 144)")
    print(f"  Loaded keys     : {num_loaded} (expected: 144)")
    print(f"  Missing keys    : {num_missing} (expected: 0)")
    print(f"  Unexpected keys : {num_unexpected} (expected: 0)")

    if missing_lora_keys:
        raise RuntimeError(f"Missing {num_missing} LoRA keys for {label}!")
    if num_unexpected != 0:
        raise RuntimeError(f"Found {num_unexpected} unexpected keys for {label}!")
    if num_saved != 144 or num_mapped != 144 or num_loaded != 144:
        raise RuntimeError(f"Tensor count mismatch for {label}: expected 144, got {num_loaded}")

    # Set scale
    lora_layers = [m for m in pipeline.unet.modules() if isinstance(m, LoraLayer)]
    for layer in lora_layers:
        layer.set_scale("default", lora_scale)

    applied_scales = {layer.scaling["default"] for layer in lora_layers}
    if applied_scales != {lora_scale}:
        raise RuntimeError(f"Scale mismatch for {label}: expected {{{lora_scale}}}, got {applied_scales}")

    print(f"  VERIFICATION PASSED: Scale {lora_scale} successfully applied to all {len(lora_layers)} LoRA layers.")
    return pipeline


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update variant paths from args
    variants = []
    for v in VARIANTS:
        v_copy = dict(v)
        if v_copy["lora_version"] == "V2":
            v_copy["lora_dir"] = Path(args.v2_lora_dir)
        elif v_copy["lora_version"] == "V4":
            v_copy["lora_dir"] = Path(args.v4_lora_dir)
        variants.append(v_copy)

    print("=" * 70)
    print("PixelDream AI - Task 1 Model Comparison (V2 vs V4)")
    print("=" * 70)
    print(f"Base model         : {BASE_MODEL}")
    print(f"Device             : {DEVICE}")
    print(f"Resolution         : {args.image_size}x{args.image_size}")
    print(f"Inference steps    : {args.num_inference_steps}")
    print(f"Guidance scale     : {args.guidance_scale}")
    print(f"Output directory   : {output_dir}")
    print(f"Variants count     : {len(variants)}")
    print(f"Prompts count      : {len(EVAL_PROMPTS)}")
    if args.verify_only:
        print("Mode               : VERIFY-ONLY (no images generated)")
    print("=" * 70)

    all_metadata = []

    for idx, variant in enumerate(variants, start=1):
        label = variant["label"]
        print(f"\n{'=' * 70}")
        print(f"Processing Variant {idx}/{len(variants)}: {label}")
        print(f"{'=' * 70}")

        pipeline = prepare_pipeline_for_variant(variant)

        if args.verify_only:
            print(f"Verification-only mode: Successfully verified {label}.")
            del pipeline
            gc.collect()
            continue

        variant_dir = output_dir / label
        variant_dir.mkdir(parents=True, exist_ok=True)

        for p_idx, (prompt, seed) in enumerate(zip(EVAL_PROMPTS, EVAL_SEEDS), start=1):
            print(f"  [{p_idx}/{len(EVAL_PROMPTS)}] Seed: {seed} | \"{prompt}\"")

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

            filename = f"img_{p_idx:02d}_seed{seed}.png"
            filepath = variant_dir / filename
            image.save(filepath)

            print(f"    -> Saved: {filepath}")

            all_metadata.append({
                "model": BASE_MODEL,
                "variant": label,
                "lora_version": variant["lora_version"],
                "lora_path": str(variant["lora_dir"]),
                "lora_scale": variant["lora_scale"],
                "lora_rank": variant["lora_rank"],
                "prompt": prompt,
                "seed": seed,
                "resolution": f"{args.image_size}x{args.image_size}",
                "inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "generated_filename": str(filepath),
            })

            del image
            del result
            gc.collect()

        del pipeline
        gc.collect()

    if args.verify_only:
        print()
        print("=" * 70)
        print("ALL VARIANTS VERIFIED SUCCESSFULLY (VERIFY-ONLY)")
        print("=" * 70)
        return

    metadata_file = output_dir / "comparison_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print()
    print("=" * 70)
    print("V2 vs V4 COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Total images generated : {len(all_metadata)}")
    print(f"Output directory       : {output_dir}")
    print(f"Metadata file          : {metadata_file}")
    for variant in variants:
        variant_dir = output_dir / variant["label"]
        n = len(list(variant_dir.glob("*.png")))
        print(f"  {variant['label']}/ : {n} images")
    print("=" * 70)


if __name__ == "__main__":
    main()
