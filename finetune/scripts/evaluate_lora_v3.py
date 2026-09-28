import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA_DIR = ROOT / "finetune" / "models" / "tiny-sd-lora-v3"
DEFAULT_OUTPUT_DIR = ROOT / "finetune" / "outputs" / "lora_evaluation_v3"
DEVICE = torch.device("cpu")

BASE_MODEL = "segmind/tiny-sd"

EVAL_PROMPTS = [
    "a motorcycle parked outdoors",
    "a cozy room with chairs",
    "a person standing in a kitchen",
    "a desk with electronic equipment",
    "a group of people outdoors",
]


def main():
    parser = argparse.ArgumentParser(
        description="PixelDream AI - Task 1 LoRA Evaluation v3"
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
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha (default: 8)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
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
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_file = lora_dir / "pytorch_lora_weights.safetensors"

    print("=" * 70)
    print("PixelDream AI - Task 1 LoRA Evaluation v3")
    print("=" * 70)
    print(f"Device         : {DEVICE}")
    print(f"Base model     : {BASE_MODEL}")
    print(f"LoRA file      : {adapter_file}")
    print(f"LoRA Rank (r)  : {args.lora_rank}")
    print(f"LoRA Alpha     : {args.lora_alpha}")
    print(f"LoRA Dropout   : {args.lora_dropout}")
    print(f"Output dir     : {output_dir}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale : {args.guidance_scale}")
    print(f"Resolution     : {args.image_size}x{args.image_size}")
    print()

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
    )
    pipeline = pipeline.to(DEVICE)
    print("Base model loaded successfully.\n")

    # 2. Attach the same PEFT LoRA structure with V3 configuration (rank=8, alpha=8)
    print("Step 2: Attaching PEFT LoRA structure to UNet (Rank 8, Alpha 8)...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
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

    # 4. Report saved, mapped, expected, loaded, missing, and unexpected LoRA tensors
    num_saved = len(saved_state)
    num_mapped = len(peft_state)
    num_expected = len(expected_lora_keys)
    num_loaded = len(loaded_lora_keys)
    num_missing_lora = len(missing_lora_keys)
    num_unexpected = len(unexpected)

    print()
    print("=" * 70)
    print("Step 4: LoRA Tensor Verification & Reporting (v3)")
    print("=" * 70)
    print(f"Saved LoRA tensors     : {num_saved}  (expected: 144)")
    print(f"Mapped LoRA tensors    : {num_mapped}  (expected: 144)")
    print(f"Expected LoRA keys     : {num_expected}  (expected: 144)")
    print(f"Loaded LoRA keys       : {num_loaded}  (expected: 144)")
    print(f"Missing LoRA keys      : {num_missing_lora}  (expected: 0)")
    print(f"Unexpected LoRA keys   : {num_unexpected}  (expected: 0)")
    print(f"Base-model keys unmod. : {len(missing)} (normal - LoRA adapter only)")
    print("=" * 70)

    if missing_lora_keys:
        print("ERROR - Missing LoRA keys:")
        for key in sorted(missing_lora_keys):
            print(f"  - {key}")
        raise RuntimeError(f"Missing {len(missing_lora_keys)} LoRA weights.")

    if num_unexpected != 0:
        print("ERROR - Unexpected keys:")
        for key in sorted(unexpected):
            print(f"  - {key}")
        raise RuntimeError(f"Found {num_unexpected} unexpected keys.")

    assert num_saved == 144, f"Verification failed: Saved LoRA tensors = {num_saved}, expected 144"
    assert num_mapped == 144, f"Verification failed: Mapped LoRA tensors = {num_mapped}, expected 144"
    assert num_expected == 144, f"Verification failed: Expected LoRA keys = {num_expected}, expected 144"
    assert num_loaded == 144, f"Verification failed: LoRA keys loaded = {num_loaded}, expected 144"
    assert num_missing_lora == 0, f"Verification failed: Missing LoRA keys = {num_missing_lora}, expected 0"
    assert num_unexpected == 0, f"Verification failed: Unexpected LoRA keys = {num_unexpected}, expected 0"

    print("VERIFICATION PASSED: All 144 LoRA v3 tensors verified and mapped without error.\n")

    if args.verify_only:
        print("Verification-only mode completed successfully.")
        return

    # 5. Generate 5 evaluation images using fixed seeds
    print("=" * 70)
    print("Step 5: Generating 5 Evaluation Images (v3)")
    print("=" * 70)

    metadata = []

    for i, prompt in enumerate(EVAL_PROMPTS, start=1):
        seed = 100 + i
        print(f"[{i}/{len(EVAL_PROMPTS)}] Prompt: \"{prompt}\" | Seed: {seed}")

        generator = torch.Generator(device="cpu").manual_seed(seed)

        image = pipeline(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.image_size,
            height=args.image_size,
            generator=generator,
        ).images[0]

        # 6. Save images to finetune/outputs/lora_evaluation_v3/
        output_file = output_dir / f"lora_eval_{i}.png"
        image.save(output_file)

        metadata.append({
            "index": i,
            "prompt": prompt,
            "seed": seed,
            "output": str(output_file),
            "resolution": f"{args.image_size}x{args.image_size}",
            "inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        })

        print(f"  -> Saved: {output_file}\n")

    metadata_file = output_dir / "evaluation_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("LoRA v3 Evaluation Completed Successfully.")
    print(f"Output directory : {output_dir}")
    print(f"Metadata file    : {metadata_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
