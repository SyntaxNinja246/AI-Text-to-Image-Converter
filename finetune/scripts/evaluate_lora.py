from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = ROOT / "finetune" / "models" / "tiny-sd-lora"
OUTPUT_DIR = ROOT / "finetune" / "outputs" / "lora_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

print("=" * 70)
print("PixelDream AI - Task 1 LoRA Evaluation")
print("=" * 70)
print(f"Device     : {DEVICE}")
print("Base model : segmind/tiny-sd")
print(f"LoRA file  : {LORA_DIR / 'pytorch_lora_weights.safetensors'}")
print()

print("Loading pretrained Tiny-SD...")

pipeline = StableDiffusionPipeline.from_pretrained(
    "segmind/tiny-sd",
    torch_dtype=torch.float32
)

pipeline = pipeline.to(DEVICE)

print("Base model loaded successfully.")
print()

# ---------------------------------------------------------
# Attach the SAME LoRA configuration used during training
# ---------------------------------------------------------

print("Attaching PEFT LoRA to UNet...")

lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "to_q",
        "to_k",
        "to_v",
        "to_out.0"
    ],
    init_lora_weights=False
)

pipeline.unet = get_peft_model(
    pipeline.unet,
    lora_config
)

print("LoRA structure created.")
print()

# ---------------------------------------------------------
# Load trained LoRA weights
# ---------------------------------------------------------

print("Loading trained LoRA weights...")

adapter_file = LORA_DIR / "pytorch_lora_weights.safetensors"

saved_state = load_file(str(adapter_file))

# Convert our training-time names to PEFT names
peft_state = {}

for key, value in saved_state.items():

    new_key = "base_model.model." + key

    new_key = new_key.replace(
        ".lora_A.weight",
        ".lora_A.default.weight"
    )

    new_key = new_key.replace(
        ".lora_B.weight",
        ".lora_B.default.weight"
    )

    peft_state[new_key] = value

# Load into PEFT model.
# Missing base-model weights are EXPECTED because this file
# contains only LoRA adapter weights.
missing, unexpected = pipeline.unet.load_state_dict(
    peft_state,
    strict=False
)

# Only LoRA keys should be required.
expected_lora_keys = {
    key for key in pipeline.unet.state_dict().keys()
    if ".lora_A." in key or ".lora_B." in key
}

loaded_lora_keys = {
    key for key in peft_state.keys()
    if key in pipeline.unet.state_dict()
}

missing_lora_keys = expected_lora_keys - loaded_lora_keys

print(f"Saved LoRA tensors       : {len(saved_state)}")
print(f"Mapped LoRA tensors      : {len(peft_state)}")
print(f"LoRA keys expected       : {len(expected_lora_keys)}")
print(f"LoRA keys loaded         : {len(loaded_lora_keys)}")
print(f"Base-model keys missing  : {len(missing)}")
print(f"Unexpected keys          : {len(unexpected)}")
print()

if missing_lora_keys:
    print("ERROR - Missing LoRA keys:")
    for key in sorted(missing_lora_keys):
        print(key)
    raise RuntimeError("Some LoRA weights were not loaded.")

if unexpected:
    print("ERROR - Unexpected keys:")
    for key in sorted(unexpected):
        print(key)
    raise RuntimeError("Unexpected LoRA keys found.")

if len(loaded_lora_keys) != len(peft_state):
    raise RuntimeError(
        "Not all saved LoRA tensors were matched to the PEFT model."
    )

print("SUCCESS: All trained LoRA weights loaded correctly.")
print()

# ---------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------

prompts = [
    "a motorcycle parked outdoors",
    "a cozy room with chairs",
    "a person standing in a kitchen",
    "a desk with electronic equipment",
    "a group of people outdoors"
]

print("Generating evaluation images...")
print()

metadata = []

for i, prompt in enumerate(prompts, start=1):

    print(f"[{i}/{len(prompts)}] {prompt}")

    generator = torch.Generator(device="cpu").manual_seed(100 + i)

    image = pipeline(
        prompt=prompt,
        num_inference_steps=10,
        guidance_scale=7.5,
        width=256,
        height=256,
        generator=generator
    ).images[0]

    output_file = OUTPUT_DIR / f"lora_eval_{i}.png"
    image.save(output_file)

    metadata.append({
        "index": i,
        "prompt": prompt,
        "seed": 100 + i,
        "output": str(output_file)
    })

    print(f"Saved: {output_file}")
    print()

with open(OUTPUT_DIR / "evaluation_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("=" * 70)
print("LoRA evaluation completed.")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)
