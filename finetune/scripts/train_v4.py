"""
PixelDream AI - Task 1 LoRA Fine-Tuning v4
============================================

Trains a LoRA adapter on top of the original pretrained segmind/tiny-sd base model
using the improved, high-quality, category-diverse COCO v4 dataset (1,500 samples).

V4 Architecture:
  - Proven V2-style Rank-4, Alpha-4 LoRA architecture (NOT V3).
  - Target modules: to_q, to_k, to_v, to_out.0
  - Base model: segmind/tiny-sd (original pretrained weights, frozen VAE and text encoder).
  - Hyperparameters:
      Epochs                   : 3
      Batch size               : 1
      Gradient accumulation    : 4
      Learning rate            : 5e-6
      Resolution               : 256x256
      Max gradient clipping    : 1.0
      Device                   : CPU
      Precision                : float32
      Optimizer                : AdamW

Outputs:
  - Adapter weights : finetune/models/tiny-sd-lora-v4/
  - Artifacts       : finetune/outputs/v4/ (training_log.json, training_config.json)
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from peft import LoraConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# Default Paths and Hyperparameters
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "segmind/tiny-sd"
DEFAULT_METADATA_FILE = PROJECT_ROOT / "finetune" / "data" / "v4" / "metadata.jsonl"
DEFAULT_MODEL_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "outputs" / "v4"

DEVICE = torch.device("cpu")
DTYPE = torch.float32

# Hyperparameters for v4
IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_EPOCHS = 3
LEARNING_RATE = 5e-6
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0
SEED = 42

# LoRA Configuration (V2-proven architecture)
LORA_RANK = 4
LORA_ALPHA = 4
LORA_DROPOUT = 0.0

LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]


class CocoFineTuneDataset(Dataset):
    """COCO Image-Caption Dataset for V4 Fine-Tuning."""

    def __init__(self, metadata_file: Path, tokenizer, image_size: int = IMAGE_SIZE):
        self.metadata_file = Path(metadata_file)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.records = []

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ])

        print(f"Dataset loaded: {len(self.records)} records from {self.metadata_file}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]

        image_path = PROJECT_ROOT / record["file_name"]
        caption = record["caption"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens.input_ids.squeeze(0)

        return {
            "pixel_values": image,
            "input_ids": input_ids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PixelDream AI - Task 1 LoRA Fine-Tuning v4"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DEFAULT_METADATA_FILE),
        help=f"Path to metadata.jsonl (default: {DEFAULT_METADATA_FILE})",
    )
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT_DIR),
        help=f"Directory to save LoRA adapter (default: {DEFAULT_MODEL_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save logs and configs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help=f"Image resolution (default: {IMAGE_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum training steps for testing / dry run",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Flag indicating a dry-run test mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    metadata_file = Path(args.dataset_path)
    model_output_dir = Path(args.model_output_dir)
    output_dir = Path(args.output_dir)

    # Dry-run: enforce 1-step limit so it terminates quickly
    if args.dry_run and args.max_steps is None:
        args.max_steps = 1

    torch.manual_seed(SEED)

    print("=" * 70)
    print("PixelDream AI - Task 1 LoRA Fine-Tuning v4")
    print("=" * 70)
    print(f"Project root            : {PROJECT_ROOT}")
    print(f"Base model              : {BASE_MODEL} (original pretrained weights)")
    print(f"Device                  : {DEVICE}")
    print(f"Dtype                   : {DTYPE}")
    print(f"Dataset path            : {metadata_file}")
    print(f"Resolution              : {args.image_size}x{args.image_size}")
    print(f"Learning rate           : {args.lr}")
    print(f"Epochs                  : {args.epochs}")
    print(f"Batch size              : {args.batch_size}")
    print(f"Gradient accumulation   : {args.gradient_accumulation_steps}")
    print(f"Adapter output dir      : {model_output_dir}")
    print(f"Outputs dir             : {output_dir}")
    if args.max_steps is not None:
        print(f"Max steps (test limit)  : {args.max_steps}")
    if args.dry_run:
        print("Mode                    : DRY RUN (test execution)")
    print("=" * 70)
    print()

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Original Pretrained Tiny-SD
    print("Step 1: Loading original pretrained Tiny-SD base model...")
    pipeline = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )
    pipeline = pipeline.to(DEVICE)

    pipeline.unet.float()
    pipeline.vae.float()
    pipeline.text_encoder.float()

    print("Original pretrained Tiny-SD loaded successfully.\n")

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = pipeline.scheduler

    # Freeze VAE and Text Encoder
    print("Step 2: Freezing VAE and Text Encoder...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    # Freeze UNet base weights
    print("Step 3: Attaching V2-style Rank-4 LoRA adapter to UNet...")
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    unet.add_adapter(lora_config)

    # Identify trainable parameters
    trainable_parameters = []
    for name, parameter in unet.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True
            trainable_parameters.append(parameter)
        else:
            parameter.requires_grad = False

    total_parameters = sum(p.numel() for p in unet.parameters())
    trainable_parameters_count = sum(p.numel() for p in trainable_parameters)
    trainable_percentage = (trainable_parameters_count / total_parameters) * 100

    print()
    print("=" * 70)
    print("LoRA Parameter Verification (v4)")
    print("=" * 70)
    print(f"Total UNet parameters     : {total_parameters:,}")
    print(f"Trainable LoRA parameters : {trainable_parameters_count:,} (expected: 431,616)")
    print(f"Trainable percentage      : {trainable_percentage:.4f}%")
    print(f"LoRA Rank (r)             : {LORA_RANK}")
    print(f"LoRA Alpha (alpha)        : {LORA_ALPHA}")
    print(f"LoRA Dropout              : {LORA_DROPOUT}")
    print("Target modules:")
    for module in LORA_TARGET_MODULES:
        print(f"  - {module}")
    print("=" * 70)
    print()

    assert trainable_parameters_count == 431616, (
        f"Expected exactly 431,616 trainable parameters, got {trainable_parameters_count}"
    )

    # Load Dataset
    print("Step 4: Initializing Dataset and DataLoader...")
    dataset = CocoFineTuneDataset(
        metadata_file,
        tokenizer,
        image_size=args.image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    expected_forward_steps = len(dataset) * args.epochs
    expected_opt_steps_per_epoch = len(dataset) // args.gradient_accumulation_steps
    if len(dataset) % args.gradient_accumulation_steps != 0:
        expected_opt_steps_per_epoch += 1
    total_expected_opt_steps = expected_opt_steps_per_epoch * args.epochs

    print(f"Dataset samples         : {len(dataset)}")
    print(f"Total epochs            : {args.epochs}")
    print(f"Batch size              : {args.batch_size}")
    print(f"Gradient accumulation   : {args.gradient_accumulation_steps}")
    print(f"Expected forward passes : {expected_forward_steps:,}")
    print(f"Expected optimizer steps: {total_expected_opt_steps:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
    )
    optimizer.zero_grad()

    print("=" * 70)
    print(f"Starting training v4 (Mode: {'DRY RUN' if args.dry_run else 'FULL'})")
    print("=" * 70)
    print()

    global_step = 0
    training_log = []
    stop_early = False
    step_times = []

    train_start_time = time.time()

    for epoch in range(args.epochs):
        if stop_early:
            break

        unet.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            step_start = time.time()

            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float32)
            input_ids = batch["input_ids"].to(DEVICE)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            encoder_hidden_states = encoder_hidden_states.float()
            noise = torch.randn_like(latents)
            batch_sz = latents.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_sz,),
                device=DEVICE,
                dtype=torch.long,
            )

            noisy_latents = noise_scheduler.add_noise(
                latents,
                noise,
                timesteps,
            )

            noise_prediction = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = F.mse_loss(
                noise_prediction.float(),
                noise.float(),
            )

            loss_for_backward = loss / args.gradient_accumulation_steps
            loss_for_backward.backward()

            is_opt_step = ((step + 1) % args.gradient_accumulation_steps == 0)
            if is_opt_step or (args.max_steps is not None and (step + 1) >= args.max_steps):
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)

            actual_loss = loss.item()
            epoch_loss += actual_loss

            if (
                step == 0
                or (step + 1) % 25 == 0
                or (step + 1) == len(dataloader)
                or (args.max_steps is not None and (step + 1) >= args.max_steps)
            ):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step + 1}/{len(dataloader)} | "
                    f"Global Step: {global_step} | "
                    f"Loss: {actual_loss:.6f} | "
                    f"Step time: {step_elapsed:.2f}s"
                )

                training_log.append({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "global_step": global_step,
                    "loss": actual_loss,
                    "step_time_sec": round(step_elapsed, 4),
                })

            del pixel_values
            del input_ids
            del latents
            del encoder_hidden_states
            del noise
            del timesteps
            del noisy_latents
            del noise_prediction
            del loss
            del loss_for_backward
            gc.collect()

            if args.max_steps is not None and (step + 1) >= args.max_steps:
                print(f"\nReached max_steps ({args.max_steps}). Stopping training early.")
                stop_early = True
                break

        # Handle remaining gradients at epoch end
        if not stop_early and (len(dataloader) % args.gradient_accumulation_steps != 0):
            torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        steps_counted = args.max_steps if stop_early else len(dataloader)
        average_loss = epoch_loss / max(steps_counted, 1)

        print()
        print(f"Epoch {epoch + 1} summary | Average loss: {average_loss:.6f}")
        print()

    total_time = time.time() - train_start_time

    # Save LoRA adapter
    print("=" * 70)
    print("Saving LoRA v4 Adapter")
    print("=" * 70)

    unet.save_lora_adapter(
        model_output_dir,
        adapter_name="default",
    )
    print(f"LoRA v4 adapter saved successfully to:\n  {model_output_dir}")

    # Save training log
    training_log_file = output_dir / "training_log.json"
    with open(training_log_file, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    avg_step_sec = statistics.mean(step_times) if step_times else 0.0

    # Save training config
    training_config = {
        "base_model": BASE_MODEL,
        "dataset_path": str(metadata_file),
        "dataset_size": len(dataset),
        "resolution": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "trainable_parameter_count": trainable_parameters_count,
        "total_parameter_count": total_parameters,
        "trainable_percentage": round(trainable_percentage, 6),
        "optimizer": "AdamW",
        "random_seed": SEED,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "global_steps": global_step,
        "total_elapsed_seconds": round(total_time, 2),
        "avg_step_seconds": round(avg_step_sec, 4),
        "mode": "dry_run" if args.dry_run else "full",
    }

    config_file = output_dir / "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    del pipeline
    gc.collect()

    print()
    print("=" * 70)
    print("TRAINING RUN COMPLETE")
    print("=" * 70)
    print(f"LoRA model directory : {model_output_dir}")
    print(f"Training log file    : {training_log_file}")
    print(f"Training config file : {config_file}")
    print(f"Global steps executed: {global_step}")
    print(f"Total time elapsed   : {total_time:.2f}s")
    if avg_step_sec > 0:
        print(f"Average step time    : {avg_step_sec:.3f}s / forward pass")
    print("=" * 70)


if __name__ == "__main__":
    main()
