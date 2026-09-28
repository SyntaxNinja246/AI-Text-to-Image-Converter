import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from diffusers import DiffusionPipeline
from peft import LoraConfig


# ============================================================
# PixelDream AI - Task 1 LoRA Fine-Tuning v2 Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "segmind/tiny-sd"

METADATA_FILE = PROJECT_ROOT / "finetune" / "data" / "metadata.jsonl"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v2"
OUTPUT_DIR = PROJECT_ROOT / "finetune" / "outputs" / "v2"

DEVICE = torch.device("cpu")

# Hyperparameters for v2
IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0
SEED = 42

# LoRA Configuration
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
    """COCO Image-Caption Dataset for Fine-Tuning."""

    def __init__(self, metadata_file, tokenizer, image_size=IMAGE_SIZE):
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

        print(f"Dataset records: {len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = PROJECT_ROOT / record["file_name"]
        caption = record["caption"]

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


def main():
    parser = argparse.ArgumentParser(
        description="PixelDream AI - Task 1 LoRA Fine-Tuning v2"
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
        "--model_output_dir",
        type=str,
        default=str(MODEL_OUTPUT_DIR),
        help=f"Directory to save LoRA adapter (default: {MODEL_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Directory to save logs and configs (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum training steps for testing/smoke runs",
    )
    args = parser.parse_args()

    model_output_dir = Path(args.model_output_dir)
    output_dir = Path(args.output_dir)

    torch.manual_seed(SEED)

    print("=" * 70)
    print("PixelDream AI - Task 1 LoRA Fine-Tuning v2")
    print("=" * 70)
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Base model         : {BASE_MODEL}")
    print(f"Device             : {DEVICE}")
    print(f"Metadata           : {METADATA_FILE}")
    print(f"Resolution         : {args.image_size}x{args.image_size}")
    print(f"Learning rate      : {args.lr}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Gradient accum.    : {args.gradient_accumulation_steps}")
    print(f"Adapter output     : {model_output_dir}")
    print(f"Outputs dir        : {output_dir}")
    if args.max_steps is not None:
        print(f"Max steps (smoke)  : {args.max_steps}")
    print()

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found:\n{METADATA_FILE}")

    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained Tiny-SD...")

    pipeline = DiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )

    pipeline = pipeline.to(DEVICE)

    pipeline.unet.float()
    pipeline.vae.float()
    pipeline.text_encoder.float()

    print("Pretrained model loaded successfully.")

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = pipeline.scheduler

    print()
    print("Model components:")
    print(f"  UNet input channels : {unet.config.in_channels}")
    print(f"  UNet output channels: {unet.config.out_channels}")
    print(f"  VAE scale factor    : {vae.config.scaling_factor}")
    print(f"  Text hidden size    : {text_encoder.config.hidden_size}")
    print(f"  Tokenizer max length: {tokenizer.model_max_length}")
    print()

    print("Freezing VAE and text encoder...")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.eval()
    text_encoder.eval()

    print("Preparing UNet for LoRA...")

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

    trainable_parameters = []

    for name, parameter in unet.named_parameters():
        if "lora_" in name:
            parameter.requires_grad = True
            trainable_parameters.append(parameter)
        else:
            parameter.requires_grad = False

    total_parameters = sum(parameter.numel() for parameter in unet.parameters())
    trainable_parameters_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    trainable_percentage = (trainable_parameters_count / total_parameters) * 100

    print()
    print("=" * 70)
    print("LoRA Configuration")
    print("=" * 70)
    print(f"Total UNet parameters     : {total_parameters:,}")
    print(f"Trainable LoRA parameters : {trainable_parameters_count:,}")
    print(f"Trainable percentage      : {trainable_percentage:.4f}%")
    print(f"LoRA Rank (r)             : {LORA_RANK}")
    print(f"LoRA Alpha                : {LORA_ALPHA}")
    print(f"LoRA Dropout              : {LORA_DROPOUT}")
    print()
    print("Target modules:")
    for module in LORA_TARGET_MODULES:
        print(f"  - {module}")
    print()

    dataset = CocoFineTuneDataset(
        METADATA_FILE,
        tokenizer,
        image_size=args.image_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Training images: {len(dataset)}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Epochs         : {args.epochs}")
    print(f"Gradient accum.: {args.gradient_accumulation_steps}")
    print()

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
    )

    optimizer.zero_grad()

    print("=" * 70)
    print("Starting training v2")
    print("=" * 70)
    print()

    global_step = 0
    training_log = []
    stop_early = False

    for epoch in range(args.epochs):
        if stop_early:
            break

        unet.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
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

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            actual_loss = loss.item()
            epoch_loss += actual_loss

            if (
                step == 0
                or (step + 1) % 25 == 0
                or (step + 1) == len(dataloader)
            ):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {step + 1}/{len(dataloader)} | "
                    f"Global Step: {global_step} | "
                    f"Loss: {actual_loss:.6f}"
                )

                training_log.append({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "global_step": global_step,
                    "loss": actual_loss,
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
                print(f"Reached max_steps ({args.max_steps}). Stopping epoch.")
                stop_early = True
                break

        # Handle any remaining gradients from gradient accumulation at end of epoch
        if not stop_early and (len(dataloader) % args.gradient_accumulation_steps != 0):
            torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        steps_counted = args.max_steps if stop_early else len(dataloader)
        average_loss = epoch_loss / max(steps_counted, 1)

        print()
        print(f"Epoch {epoch + 1} complete")
        print(f"Average loss: {average_loss:.6f}")
        print()

    print("=" * 70)
    print("Saving LoRA adapter v2")
    print("=" * 70)

    unet.save_lora_adapter(
        model_output_dir,
        adapter_name="default",
    )

    print(f"LoRA v2 adapter saved to:")
    print(model_output_dir)

    training_log_file = output_dir / "training_log.json"
    with open(training_log_file, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    training_config = {
        "base_model": BASE_MODEL,
        "device": str(DEVICE),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "training_images": len(dataset),
        "total_unet_parameters": total_parameters,
        "trainable_lora_parameters": trainable_parameters_count,
        "trainable_percentage": trainable_percentage,
        "global_steps": global_step,
    }

    config_file = output_dir / "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    del pipeline
    gc.collect()

    print()
    print("=" * 70)
    print("TRAINING v2 COMPLETE")
    print("=" * 70)
    print(f"LoRA model   : {model_output_dir}")
    print(f"Training log : {training_log_file}")
    print(f"Config       : {config_file}")
    print()


if __name__ == "__main__":
    main()
