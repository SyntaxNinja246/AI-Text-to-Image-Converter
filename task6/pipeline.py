"""
PixelDream AI - Task 6 Unified Pipeline
=======================================
Integrates the complete end-to-end pipeline:
1. Diffusion generation via Tiny-SD + verified V2 LoRA adapter (144 tensors).
2. Dynamic LoRA scaling without reloading.
3. Category-conditioned CGAN generation (Task 2).
4. Prompt tokenization and CLIP text inspection (Task 3).
5. Comprehensive verification and error reporting.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from task6.config import (
        PROJECT_ROOT,
        BASE_MODEL_ID,
        LORA_V2_WEIGHTS,
        DEVICE,
        DTYPE,
        CGAN_MODEL_PATH,
        CGAN_CATEGORIES,
        CGAN_NOISE_DIM,
        TASK6_OUTPUT_DIR,
        SHARED_IMAGE_DIR,
        HISTORY_FILE,
        EXPECTED_LORA_TENSOR_COUNT,
        LORA_RANK,
        LORA_ALPHA,
        LORA_TARGET_MODULES,
        DEFAULT_LORA_SCALE,
    )
except ImportError:
    from config import (
        PROJECT_ROOT,
        BASE_MODEL_ID,
        LORA_V2_WEIGHTS,
        DEVICE,
        DTYPE,
        CGAN_MODEL_PATH,
        CGAN_CATEGORIES,
        CGAN_NOISE_DIM,
        TASK6_OUTPUT_DIR,
        SHARED_IMAGE_DIR,
        HISTORY_FILE,
        EXPECTED_LORA_TENSOR_COUNT,
        LORA_RANK,
        LORA_ALPHA,
        LORA_TARGET_MODULES,
        DEFAULT_LORA_SCALE,
    )


class PixelDreamPipeline:
    """
    Unified pipeline manager for PixelDream AI Task 6.
    Maintains loaded models in memory for efficient multi-modal generation.
    """

    _instance: Optional["PixelDreamPipeline"] = None

    def __init__(self):
        self.device = DEVICE
        self.dtype = DTYPE

        # Diffusion + LoRA components
        self.pipe: Optional[StableDiffusionPipeline] = None
        self.current_lora_scale: float = DEFAULT_LORA_SCALE
        self.lora_verification_info: Dict[str, Any] = {}

        # CGAN components
        self.cgan_generator: Optional[torch.nn.Module] = None
        self.cgan_categories = CGAN_CATEGORIES

        # Ensure output directories exist
        TASK6_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SHARED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "PixelDreamPipeline":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # Task 1: Diffusion + LoRA V2 Loading & Verification
    # =========================================================================

    def load_diffusion_pipeline(self) -> StableDiffusionPipeline:
        """
        Loads the segmind/tiny-sd base pipeline, attaches PEFT LoRA, loads
        the V2 adapter weights, and strictly verifies all 144 LoRA tensors.
        Fails loudly if any tensor count or name mismatch occurs.
        """
        if self.pipe is not None:
            return self.pipe

        print("\n" + "=" * 70)
        print("PIXELDREAM AI - TASK 6: Initializing Diffusion Pipeline")
        print("=" * 70)
        print(f"Base Model    : {BASE_MODEL_ID}")
        print(f"LoRA Adapter  : {LORA_V2_WEIGHTS}")
        print(f"Device        : {self.device}")
        print(f"Precision     : {self.dtype}")

        if not LORA_V2_WEIGHTS.exists():
            raise FileNotFoundError(
                f"V2 LoRA weights not found at: {LORA_V2_WEIGHTS}\n"
                f"Please ensure Task 1 V2 training completed successfully."
            )

        # 1. Load original Tiny-SD base model
        print("Step 1/4: Loading pretrained base pipeline (Tiny-SD)...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)

        # 2. Attach PEFT LoRA structure to UNet
        print("Step 2/4: Attaching PEFT LoRA structure (Rank 4, Alpha 4)...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0.0,
            bias="none",
            target_modules=LORA_TARGET_MODULES,
            init_lora_weights=False,
        )
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)

        # 3. Map and load saved LoRA weights into PEFT structure
        print("Step 3/4: Loading and mapping V2 LoRA safetensors...")
        saved_state = load_file(str(LORA_V2_WEIGHTS))

        peft_state = {}
        for key, value in saved_state.items():
            new_key = "base_model.model." + key
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            peft_state[new_key] = value

        missing, unexpected = pipeline.unet.load_state_dict(peft_state, strict=False)

        expected_lora_keys = {
            k for k in pipeline.unet.state_dict().keys()
            if ".lora_A." in k or ".lora_B." in k
        }
        loaded_lora_keys = {
            k for k in peft_state.keys()
            if k in pipeline.unet.state_dict()
        }
        missing_lora_keys = expected_lora_keys - loaded_lora_keys

        # 4. Strict Verification
        num_saved = len(saved_state)
        num_mapped = len(peft_state)
        num_expected = len(expected_lora_keys)
        num_loaded = len(loaded_lora_keys)
        num_missing = len(missing_lora_keys)
        num_unexpected = len(unexpected)

        print("Step 4/4: Performing Strict LoRA Tensor Verification...")
        print(f"  Saved LoRA tensors : {num_saved}  (expected: {EXPECTED_LORA_TENSOR_COUNT})")
        print(f"  Mapped LoRA tensors: {num_mapped}  (expected: {EXPECTED_LORA_TENSOR_COUNT})")
        print(f"  Expected LoRA keys : {num_expected}  (expected: {EXPECTED_LORA_TENSOR_COUNT})")
        print(f"  Loaded LoRA keys   : {num_loaded}  (expected: {EXPECTED_LORA_TENSOR_COUNT})")
        print(f"  Missing LoRA keys  : {num_missing}  (expected: 0)")
        print(f"  Unexpected keys    : {num_unexpected}  (expected: 0)")

        if missing_lora_keys:
            error_msg = f"FATAL: Missing {num_missing} LoRA keys in V2 adapter loading!"
            print(error_msg)
            for k in sorted(missing_lora_keys):
                print(f"  - {k}")
            raise RuntimeError(error_msg)

        if num_unexpected != 0:
            error_msg = f"FATAL: Found {num_unexpected} unexpected keys in V2 adapter loading!"
            print(error_msg)
            for k in sorted(unexpected):
                print(f"  - {k}")
            raise RuntimeError(error_msg)

        if (
            num_saved != EXPECTED_LORA_TENSOR_COUNT
            or num_mapped != EXPECTED_LORA_TENSOR_COUNT
            or num_loaded != EXPECTED_LORA_TENSOR_COUNT
        ):
            error_msg = (
                f"FATAL: Tensor count mismatch! Expected {EXPECTED_LORA_TENSOR_COUNT}, "
                f"got saved={num_saved}, mapped={num_mapped}, loaded={num_loaded}."
            )
            print(error_msg)
            raise RuntimeError(error_msg)

        # Apply default LoRA scale (0.75)
        lora_layers = [m for m in pipeline.unet.modules() if isinstance(m, LoraLayer)]
        for layer in lora_layers:
            layer.set_scale("default", self.current_lora_scale)

        applied_scales = {layer.scaling["default"] for layer in lora_layers}
        if applied_scales != {self.current_lora_scale}:
            raise RuntimeError(f"Scale mismatch: expected {{{self.current_lora_scale}}}, got {applied_scales}")

        self.lora_verification_info = {
            "status": "VERIFIED_ACTIVE",
            "model": BASE_MODEL_ID,
            "adapter": "tiny-sd-lora-v2",
            "saved_tensors": num_saved,
            "mapped_tensors": num_mapped,
            "expected_keys": num_expected,
            "loaded_keys": num_loaded,
            "missing_keys": num_missing,
            "unexpected_keys": num_unexpected,
            "lora_layers_count": len(lora_layers),
            "current_scale": self.current_lora_scale,
        }

        print("SUCCESS: V2 LoRA adapter fully verified and active.")
        print("=" * 70 + "\n")

        self.pipe = pipeline
        return self.pipe

    def set_lora_scale(self, scale: float):
        """
        Dynamically adjusts the LoRA scale across all UNet LoRA layers
        without reloading the base pipeline or weights.
        """
        if self.pipe is None:
            self.load_diffusion_pipeline()

        scale = float(scale)
        lora_layers = [m for m in self.pipe.unet.modules() if isinstance(m, LoraLayer)]
        for layer in lora_layers:
            layer.set_scale("default", scale)

        self.current_lora_scale = scale
        self.lora_verification_info["current_scale"] = scale

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 10,
        guidance: float = 7.5,
        lora_scale: float = 0.75,
        resolution: str = "256x256",
        seed: Optional[int] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Executes diffusion inference with V2 LoRA adapter.
        Saves output image and appends generation history.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        pipeline = self.load_diffusion_pipeline()

        # Update LoRA scale if changed
        if lora_scale != self.current_lora_scale:
            self.set_lora_scale(lora_scale)

        # Parse resolution
        try:
            w_str, h_str = resolution.lower().split("x")
            width, height = int(w_str), int(h_str)
        except Exception:
            width, height = 256, 256

        # Seed handling
        if seed is None or int(seed) < 0:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        else:
            seed = int(seed)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        start_time = time.time()
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt.strip() if negative_prompt else None,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=width,
                height=height,
                generator=generator,
            )
        elapsed = time.time() - start_time

        image = result.images[0]

        # Save to task6/outputs and shared generated_images
        timestamp = int(time.time() * 1000)
        filename = f"task6_{timestamp}_seed{seed}.png"
        out_path = TASK6_OUTPUT_DIR / filename
        image.save(out_path, format="PNG")

        # Also save copy in root generated_images for global history
        shared_path = SHARED_IMAGE_DIR / filename
        try:
            image.save(shared_path, format="PNG")
        except Exception:
            pass

        # Build metadata
        meta = {
            "image": str(out_path),
            "shared_image": str(shared_path),
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            "steps": int(steps),
            "guidance": float(guidance),
            "lora_scale": float(lora_scale),
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "seed": seed,
            "model": BASE_MODEL_ID,
            "adapter": "tiny-sd-lora-v2",
            "generation_time_sec": round(elapsed, 2),
            "timestamp": timestamp,
        }

        # Save to history file
        self._record_history(meta)

        return str(out_path), elapsed, meta

    def _record_history(self, meta: Dict[str, Any]):
        """Appends generation entry to generation_history.json safely."""
        history = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(meta)

        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not write history: {e}")

    # =========================================================================
    # Task 2: Category-Conditioned CGAN Loading & Generation
    # =========================================================================

    def load_cgan(self) -> torch.nn.Module:
        """Loads the Task 2 CGAN Generator without modifying anything."""
        if self.cgan_generator is not None:
            return self.cgan_generator

        print("\nLoading Task 2 CGAN Generator...")
        if not CGAN_MODEL_PATH.exists():
            raise FileNotFoundError(f"CGAN generator model not found at: {CGAN_MODEL_PATH}")

        # Import Generator using local cgan package
        cgan_dir = str(PROJECT_ROOT / "cgan")
        if cgan_dir not in sys.path:
            sys.path.insert(0, cgan_dir)

        from generator import Generator

        gen = Generator().to(self.device)
        state_dict = torch.load(str(CGAN_MODEL_PATH), map_location=self.device)
        gen.load_state_dict(state_dict)
        gen.eval()

        self.cgan_generator = gen
        print(f"CGAN Generator ready! Supported categories: {self.cgan_categories}")
        return self.cgan_generator

    def generate_cgan_image(
        self,
        category: str,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, str, Dict[str, Any]]:
        """
        Generates a 64x64 category-conditioned image using the Task 2 CGAN.
        """
        category = category.lower().strip()
        if category not in self.cgan_categories:
            raise ValueError(f"Unknown category '{category}'. Choose from: {self.cgan_categories}")

        gen = self.load_cgan()

        cat_idx = self.cgan_categories.index(category)

        if seed is None or int(seed) < 0:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        else:
            seed = int(seed)

        rng = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(1, CGAN_NOISE_DIM, generator=rng, device=self.device)
        label = torch.tensor([cat_idx], dtype=torch.long, device=self.device)

        start = time.time()
        with torch.no_grad():
            tensor = gen(noise, label)
        elapsed = time.time() - start

        # Convert [-1, 1] tensor to [0, 255] PIL image
        img_np = tensor.squeeze(0).cpu().clamp(-1.0, 1.0).numpy()
        img_np = ((img_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0))  # C,H,W -> H,W,C
        pil_img = Image.fromarray(img_np)

        # Save image
        out_filename = f"cgan_{category}_seed{seed}.png"
        out_path = TASK6_OUTPUT_DIR / out_filename
        pil_img.save(out_path)

        meta = {
            "model": "Category-Conditioned CGAN (Task 2)",
            "category": category,
            "category_index": cat_idx,
            "seed": seed,
            "image_size": "64x64",
            "generation_time_sec": round(elapsed, 4),
            "output_path": str(out_path),
        }

        return pil_img, str(out_path), meta

    # =========================================================================
    # Task 3: Prompt Tokenization & CLIP Inspection
    # =========================================================================

    def inspect_prompt_tokens(self, prompt: str) -> Dict[str, Any]:
        """
        Tokenizes the input prompt using Tiny-SD's CLIP tokenizer,
        exposing token breakdown, IDs, and sequence padding details.
        """
        if self.pipe is None:
            self.load_diffusion_pipeline()

        tokenizer = self.pipe.tokenizer
        encoding = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids[0].tolist()
        attention_mask = encoding.attention_mask[0].tolist()

        # Decode individual tokens
        token_strings = [tokenizer.decode([tid]) for tid in input_ids]

        # Extract non-padding tokens
        actual_tokens = []
        for i, (tid, s, mask) in enumerate(zip(input_ids, token_strings, attention_mask)):
            if tid == tokenizer.eos_token_id:
                actual_tokens.append({"pos": i, "token": "<|endoftext|>", "id": tid})
                break
            elif tid == tokenizer.bos_token_id:
                actual_tokens.append({"pos": i, "token": "<|startoftext|>", "id": tid})
            else:
                actual_tokens.append({"pos": i, "token": s.strip(), "id": tid})

        return {
            "prompt": prompt,
            "model_tokenizer": "CLIPTokenizer (ViT-L/14)",
            "max_sequence_length": tokenizer.model_max_length,
            "active_token_count": len(actual_tokens),
            "padding_token_count": tokenizer.model_max_length - len(actual_tokens),
            "tokens": actual_tokens,
            "raw_input_ids_preview": input_ids[:16],
        }
