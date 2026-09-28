"""
PixelDream AI - Task 6 Configuration
====================================
Defines paths, hyperparameters, model constants, and verification thresholds
for the unified PixelDream AI end-to-end pipeline.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Task 1: Diffusion + LoRA
BASE_MODEL_ID = "segmind/tiny-sd"
LORA_V2_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v2"
LORA_V2_WEIGHTS = LORA_V2_DIR / "pytorch_lora_weights.safetensors"

# Experimental models (for reference only, NEVER default)
LORA_V3_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v3"
LORA_V4_DIR = PROJECT_ROOT / "finetune" / "models" / "tiny-sd-lora-v4"

# Task 2: CGAN
CGAN_DIR = PROJECT_ROOT / "cgan"
CGAN_MODEL_PATH = CGAN_DIR / "models" / "final_cgan" / "generator.pth"
CGAN_CATEGORIES = ["car", "chair", "bottle", "bird", "boat"]
CGAN_NOISE_DIM = 100
CGAN_IMAGE_SIZE = 64

# Task 3: Transformer & Embeddings
TASK3_DIR = PROJECT_ROOT / "dataset" / "analysis_results" / "embeddings"
TASK3_METADATA_FILE = TASK3_DIR / "embedding_metadata.json"

# Task 4: COCO Dataset Analysis
TASK4_DIR = PROJECT_ROOT / "dataset" / "analysis_results"
TASK4_SUMMARY_FILE = TASK4_DIR / "dataset_summary.txt"

# Task 5: Attention Mechanism
TASK5_DIR = PROJECT_ROOT / "attention" / "outputs"
TASK5_ANALYSIS_FILE = TASK5_DIR / "attention_analysis.json"
TASK5_WORD_METADATA_FILE = TASK5_DIR / "word_attention_metadata.json"

# Task 6 Outputs and History
TASK6_OUTPUT_DIR = PROJECT_ROOT / "task6" / "outputs"
SHARED_IMAGE_DIR = PROJECT_ROOT / "generated_images"
HISTORY_FILE = PROJECT_ROOT / "generation_history.json"

# ---------------------------------------------------------------------------
# Device & Computation Settings
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
DTYPE = torch.float32

# ---------------------------------------------------------------------------
# Diffusion Default Settings
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = "a cozy room with chairs"
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, cartoon, bad anatomy"
DEFAULT_STEPS = 10
DEFAULT_GUIDANCE = 7.5
DEFAULT_LORA_SCALE = 0.75  # Optimal scale determined during V2 vs V4 evaluation
DEFAULT_SEED = 102
DEFAULT_RESOLUTION = "256x256"

# ---------------------------------------------------------------------------
# LoRA Verification Constraints
# ---------------------------------------------------------------------------
EXPECTED_LORA_TENSOR_COUNT = 144
LORA_RANK = 4
LORA_ALPHA = 4
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
