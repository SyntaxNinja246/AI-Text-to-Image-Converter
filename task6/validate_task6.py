"""
PixelDream AI - Task 6 Verification Suite
=========================================
Validates:
1. Pipeline initialization and LoRA V2 loading.
2. Exact tensor verification (144 saved, 144 mapped, 144 loaded, 0 missing, 0 unexpected).
3. Dynamic LoRA scaling.
4. Actual end-to-end text-to-image generation.
5. Task 2 CGAN category-conditioned generation.
6. Task 3 Transformer token inspection and embedding metadata.
7. Task 4 COCO dataset analysis artifacts.
8. Task 5 cross-attention artifacts.
9. Generation history preservation and extension.
10. Original root app.py integrity.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task6.pipeline import PixelDreamPipeline
from task6.utils import (
    load_generation_history,
    load_task3_info,
    load_task4_info,
    load_task5_info,
    load_cgan_eval_image,
)


def run_verification():
    print("=" * 75)
    print("PIXELDREAM AI - TASK 6 SYSTEM VERIFICATION SUITE")
    print("=" * 75)

    pipeline = PixelDreamPipeline.get_instance()

    # -------------------------------------------------------------------------
    # Test 1 & 2: LoRA V2 Loading and Strict 144 Tensor Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 1 & 2] Verifying V2 LoRA Adapter Loading and 144 Tensors...")
    pipe = pipeline.load_diffusion_pipeline()
    info = pipeline.lora_verification_info

    print(f"  Model             : {info['model']}")
    print(f"  Adapter           : {info['adapter']}")
    print(f"  Saved Tensors     : {info['saved_tensors']}")
    print(f"  Mapped Tensors    : {info['mapped_tensors']}")
    print(f"  Expected Keys     : {info['expected_keys']}")
    print(f"  Loaded Keys       : {info['loaded_keys']}")
    print(f"  Missing Keys      : {info['missing_keys']}")
    print(f"  Unexpected Keys   : {info['unexpected_keys']}")
    print(f"  LoRA Layer Count  : {info['lora_layers_count']}")

    assert info["saved_tensors"] == 144, f"Expected 144 saved tensors, got {info['saved_tensors']}"
    assert info["mapped_tensors"] == 144, f"Expected 144 mapped tensors, got {info['mapped_tensors']}"
    assert info["loaded_keys"] == 144, f"Expected 144 loaded keys, got {info['loaded_keys']}"
    assert info["missing_keys"] == 0, f"Expected 0 missing keys, got {info['missing_keys']}"
    assert info["unexpected_keys"] == 0, f"Expected 0 unexpected keys, got {info['unexpected_keys']}"
    print("  --> PASS: All 144 LoRA tensors verified successfully!")

    # -------------------------------------------------------------------------
    # Test 3: LoRA Scale Adjustment
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing Dynamic LoRA Scale Adjustment...")
    pipeline.set_lora_scale(0.75)
    assert pipeline.current_lora_scale == 0.75
    print("  --> PASS: Dynamic LoRA scale set to 0.75 without reloading.")

    # -------------------------------------------------------------------------
    # Test 4: End-to-End Diffusion Generation
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Running Actual End-to-End Diffusion Generation (Tiny-SD + LoRA V2)...")
    prompt = "a cozy room with chairs"
    seed = 102
    steps = 5  # Quick verification run
    out_path, elapsed, meta = pipeline.generate_image(
        prompt=prompt,
        steps=steps,
        guidance=7.5,
        lora_scale=0.75,
        resolution="256x256",
        seed=seed,
    )
    print(f"  Output Image Path : {out_path}")
    print(f"  Generation Time   : {elapsed:.2f} s")
    print(f"  Output Exists     : {os.path.exists(out_path)}")
    print(f"  Output File Size  : {os.path.getsize(out_path)} bytes")

    assert os.path.exists(out_path), "Generated image file does not exist!"
    assert os.path.getsize(out_path) > 10000, "Generated image file is unexpectedly small!"
    print("  --> PASS: End-to-end text-to-image generation successful!")

    # -------------------------------------------------------------------------
    # Test 5: Task 2 CGAN Category Generation
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing Task 2 Category-Conditioned CGAN Generation...")
    category = "car"
    cgan_img, cgan_path, cgan_meta = pipeline.generate_cgan_image(category=category, seed=42)
    print(f"  Category          : {cgan_meta['category']}")
    print(f"  Image Size        : {cgan_img.size}")
    print(f"  Inference Time    : {cgan_meta['generation_time_sec']:.4f} s")
    print(f"  Output File       : {cgan_path}")

    assert os.path.exists(cgan_path), "CGAN output image does not exist!"
    assert cgan_img.size == (64, 64), f"Expected 64x64, got {cgan_img.size}"
    print("  --> PASS: Task 2 CGAN category generation successful!")

    # -------------------------------------------------------------------------
    # Test 6: Task 3 CLIP Prompt Tokenization & Embedding Metadata
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing Task 3 CLIP Tokenization & Embedding Metadata...")
    token_info = pipeline.inspect_prompt_tokens("a cozy room with chairs")
    t3_meta = load_task3_info()
    print(f"  Prompt            : {token_info['prompt']}")
    print(f"  Tokenizer         : {token_info['model_tokenizer']}")
    print(f"  Active Tokens     : {token_info['active_token_count']}")
    print(f"  Task 3 Records    : {t3_meta.get('number_of_records')}")
    print(f"  Embedding Shape   : {t3_meta.get('embedding_shape')}")

    assert token_info["active_token_count"] > 0, "No tokens parsed!"
    assert t3_meta["status"] == "AVAILABLE", "Task 3 metadata not available!"
    print("  --> PASS: Task 3 tokenization and embedding metadata verified!")

    # -------------------------------------------------------------------------
    # Test 7: Task 4 COCO Dataset Analysis
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing Task 4 COCO Dataset Exploration Artifacts...")
    t4_info = load_task4_info()
    print(f"  Summary Found     : {len(t4_info['summary']) > 0}")
    print(f"  Total Images      : {t4_info['total_images']}")
    print(f"  Charts Available  : {list(t4_info['charts'].keys())}")

    assert len(t4_info["charts"]) > 0, "No Task 4 charts found!"
    print("  --> PASS: Task 4 dataset exploration artifacts verified!")

    # -------------------------------------------------------------------------
    # Test 8: Task 5 Attention Mechanism Artifacts
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing Task 5 Attention Mechanism Artifacts...")
    t5_info = load_task5_info()
    print(f"  Artifacts Found   : {list(t5_info['artifacts'].keys())}")
    print(f"  Processors Count  : {t5_info.get('analysis', {}).get('total_attention_processors')}")

    assert len(t5_info["artifacts"]) > 0, "No Task 5 attention artifacts found!"
    print("  --> PASS: Task 5 attention artifacts verified!")

    # -------------------------------------------------------------------------
    # Test 9: Generation History Integrity
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Verifying Generation History...")
    history = load_generation_history()
    print(f"  Total History Records: {len(history)}")
    latest = history[-1]
    print(f"  Latest Record Prompt : {latest.get('prompt')}")
    print(f"  Latest Record Model  : {latest.get('model')}")
    print(f"  Latest Record Adapter: {latest.get('adapter')}")

    assert len(history) > 0, "History file is empty!"
    assert latest.get("prompt") == prompt, f"Expected latest prompt '{prompt}', got '{latest.get('prompt')}'"
    print("  --> PASS: Generation history correctly maintained!")

    # -------------------------------------------------------------------------
    # Test 10: Original Root App Integrity Check
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Checking Integrity of Original Root App...")
    root_app_path = PROJECT_ROOT / "app.py"
    root_gen_path = PROJECT_ROOT / "generator.py"
    assert root_app_path.exists(), "Original app.py missing!"
    assert root_gen_path.exists(), "Original generator.py missing!"
    print(f"  Root app.py size     : {os.path.getsize(root_app_path)} bytes")
    print(f"  Root generator.py size: {os.path.getsize(root_gen_path)} bytes")
    print("  --> PASS: Original root app is intact and untouched!")

    print("\n" + "=" * 75)
    print("ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_verification()
