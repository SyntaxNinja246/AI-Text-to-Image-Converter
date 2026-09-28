"""
PixelDream AI - Task 6 Utilities
================================
Provides helper functions for loading metadata and existing artifacts
from Tasks 1, 2, 3, 4, and 5 without recomputing expensive assets.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from task6.config import (
        HISTORY_FILE,
        TASK3_METADATA_FILE,
        TASK4_DIR,
        TASK4_SUMMARY_FILE,
        TASK5_DIR,
        TASK5_ANALYSIS_FILE,
        TASK5_WORD_METADATA_FILE,
        CGAN_DIR,
    )
except ImportError:
    from config import (
        HISTORY_FILE,
        TASK3_METADATA_FILE,
        TASK4_DIR,
        TASK4_SUMMARY_FILE,
        TASK5_DIR,
        TASK5_ANALYSIS_FILE,
        TASK5_WORD_METADATA_FILE,
        CGAN_DIR,
    )


def load_generation_history() -> List[Dict[str, Any]]:
    """Loads generation history from generation_history.json."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Filter only existing images
        valid = []
        for item in history:
            img_path = item.get("image") or item.get("output_path") or item.get("shared_image")
            if img_path and os.path.exists(img_path):
                valid.append(item)
        return valid
    except Exception as e:
        print(f"Notice: Could not load generation history: {e}")
        return []


def get_gallery_items() -> List[str]:
    """Returns valid image paths for the Gradio gallery."""
    history = load_generation_history()
    images = []
    for item in reversed(history):
        img_path = item.get("image") or item.get("output_path")
        if img_path and os.path.exists(img_path):
            images.append(img_path)
    return images


def load_task3_info() -> Dict[str, Any]:
    """Loads Task 3 Transformer caption embedding metadata."""
    if not TASK3_METADATA_FILE.exists():
        return {
            "status": "NOT_FOUND",
            "message": "Task 3 embedding metadata not found.",
        }

    with open(TASK3_METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["status"] = "AVAILABLE"
    meta["model_architecture"] = "CLIP ViT-L/14 Text Transformer"
    meta["embedding_file"] = "dataset/analysis_results/embeddings/caption_embeddings.npy"
    meta["size_gb"] = 2.96
    return meta


def load_task4_info() -> Dict[str, Any]:
    """Loads Task 4 COCO dataset exploration summary and plot paths."""
    summary_text = ""
    if TASK4_SUMMARY_FILE.exists():
        with open(TASK4_SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary_text = f.read()

    charts = {
        "Category Distribution": str(TASK4_DIR / "category_distribution.png"),
        "Caption Length Distribution": str(TASK4_DIR / "caption_length_distribution.png"),
        "Image Width Distribution": str(TASK4_DIR / "image_width_distribution.png"),
        "Image Height Distribution": str(TASK4_DIR / "image_height_distribution.png"),
        "Sample Images with Captions": str(TASK4_DIR / "sample_images_with_captions.png"),
    }

    # Verify chart paths
    existing_charts = {k: v for k, v in charts.items() if os.path.exists(v)}

    return {
        "status": "AVAILABLE",
        "summary": summary_text.strip(),
        "total_images": 5000,
        "total_captions": 25014,
        "total_categories": 80,
        "charts": existing_charts,
    }


def load_task5_info() -> Dict[str, Any]:
    """Loads Task 5 cross-attention mechanism analysis and heatmap paths."""
    analysis_data = {}
    if TASK5_ANALYSIS_FILE.exists():
        with open(TASK5_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)

    word_meta = {}
    if TASK5_WORD_METADATA_FILE.exists():
        with open(TASK5_WORD_METADATA_FILE, "r", encoding="utf-8") as f:
            word_meta = json.load(f)

    artifacts = {
        "Full Cross-Attention Map": str(TASK5_DIR / "word_level_attention.png"),
        "Source Test Image": str(TASK5_DIR / "attention_test_image.png"),
        "Attention: 'red'": str(TASK5_DIR / "attention_red.png"),
        "Attention: 'car'": str(TASK5_DIR / "attention_car.png"),
        "Attention: 'blue'": str(TASK5_DIR / "attention_blue.png"),
        "Attention: 'house'": str(TASK5_DIR / "attention_house.png"),
        "Overall Spatial Attention": str(TASK5_DIR / "overall_attention_map.png"),
    }

    existing_artifacts = {k: v for k, v in artifacts.items() if os.path.exists(v)}

    return {
        "status": "AVAILABLE",
        "analysis": analysis_data,
        "word_metadata": word_meta,
        "artifacts": existing_artifacts,
    }


def load_cgan_eval_image() -> Optional[str]:
    """Loads the precomputed Task 2 CGAN evaluation grid if present."""
    eval_path = CGAN_DIR / "samples" / "final_cgan" / "evaluation.png"
    if eval_path.exists():
        return str(eval_path)
    return None
