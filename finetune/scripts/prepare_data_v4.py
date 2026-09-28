"""
PixelDream AI - Task 1 V4 Dataset Preparation Script
=====================================================

Prepares a high-quality, diverse dataset of exactly 1,500 unique image-caption
pairs from the local COCO dataset (val2017).

Core Principles:
  - Exactly 1,500 UNIQUE image-caption pairs (one caption per unique image).
  - Every selected image must exist locally.
  - No empty captions, no very short captions.
  - Concrete visual descriptions preferred (colors, spatial relationships,
    actions, concrete scenes, and objects).
  - COCO original caption text preserved without rewriting or synthetic text.
  - Broad diversity across all 12 COCO supercategories and 80 categories.
  - Fully deterministic with fixed seed (SEED = 42).

Outputs:
  - finetune/data/v4/metadata.jsonl
  - finetune/data/v4/dataset_report.json
"""

import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# Paths and Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "dataset" / "images" / "val2017"
CAPTIONS_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "annotations"
    / "annotations"
    / "captions_val2017.json"
)
INSTANCES_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "annotations"
    / "annotations"
    / "instances_val2017.json"
)

OUTPUT_DIR = PROJECT_ROOT / "finetune" / "data" / "v4"
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"
REPORT_FILE = OUTPUT_DIR / "dataset_report.json"

TARGET_RECORDS = 1500
SEED = 42

# Min length constraints
MIN_WORD_COUNT = 6
MIN_CHAR_COUNT = 25

# Lexicon for concrete visual scoring
COLOR_TERMS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
    "purple", "pink", "gray", "grey", "silver", "gold", "dark", "light",
    "tan", "burgundy", "blonde", "golden", "beige", "navy", "maroon"
}

RELATIONSHIP_PREPOSITIONS = {
    "in front of", "next to", "on top of", "sitting on", "standing on",
    "walking on", "behind", "between", "beside", "holding", "inside",
    "outside", "near", "under", "along", "across", "through", "with",
    "against", "surrounded by", "perched on", "mounted on", "parked on"
}

ACTION_VERBS = {
    "sitting", "standing", "walking", "running", "riding", "playing",
    "holding", "eating", "looking", "driving", "flying", "jumping",
    "smiling", "cooking", "reading", "sleeping", "resting", "talking",
    "wearing", "grazing", "swimming", "jumping", "cutting", "serving",
    "watching", "carrying", "leaning", "skating", "skiing", "surfing"
}

SCENE_TERMS = {
    "kitchen", "living room", "bedroom", "bathroom", "street", "road",
    "highway", "sidewalk", "park", "beach", "ocean", "forest", "field",
    "office", "restaurant", "market", "counter", "table", "desk",
    "water", "snow", "grass", "trees", "sky", "building", "driveway",
    "garage", "yard", "harbor", "airport", "runway", "dock", "slope"
}

VAGUE_PHRASES = [
    "looks like", "something", "an image of", "a photo of", "a picture of",
    "photograph of", "close up of", "close-up of", "stuff", "things",
    "very cluttered", "seems to be", "blurry", "unknown"
]


def score_caption(caption: str, category_names: set[str]) -> float:
    """
    Computes a concrete visual quality score for a caption.
    Higher score means more descriptive, visually grounded content.
    """
    text_lower = caption.lower().strip()
    words = text_lower.split()
    word_count = len(words)

    # Base score
    score = 0.0

    # Length preference: ideal range 8-22 words
    if 8 <= word_count <= 22:
        score += 3.0
    elif 6 <= word_count < 8:
        score += 1.5
    elif 23 <= word_count <= 30:
        score += 1.5
    else:
        score += 0.0

    # Color richness
    for color in COLOR_TERMS:
        if re.search(r"\b" + re.escape(color) + r"\b", text_lower):
            score += 1.5

    # Spatial relationships
    for phrase in RELATIONSHIP_PREPOSITIONS:
        if phrase in text_lower:
            score += 2.0

    # Action verbs
    for verb in ACTION_VERBS:
        if re.search(r"\b" + re.escape(verb) + r"\b", text_lower):
            score += 1.5

    # Concrete scenes
    for scene in SCENE_TERMS:
        if re.search(r"\b" + re.escape(scene) + r"\b", text_lower):
            score += 1.5

    # Grounded category objects
    for cat in category_names:
        if re.search(r"\b" + re.escape(cat) + r"\b", text_lower):
            score += 1.0

    # Penalties for meta/vague phrasing
    for vague in VAGUE_PHRASES:
        if vague in text_lower:
            score -= 3.0

    # Bonus for standard capitalization and ending punctuation
    if caption[0].isupper():
        score += 0.5
    if caption.endswith("."):
        score += 0.5

    return score


def is_caption_valid(caption: str) -> tuple[bool, str]:
    """
    Validates caption according to quality rules.
    Returns (is_valid, reason).
    """
    text = caption.strip()
    if not text:
        return False, "empty_caption"

    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False, "very_short_caption_words"

    if len(text) < MIN_CHAR_COUNT:
        return False, "very_short_caption_chars"

    text_lower = text.lower()
    # Reject captions that start with purely meta phrases without substantial content
    if text_lower.startswith(("a close up of", "close up of", "a photo of a photo")):
        return False, "meta_phrase"

    return True, "valid"


def main():
    print("=" * 70)
    print("PixelDream AI - Task 1 V4 Dataset Preparation")
    print("=" * 70)
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Images source  : {IMAGE_DIR}")
    print(f"Captions file  : {CAPTIONS_FILE}")
    print(f"Instances file : {INSTANCES_FILE}")
    print(f"Target records : {TARGET_RECORDS}")
    print(f"Random seed    : {SEED}")
    print()

    # Verify input paths
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    if not CAPTIONS_FILE.exists():
        raise FileNotFoundError(f"Captions file not found: {CAPTIONS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load COCO Captions
    print("Step 1: Loading COCO captions...")
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        coco_captions = json.load(f)

    coco_images = {img["id"]: img["file_name"] for img in coco_captions["images"]}
    annotations = coco_captions["annotations"]

    total_images_considered = len(coco_images)
    total_captions_considered = len(annotations)

    print(f"  Total COCO images considered   : {total_images_considered}")
    print(f"  Total COCO captions considered : {total_captions_considered}")

    # 2. Load COCO Instances for category awareness
    img_to_categories = defaultdict(set)
    img_to_supercategories = defaultdict(set)
    category_id_to_name = {}

    if INSTANCES_FILE.exists():
        print("\nStep 2: Loading COCO category annotations for diversity...")
        with open(INSTANCES_FILE, "r", encoding="utf-8") as f:
            coco_instances = json.load(f)

        for cat in coco_instances["categories"]:
            category_id_to_name[cat["id"]] = cat["name"]

        for ann in coco_instances["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            cat_name = category_id_to_name.get(cat_id, "unknown")
            img_to_categories[img_id].add(cat_name)

        cat_id_to_super = {
            c["id"]: c["supercategory"] for c in coco_instances["categories"]
        }
        for ann in coco_instances["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            super_name = cat_id_to_super.get(cat_id, "other")
            img_to_supercategories[img_id].add(super_name)

        print(f"  Loaded category labels for {len(img_to_categories)} images.")
    else:
        print("\nStep 2: Instances file not found, proceeding without category labels.")

    # 3. Filter and score captions per image
    print("\nStep 3: Filtering captions and selecting best descriptive caption per image...")
    img_captions = defaultdict(list)
    captions_rejected_count = 0
    rejection_reasons = Counter()

    for ann in annotations:
        img_id = ann["image_id"]
        raw_caption = ann["caption"]

        valid, reason = is_caption_valid(raw_caption)
        if not valid:
            captions_rejected_count += 1
            rejection_reasons[reason] += 1
            continue

        img_captions[img_id].append(raw_caption.strip())

    print(f"  Valid captions retained : {sum(len(c) for c in img_captions.values())}")
    print(f"  Captions rejected       : {captions_rejected_count}")
    for reason, count in rejection_reasons.items():
        print(f"    - {reason}: {count}")

    # 4. Verify local image existence and select the single best caption per image
    candidate_records = []
    category_names_all = set(category_id_to_name.values())

    for img_id, caption_list in img_captions.items():
        if img_id not in coco_images:
            continue

        file_name = coco_images[img_id]
        image_path = IMAGE_DIR / file_name

        if not image_path.exists():
            continue

        if not caption_list:
            continue

        cats = img_to_categories.get(img_id, set())

        # Score candidate captions
        scored_captions = [
            (score_caption(cap, cats or category_names_all), cap)
            for cap in caption_list
        ]
        # Sort descending by score, tie-break by longer caption
        scored_captions.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

        rel_path = f"dataset/images/val2017/{file_name}"

        candidate_records.append({
            "image_id": img_id,
            "file_name": rel_path,
            "scored_captions": scored_captions,
            "categories": sorted(list(cats)),
            "supercategories": sorted(list(img_to_supercategories.get(img_id, set()))),
        })

    print(f"\nStep 4: Unique local candidate images with quality captions: {len(candidate_records)}")

    if len(candidate_records) < TARGET_RECORDS:
        raise RuntimeError(
            f"Cannot meet target: only {len(candidate_records)} candidates available, "
            f"but {TARGET_RECORDS} are required!"
        )

    # 5. Deterministic Diverse Sampling across Supercategories & Categories
    print(f"\nStep 5: Performing category-aware diverse sampling to select exactly {TARGET_RECORDS} images...")
    random.seed(SEED)

    # Group candidate records by supercategory (or 'other' if uncategorized)
    super_buckets = defaultdict(list)
    for rec in candidate_records:
        sups = rec["supercategories"]
        if sups:
            for s in sups:
                super_buckets[s].append(rec)
        else:
            super_buckets["other"].append(rec)

    # Shuffle each bucket deterministically
    for s in super_buckets:
        random.shuffle(super_buckets[s])

    # Balanced round-robin selection across supercategories to ensure diversity
    selected_dict = {}
    super_keys = sorted(list(super_buckets.keys()))

    bucket_ptrs = {s: 0 for s in super_keys}
    while len(selected_dict) < TARGET_RECORDS:
        made_progress = False
        for s in super_keys:
            if len(selected_dict) >= TARGET_RECORDS:
                break
            bucket = super_buckets[s]
            ptr = bucket_ptrs[s]
            while ptr < len(bucket):
                candidate = bucket[ptr]
                ptr += 1
                bucket_ptrs[s] = ptr
                img_id = candidate["image_id"]
                if img_id not in selected_dict:
                    selected_dict[img_id] = candidate
                    made_progress = True
                    break
        if not made_progress:
            # Fallback if buckets exhausted: draw from remaining unselected
            remaining = [r for r in candidate_records if r["image_id"] not in selected_dict]
            random.shuffle(remaining)
            for r in remaining:
                if len(selected_dict) >= TARGET_RECORDS:
                    break
                selected_dict[r["image_id"]] = r
            break

    final_selected = list(selected_dict.values())
    # Deterministic sort for reproducible output order
    final_selected.sort(key=lambda r: r["image_id"])

    # Assign best caption per image, ensuring no duplicate caption across the dataset
    seen_captions = set()
    for r in final_selected:
        chosen_caption = None
        chosen_score = None
        for score, cap in r["scored_captions"]:
            if cap not in seen_captions:
                chosen_caption = cap
                chosen_score = score
                seen_captions.add(cap)
                break
        if chosen_caption is None:
            # Fallback to top scored caption if all candidates were taken
            chosen_score, chosen_caption = r["scored_captions"][0]
        r["caption"] = chosen_caption
        r["score"] = chosen_score

    # 6. Verification of the selected set
    selected_count = len(final_selected)
    unique_images_count = len(set(r["file_name"] for r in final_selected))
    empty_captions_count = sum(1 for r in final_selected if not r["caption"].strip())
    very_short_count = sum(1 for r in final_selected if len(r["caption"].split()) < MIN_WORD_COUNT)
    duplicate_captions_count = selected_count - len(set(r["caption"] for r in final_selected))

    # All files exist locally
    all_files_exist = all(
        (PROJECT_ROOT / r["file_name"]).exists() for r in final_selected
    )

    print()
    print("=" * 70)
    print("Dataset Verification Checks")
    print("=" * 70)
    print(f"Selected records         : {selected_count} (target: {TARGET_RECORDS})")
    print(f"Unique images            : {unique_images_count} (target: {TARGET_RECORDS})")
    print(f"Empty captions           : {empty_captions_count} (target: 0)")
    print(f"Very short captions      : {very_short_count} (target: 0)")
    print(f"Duplicate images         : {selected_count - unique_images_count} (target: 0)")
    print(f"Duplicate captions       : {duplicate_captions_count}")
    print(f"All image files exist    : {all_files_exist}")
    print("=" * 70)

    if selected_count != TARGET_RECORDS:
        raise RuntimeError(f"Selected count {selected_count} does not match {TARGET_RECORDS}!")
    if unique_images_count != TARGET_RECORDS:
        raise RuntimeError(f"Unique images count {unique_images_count} does not match {TARGET_RECORDS}!")
    if empty_captions_count != 0:
        raise RuntimeError(f"Found {empty_captions_count} empty captions!")
    if not all_files_exist:
        raise RuntimeError("Some selected images do not exist on disk!")

    # 7. Compute caption statistics
    word_counts = [len(r["caption"].split()) for r in final_selected]
    char_counts = [len(r["caption"]) for r in final_selected]

    caption_length_stats = {
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "mean_words": round(statistics.mean(word_counts), 2),
        "median_words": round(statistics.median(word_counts), 2),
        "min_characters": min(char_counts),
        "max_characters": max(char_counts),
        "mean_characters": round(statistics.mean(char_counts), 2),
        "median_characters": round(statistics.median(char_counts), 2),
    }

    # 8. Compute category distribution
    super_dist = Counter()
    cat_dist = Counter()
    for r in final_selected:
        for s in r["supercategories"]:
            super_dist[s] += 1
        for c in r["categories"]:
            cat_dist[c] += 1

    category_distribution = {
        "by_supercategory": dict(super_dist.most_common()),
        "top_25_categories": dict(cat_dist.most_common(25)),
        "total_categories_represented": len(cat_dist),
    }

    # 9. Save metadata.jsonl
    print(f"\nStep 6: Saving JSONL metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for r in final_selected:
            json_line = json.dumps({
                "file_name": r["file_name"],
                "caption": r["caption"]
            }, ensure_ascii=False)
            f.write(json_line + "\n")

    # 10. Save dataset_report.json
    dataset_report = {
        "total_coco_images_considered": total_images_considered,
        "total_captions_considered": total_captions_considered,
        "unique_images_available": len(candidate_records),
        "captions_rejected": captions_rejected_count,
        "selected_records": selected_count,
        "unique_selected_images": unique_images_count,
        "empty_caption_count": empty_captions_count,
        "very_short_caption_count": very_short_count,
        "duplicate_image_count": selected_count - unique_images_count,
        "duplicate_caption_count": duplicate_captions_count,
        "caption_length_statistics": caption_length_stats,
        "category_distribution": category_distribution,
        "random_seed": SEED,
        "status_confirmation": {
            "selected_records": selected_count,
            "unique_images": unique_images_count,
            "empty_captions": empty_captions_count,
            "all_local_images_verified": True
        }
    }

    print(f"Step 7: Saving dataset report to {REPORT_FILE}...")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("DATASET PREPARATION V4 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Metadata file  : {METADATA_FILE}")
    print(f"Report file    : {REPORT_FILE}")
    print(f"Records        : {selected_count}")
    print(f"Unique images  : {unique_images_count}")
    print(f"Empty captions : {empty_captions_count}")
    print()
    print("Sample records:")
    for r in final_selected[:3]:
        print(f"  - Image  : {r['file_name']}")
        print(f"    Caption: {r['caption']}")
        print(f"    Cats   : {', '.join(r['categories'][:5])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
