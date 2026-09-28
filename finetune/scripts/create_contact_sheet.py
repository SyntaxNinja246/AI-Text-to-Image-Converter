"""
Script: create_contact_sheet.py
Purpose: Generates a 4x5 contact sheet comparing Tiny-SD LoRA V2 vs V4 models
         across 5 benchmark prompts.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont

def build_contact_sheet():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs", "model_comparison_v2_v4")
    metadata_path = os.path.join(output_dir, "comparison_metadata.json")
    out_contact_sheet = os.path.join(output_dir, "contact_sheet.png")

    # Define the 4 rows in exact order
    row_specs = [
        {
            "num": 1,
            "label": "1. V2 @ 0.75",
            "model_tag": "V2 (tiny-sd-lora-v2)",
            "scale_tag": "Scale: 0.75",
            "dir_name": "v2_scale_0.75",
            "accent_color": (99, 102, 241),   # Indigo
            "badge_bg": (49, 46, 129),        # Dark indigo
        },
        {
            "num": 2,
            "label": "2. V4 @ 0.50",
            "model_tag": "V4 (tiny-sd-lora-v4)",
            "scale_tag": "Scale: 0.50",
            "dir_name": "v4_scale_0.5",
            "accent_color": (6, 182, 212),    # Cyan
            "badge_bg": (22, 78, 99),         # Dark cyan
        },
        {
            "num": 3,
            "label": "3. V4 @ 0.75",
            "model_tag": "V4 (tiny-sd-lora-v4)",
            "scale_tag": "Scale: 0.75",
            "dir_name": "v4_scale_0.75",
            "accent_color": (59, 130, 246),   # Blue
            "badge_bg": (30, 58, 138),        # Dark blue
        },
        {
            "num": 4,
            "label": "4. V4 @ 1.00",
            "model_tag": "V4 (tiny-sd-lora-v4)",
            "scale_tag": "Scale: 1.00",
            "dir_name": "v4_scale_1.0",
            "accent_color": (168, 85, 247),   # Purple
            "badge_bg": (88, 28, 135),        # Dark purple
        }
    ]

    # Benchmark prompt definitions in column order
    cols = [
        {"idx": 1, "seed": 101, "file": "img_01_seed101.png", "prompt": "a motorcycle parked outdoors"},
        {"idx": 2, "seed": 102, "file": "img_02_seed102.png", "prompt": "a cozy room with chairs"},
        {"idx": 3, "seed": 103, "file": "img_03_seed103.png", "prompt": "a person standing in a kitchen"},
        {"idx": 4, "seed": 104, "file": "img_04_seed104.png", "prompt": "a desk with electronic equipment"},
        {"idx": 5, "seed": 105, "file": "img_05_seed105.png", "prompt": "a group of people outdoors"},
    ]

    # Fonts
    font_bold_path = "C:/Windows/Fonts/segoeuib.ttf"
    font_reg_path = "C:/Windows/Fonts/segoeui.ttf"
    font_mono_path = "C:/Windows/Fonts/consola.ttf"

    font_title = ImageFont.truetype(font_bold_path, 24)
    font_subtitle = ImageFont.truetype(font_reg_path, 13)
    font_row_label = ImageFont.truetype(font_bold_path, 17)
    font_row_meta = ImageFont.truetype(font_reg_path, 13)
    font_prompt = ImageFont.truetype(font_bold_path, 13)
    font_seed = ImageFont.truetype(font_mono_path, 12)

    # Canvas Dimensions
    img_w, img_h = 256, 256
    num_cols = 5
    num_rows = 4
    col_gap = 20
    margin_x = 32

    content_w = num_cols * img_w + (num_cols - 1) * col_gap  # 1280 + 80 = 1360
    total_w = content_w + 2 * margin_x                       # 1360 + 64 = 1424

    header_h = 96
    banner_h = 42
    banner_to_img_gap = 12
    caption_h = 58
    row_bottom_gap = 28

    single_row_h = banner_h + banner_to_img_gap + img_h + caption_h + row_bottom_gap
    total_h = header_h + num_rows * single_row_h + 20

    # Colors
    bg_color = (15, 23, 42)           # #0f172a slate-900
    card_bg = (30, 41, 59)            # #1e293b slate-800
    card_border = (51, 65, 85)        # #334155 slate-700
    text_white = (248, 250, 252)      # #f8fafc slate-50
    text_dim = (148, 163, 184)        # #94a3b8 slate-400
    text_muted = (100, 116, 139)      # #64748b slate-500
    accent_seed = (56, 189, 248)      # #38bdf8 light sky blue
    banner_bg = (24, 33, 47)          # Slightly lighter slate
    banner_border = (51, 65, 85)      # Slate-700

    sheet = Image.new("RGB", (total_w, total_h), color=bg_color)
    draw = ImageDraw.Draw(sheet)

    # 1. Main Header
    title_text = "Tiny-SD Fine-Tuning: Model Comparison (LoRA V2 vs V4)"
    subtitle_text = "4 Rows × 5 Benchmark Prompts  •  256×256 Resolution  •  Inference Steps: 10  •  Guidance Scale: 7.5"

    draw.text((margin_x, 24), title_text, fill=text_white, font=font_title)
    draw.text((margin_x, 58), subtitle_text, fill=text_dim, font=font_subtitle)

    # Header divider
    draw.line([(margin_x, 86), (margin_x + content_w, 86)], fill=card_border, width=1)

    curr_y = header_h

    # 2. Render each row
    for r_idx, r_spec in enumerate(row_specs):
        row_y = curr_y

        # A. Row Banner
        banner_bbox = [margin_x, row_y, margin_x + content_w, row_y + banner_h]
        draw.rounded_rectangle(banner_bbox, radius=6, fill=banner_bg, outline=banner_border, width=1)

        # Row color indicator bar on the far left
        indicator_bbox = [margin_x + 2, row_y + 2, margin_x + 8, row_y + banner_h - 2]
        draw.rounded_rectangle(indicator_bbox, radius=3, fill=r_spec["accent_color"])

        # Row label pill
        label_text = r_spec["label"]
        draw.text((margin_x + 20, row_y + 10), label_text, fill=text_white, font=font_row_label)

        # Row meta tag on the right side of the banner
        meta_text = f"{r_spec['model_tag']}  |  {r_spec['scale_tag']}"
        meta_bbox = draw.textbbox((0, 0), meta_text, font=font_row_meta)
        meta_w = meta_bbox[2] - meta_bbox[0]
        meta_x = margin_x + content_w - meta_w - 20
        draw.text((meta_x, row_y + 12), meta_text, fill=text_dim, font=font_row_meta)

        # B. Images and Captions
        img_start_y = row_y + banner_h + banner_to_img_gap
        row_folder = os.path.join(output_dir, r_spec["dir_name"])

        for c_idx, col in enumerate(cols):
            col_x = margin_x + c_idx * (img_w + col_gap)
            img_path = os.path.join(row_folder, col["file"])

            # Card Background behind image and caption
            card_bbox = [col_x, img_start_y, col_x + img_w, img_start_y + img_h + caption_h]
            draw.rounded_rectangle(card_bbox, radius=6, fill=card_bg, outline=card_border, width=1)

            # Paste original image unchanged
            if os.path.exists(img_path):
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    # Ensure exact size
                    if im_rgb.size != (img_w, img_h):
                        im_rgb = im_rgb.resize((img_w, img_h), Image.Resampling.LANCZOS)
                    sheet.paste(im_rgb, (col_x, img_start_y))
            else:
                # Placeholder if image missing (should not happen)
                draw.rectangle([col_x, img_start_y, col_x + img_w, img_start_y + img_h], fill=(40, 40, 40))
                draw.text((col_x + 20, img_start_y + 100), "Image Missing", fill=(255, 100, 100), font=font_prompt)

            # Caption area below image
            cap_y = img_start_y + img_h

            # Divider line between image and caption
            draw.line([(col_x, cap_y), (col_x + img_w, cap_y)], fill=card_border, width=1)

            # Prompt text (centered)
            p_text = f"\"{col['prompt']}\""
            p_bbox = draw.textbbox((0, 0), p_text, font=font_prompt)
            p_w = p_bbox[2] - p_bbox[0]
            p_x = col_x + (img_w - p_w) // 2
            draw.text((p_x, cap_y + 8), p_text, fill=text_white, font=font_prompt)

            # Seed text (centered)
            s_text = f"Seed: {col['seed']}"
            s_bbox = draw.textbbox((0, 0), s_text, font=font_seed)
            s_w = s_bbox[2] - s_bbox[0]
            s_x = col_x + (img_w - s_w) // 2
            draw.text((s_x, cap_y + 32), s_text, fill=accent_seed, font=font_seed)

        curr_y += single_row_h

    # Ensure output directory exists and save
    os.makedirs(output_dir, exist_ok=True)
    sheet.save(out_contact_sheet, format="PNG", quality=95)
    print(f"Contact sheet successfully saved to: {out_contact_sheet}")
    print(f"Dimensions: {sheet.size[0]}x{sheet.size[1]} px")

if __name__ == "__main__":
    build_contact_sheet()
