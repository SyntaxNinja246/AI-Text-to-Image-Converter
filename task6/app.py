"""
PixelDream AI - Task 6 Unified Application
==========================================
End-to-end multi-modal AI pipeline combining:
- Task 1: Tiny-SD + verified V2 LoRA fine-tuned diffusion generator
- Task 2: Category-conditioned CGAN generation
- Task 3: Transformer caption preprocessing & prompt token analysis
- Task 4: COCO dataset exploration & summary analytics
- Task 5: Cross-attention inspection & spatial attribution
- Generation history & interactive gallery
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from typing import Tuple, List, Optional

# Ensure project root is in sys.path when running as a script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from PIL import Image

try:
    from task6.config import (
        PROJECT_ROOT,
        DEFAULT_PROMPT,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_STEPS,
        DEFAULT_GUIDANCE,
        DEFAULT_LORA_SCALE,
        DEFAULT_SEED,
        DEFAULT_RESOLUTION,
        CGAN_CATEGORIES,
        BASE_MODEL_ID,
    )
    from task6.pipeline import PixelDreamPipeline
    from task6.utils import (
        load_generation_history,
        get_gallery_items,
        load_task3_info,
        load_task4_info,
        load_task5_info,
        load_cgan_eval_image,
    )
except ImportError:
    from config import (
        PROJECT_ROOT,
        DEFAULT_PROMPT,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_STEPS,
        DEFAULT_GUIDANCE,
        DEFAULT_LORA_SCALE,
        DEFAULT_SEED,
        DEFAULT_RESOLUTION,
        CGAN_CATEGORIES,
        BASE_MODEL_ID,
    )
    from pipeline import PixelDreamPipeline
    from utils import (
        load_generation_history,
        get_gallery_items,
        load_task3_info,
        load_task4_info,
        load_task5_info,
        load_cgan_eval_image,
    )

# Custom CSS for modern dark-mode aesthetic
CUSTOM_CSS = """
body {
    background: #0b0f19 !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    background: #0b0f19 !important;
    color: #f1f5f9 !important;
}

.main-header {
    text-align: center;
    padding: 30px 20px 20px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 20px;
}

.main-title {
    font-size: 38px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.main-subtitle {
    color: #94a3b8 !important;
    font-size: 15px !important;
    font-weight: 400;
}

.panel-card {
    background: #131b2e !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 18px !important;
    margin-bottom: 16px !important;
}

.badge-v2 {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid #6366f1;
    color: #a5b4fc;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
}

.status-verified {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid #10b981;
    color: #6ee7b7;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}

.stat-box {
    background: #182238;
    border: 1px solid #24324f;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}

.stat-val {
    font-size: 24px;
    font-weight: 800;
    color: #38bdf8;
}

.stat-lbl {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}
"""

pipeline = PixelDreamPipeline.get_instance()


# =============================================================================
# Helper Handlers
# =============================================================================

def handle_generate_diffusion(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    lora_scale: float,
    resolution: str,
    seed: int,
) -> Tuple[Optional[str], str, List[str]]:
    """Handles text-to-image generation via Tiny-SD + V2 LoRA."""
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    try:
        out_path, elapsed, meta = pipeline.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            lora_scale=lora_scale,
            resolution=resolution,
            seed=seed,
        )

        status_md = (
            f"### Generation Successful\n\n"
            f"- **Execution Time:** `{elapsed:.2f} s`\n"
            f"- **Model:** `{meta['model']}`\n"
            f"- **LoRA Adapter:** `{meta['adapter']}` (Scale: `{meta['lora_scale']}`)\n"
            f"- **Resolution:** `{meta['resolution']}`\n"
            f"- **Denoising Steps:** `{meta['steps']}`\n"
            f"- **Guidance Scale:** `{meta['guidance']}`\n"
            f"- **Seed:** `{meta['seed']}`\n"
            f"- **Output File:** `{Path(out_path).name}`"
        )

        gallery = get_gallery_items()
        return out_path, status_md, gallery

    except Exception as e:
        raise gr.Error(f"Generation error: {str(e)}")


def handle_generate_cgan(
    category: str,
    seed: int,
) -> Tuple[Optional[Image.Image], str]:
    """Handles category-conditioned CGAN generation (Task 2)."""
    try:
        pil_img, out_path, meta = pipeline.generate_cgan_image(
            category=category,
            seed=seed,
        )

        status_md = (
            f"### CGAN Sample Generated\n\n"
            f"- **Model:** `Category-Conditioned CGAN (Task 2)`\n"
            f"- **Category:** `{category}` (Index: `{meta['category_index']}`)\n"
            f"- **Resolution:** `{meta['image_size']}`\n"
            f"- **Seed:** `{meta['seed']}`\n"
            f"- **Inference Time:** `{meta['generation_time_sec']:.4f} s`\n"
            f"- **File:** `{Path(out_path).name}`"
        )

        return pil_img, status_md

    except Exception as e:
        raise gr.Error(f"CGAN generation error: {str(e)}")


def handle_inspect_tokens(prompt: str) -> str:
    """Handles CLIP token inspection for any prompt."""
    if not prompt or not prompt.strip():
        return "Please enter a prompt to inspect."

    info = pipeline.inspect_prompt_tokens(prompt)
    table_rows = []
    for t in info["tokens"]:
        table_rows.append(f"| {t['pos']} | `{t['token']}` | `{t['id']}` |")

    table_md = "\n".join(table_rows)

    return (
        f"### CLIP Tokenizer Breakdown (Tiny-SD)\n\n"
        f"- **Model Tokenizer:** `{info['model_tokenizer']}`\n"
        f"- **Sequence Length:** `{info['active_token_count']} / {info['max_sequence_length']}`\n"
        f"- **Padding Tokens:** `{info['padding_token_count']}`\n\n"
        f"| Position | Token String | Token ID |\n"
        f"| :---: | :--- | :---: |\n"
        f"{table_md}\n\n"
        f"**Raw Token IDs Preview (First 16):**\n"
        f"`{info['raw_input_ids_preview']}`"
    )


def handle_select_gallery_item(evt: gr.SelectData):
    """Loads a clicked gallery image back into prompt and parameters."""
    history = load_generation_history()
    idx = evt.index
    if isinstance(idx, tuple):
        idx = idx[0]

    # History is displayed reversed in gallery
    rev_history = list(reversed(history))
    if idx < 0 or idx >= len(rev_history):
        return None, "", "", 10, 7.5, 0.75, "256x256", 42, "Item not found"

    item = rev_history[idx]
    status = (
        f"### Loaded Historical Generation\n\n"
        f"- **Prompt:** \"{item.get('prompt', '')}\"\n"
        f"- **Seed:** `{item.get('seed', 42)}`\n"
        f"- **Model:** `{item.get('model', BASE_MODEL_ID)}`\n"
        f"- **Adapter:** `{item.get('adapter', 'tiny-sd-lora-v2')}` (Scale: `{item.get('lora_scale', 0.75)}`)"
    )

    return (
        item.get("image") or item.get("output_path"),
        item.get("prompt", ""),
        item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        item.get("steps", DEFAULT_STEPS),
        item.get("guidance", DEFAULT_GUIDANCE),
        item.get("lora_scale", DEFAULT_LORA_SCALE),
        item.get("resolution", DEFAULT_RESOLUTION),
        item.get("seed", DEFAULT_SEED),
        status,
    )


def random_seed_generator() -> int:
    return random.randint(0, 2147483647)


# =============================================================================
# Build Gradio UI
# =============================================================================

def build_app() -> gr.Blocks:
    # Preload artifact metadata
    task3_info = load_task3_info()
    task4_info = load_task4_info()
    task5_info = load_task5_info()
    cgan_eval_img = load_cgan_eval_image()

    with gr.Blocks(title="PixelDream AI - Task 6 Pipeline") as demo:

        # Header
        gr.HTML(
            """
            <div class="main-header">
                <div class="main-title">PixelDream AI</div>
                <div class="main-subtitle">
                    Task 6 &mdash; Text-to-Image Pipeline: Prompt &rarr; Text Preprocessing &rarr; Text Embedding &rarr; Image Generation
                </div>
            </div>
            """
        )

        with gr.Tabs() as tabs:

            # =================================================================
            # TAB 1: TEXT-TO-IMAGE GENERATION (Diffusion + LoRA V2)
            # =================================================================
            with gr.TabItem("Generate", id="tab_generate"):
                with gr.Row():
                    # Left Column: Inputs & Controls
                    with gr.Column(scale=1):
                        prompt_input = gr.Textbox(
                            label="Prompt  (Tiny-SD + V2 LoRA · CLIP tokenized · 77-token max)",
                            value=DEFAULT_PROMPT,
                            placeholder="Describe the image you want to generate...",
                            lines=2,
                            elem_id="prompt_input",
                        )

                        # Compact preset row
                        with gr.Row():
                            gr.Markdown("*Presets:*", elem_id="presets_label")
                            btn_preset1 = gr.Button("Cozy Room", size="sm", min_width=80)
                            btn_preset2 = gr.Button("Motorcycle", size="sm", min_width=80)
                            btn_preset3 = gr.Button("Kitchen", size="sm", min_width=80)
                            btn_preset4 = gr.Button("Desk", size="sm", min_width=60)
                            btn_preset5 = gr.Button("People", size="sm", min_width=60)

                        with gr.Row():
                            steps_slider = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=DEFAULT_STEPS,
                                step=1,
                                label="Steps",
                                info="10 recommended on CPU",
                            )
                            guidance_slider = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=DEFAULT_GUIDANCE,
                                step=0.5,
                                label="Guidance Scale",
                            )

                        with gr.Row():
                            lora_scale_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.5,
                                value=DEFAULT_LORA_SCALE,
                                step=0.05,
                                label="LoRA Scale",
                                info="0.75 optimal",
                            )
                            seed_input = gr.Number(
                                label="Seed",
                                value=DEFAULT_SEED,
                                precision=0,
                            )

                        with gr.Row():
                            btn_generate = gr.Button(
                                "Generate",
                                variant="primary",
                                elem_id="btn_generate",
                                scale=3,
                            )
                            btn_random_seed = gr.Button("🎲", size="sm", scale=1)
                            btn_clear = gr.Button("Reset", variant="secondary", size="sm", scale=1)

                        with gr.Accordion("Advanced Settings", open=False):
                            negative_prompt_input = gr.Textbox(
                                label="Negative Prompt",
                                value=DEFAULT_NEGATIVE_PROMPT,
                                placeholder="Things to avoid...",
                                lines=1,
                                elem_id="negative_prompt_input",
                            )
                            resolution_dropdown = gr.Dropdown(
                                choices=["256x256", "128x128", "512x512"],
                                value=DEFAULT_RESOLUTION,
                                label="Resolution",
                            )

                    # Right Column: Image Output
                    with gr.Column(scale=1):
                        gr.HTML('<p style="color:#10b981;font-size:13px;margin:0 0 8px 0;">✓ Model ready &nbsp;·&nbsp; Tiny-SD + LoRA V2</p>')

                        output_image = gr.Image(
                            label="Generated Image",
                            type="filepath",
                            elem_id="output_image",
                            height=380,
                        )

                        with gr.Accordion("Generation Info", open=False):
                            status_output = gr.Markdown(
                                "Generation details will appear here after each run."
                            )

                # History Gallery
                gr.Markdown("---")
                gr.Markdown("**Recent History** — click an image to restore its settings")
                history_gallery = gr.Gallery(
                    value=get_gallery_items(),
                    label="History",
                    columns=6,
                    rows=1,
                    height=160,
                    allow_preview=True,
                    elem_id="history_gallery",
                )

            # =================================================================
            # TAB 2: CONDITIONAL GAN (Task 2)
            # =================================================================
            with gr.TabItem("Conditional GAN (Task 2)", id="tab_cgan"):
                gr.Markdown(
                    "### Category-Conditioned Image Generation (Task 2)\n"
                    "Generates 64×64 pixel objects directly using the custom-trained category-conditioned GAN. "
                    "Trained on COCO instances across 5 distinct object classes."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        cgan_category_dropdown = gr.Dropdown(
                            choices=CGAN_CATEGORIES,
                            value="car",
                            label="Select Category",
                            info="Trained category conditioning vector",
                        )
                        cgan_seed_input = gr.Number(
                            label="Noise Seed",
                            value=42,
                            precision=0,
                        )
                        btn_cgan_random = gr.Button("Randomize Noise Seed", size="sm")
                        btn_generate_cgan = gr.Button(
                            "Generate CGAN Sample",
                            variant="primary",
                        )

                        cgan_status_box = gr.Markdown(
                            "### CGAN Ready\n"
                            "Select a category and click **Generate CGAN Sample**."
                        )

                    with gr.Column(scale=1):
                        cgan_output_image = gr.Image(
                            label="Generated CGAN Sample (64x64)",
                            type="pil",
                            height=320,
                        )

                # Precomputed Evaluation Grid
                if cgan_eval_img:
                    gr.Markdown("---")
                    gr.Markdown("### Task 2 CGAN Precomputed Evaluation Matrix (10 samples × 5 categories)")
                    gr.Image(
                        value=cgan_eval_img,
                        label="Task 2 CGAN Evaluation Grid (Car, Chair, Bottle, Bird, Boat)",
                        height=280,
                    )

            # =================================================================
            # TAB 3: ATTENTION MECHANISM (Task 5)
            # =================================================================
            with gr.TabItem("Attention Analysis (Task 5)", id="tab_attention"):
                gr.Markdown(
                    "### Cross-Attention Spatial Attribution (Task 5)\n"
                    "Visualizes how text tokens in the conditioning prompt spatially guide diffusion generation "
                    "across the 9 cross-attention blocks of the Tiny-SD UNet."
                )

                with gr.Row():
                    # Interactive Tokenizer
                    with gr.Column(scale=1):
                        gr.Markdown("#### Interactive Prompt Token Inspector")
                        inspect_prompt_input = gr.Textbox(
                            label="Enter Prompt to Tokenize",
                            value="a red car beside a blue house",
                            lines=2,
                        )
                        btn_inspect_tokens = gr.Button("Inspect Tokens via CLIP Tokenizer", variant="secondary")
                        token_inspector_output = gr.Markdown(
                            "Click **Inspect Tokens** to see sequence position and token IDs."
                        )

                    # Attention Visualizations
                    with gr.Column(scale=1):
                        gr.Markdown("#### Task 5 Precomputed Cross-Attention Maps")
                        gr.Markdown("Benchmark test: `\"a red car beside a blue house\"`")

                        artifact_keys = list(task5_info.get("artifacts", {}).keys())
                        if artifact_keys:
                            artifact_selector = gr.Radio(
                                choices=artifact_keys,
                                value=artifact_keys[0],
                                label="Select Attention Visualization",
                            )
                            attention_display = gr.Image(
                                value=task5_info["artifacts"][artifact_keys[0]],
                                label="Spatial Attention Map",
                                height=360,
                            )
                        else:
                            gr.Markdown("No precomputed attention maps found in `attention/outputs/`.")

                # Architecture Summary
                gr.Markdown("---")
                gr.Markdown("#### Attention Architecture Specifications")
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">18</div>
                                <div class="stat-lbl">Total Attention Processors</div>
                            </div>
                            """
                        )
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">9</div>
                                <div class="stat-lbl">Cross-Attention Blocks</div>
                            </div>
                            """
                        )
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">8</div>
                                <div class="stat-lbl">Attention Heads</div>
                            </div>
                            """
                        )
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">32 × 32</div>
                                <div class="stat-lbl">Spatial Feature Grid (1024)</div>
                            </div>
                            """
                        )

            # =================================================================
            # TAB 4: DATASET & EMBEDDINGS (Tasks 3 & 4)
            # =================================================================
            with gr.TabItem("Dataset & Embeddings (Tasks 3 & 4)", id="tab_analysis"):
                gr.Markdown("### Dataset Analytics & Transformer Text Embeddings")

                # Metrics Row
                with gr.Row():
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">5,000</div>
                                <div class="stat-lbl">COCO 2017 Val Images (Task 4)</div>
                            </div>
                            """
                        )
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">25,014</div>
                                <div class="stat-lbl">Total Captions Processed (Task 3)</div>
                            </div>
                            """
                        )
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">80</div>
                                <div class="stat-lbl">Object Categories (COCO)</div>
                            </div>
                            """
                        )
                    with gr.Column():
                        gr.HTML(
                            """
                            <div class="stat-box">
                                <div class="stat-val">2.96 GB</div>
                                <div class="stat-lbl">Dense Embedding Store [25014, 77, 768]</div>
                            </div>
                            """
                        )

                gr.Markdown("---")
                with gr.Row():
                    # Task 3 Embeddings Info
                    with gr.Column(scale=1):
                        gr.Markdown("#### Task 3: Transformer Embeddings")
                        gr.Markdown(
                            f"- **Text Encoder:** `{task3_info.get('model', 'segmind/tiny-sd')}` (CLIP ViT-L/14)\n"
                            f"- **Embedding Shape:** `[25014, 77, 768]`\n"
                            f"- **Precision:** `{task3_info.get('embedding_dtype', 'float16')}`\n"
                            f"- **Sequence Length:** `{task3_info.get('sequence_length', 77)}` tokens\n"
                            f"- **Hidden Dimension:** `{task3_info.get('embedding_dimension', 768)}`\n"
                            f"- **Embedding Storage:** `dataset/analysis_results/embeddings/caption_embeddings.npy` (2.96 GB)\n"
                            f"- **Status:** Ready and preserved without recomputation."
                        )

                    # Task 4 Dataset Exploration Charts
                    with gr.Column(scale=1):
                        gr.Markdown("#### Task 4: COCO Exploration Charts")
                        chart_keys = list(task4_info.get("charts", {}).keys())
                        if chart_keys:
                            chart_selector = gr.Dropdown(
                                choices=chart_keys,
                                value=chart_keys[0],
                                label="Select COCO Distribution Plot",
                            )
                            chart_display = gr.Image(
                                value=task4_info["charts"][chart_keys[0]],
                                label="Distribution Chart",
                                height=380,
                            )
                        else:
                            gr.Markdown("No precomputed distribution charts found in `dataset/analysis_results/`.")

            # =================================================================
            # TAB 5: ABOUT & PIPELINE ARCHITECTURE
            # =================================================================
            with gr.TabItem("About & Pipeline", id="tab_about"):
                gr.Markdown("### PixelDream AI End-to-End Architecture")

                gr.Markdown(
                    """
                    PixelDream AI is a comprehensive multi-task research and engineering pipeline that integrates:
                    
                    | Component | Role | Artifacts & Models | Status |
                    | :--- | :--- | :--- | :--- |
                    | **Task 1** | **LoRA Diffusion Fine-Tuning** | `tiny-sd-lora-v2` (Rank 4, Alpha 4, 250 steps) | **Final / Production** |
                    | **Task 1 (Exp)** | High-Rank LoRA Research | `tiny-sd-lora-v3` (Rank 8, tested capacity limits) | Experimental Only |
                    | **Task 1 (Exp)** | Large-Dataset LoRA Research | `tiny-sd-lora-v4` (1500 samples, 1125 steps, over-adapted) | Experimental Only |
                    | **Task 2** | **Conditional GAN** | 5-class category generator (64×64 pixels) | Completed & Integrated |
                    | **Task 3** | **Transformer Preprocessing** | CLIP ViT-L/14 text embeddings (25,014 records) | Completed & Integrated |
                    | **Task 4** | **COCO Dataset Analysis** | 5,000 images, 25,014 captions, 80 categories | Completed & Integrated |
                    | **Task 5** | **Attention Attribution** | 9 cross-attention blocks spatial heatmap extraction | Completed & Integrated |
                    | **Task 6** | **Unified Production Pipeline** | Unified Gradio interface, strict verification, history | **Completed** |
                    """
                )

                gr.Markdown("---")
                gr.Markdown(
                    """
                    #### Model Lineage Decision: Why V2 is the Final Production Model
                    - **V2 (Production Default)**: Trained for 250 steps on 500 COCO images with rank 4, alpha 4. Visual inspection and quantitative edge sharpness analysis (Laplacian variance `418.25`) demonstrated that V2 achieved the highest structural clarity, best textural fidelity, and zero artifacting at scale 0.75.
                    - **V3 (Experimental)**: Increased rank to 8, but caused cross-attention instability without quality gain.
                    - **V4 (Experimental)**: Evaluated on 1,500 diverse samples for 1,125 steps. Proved over-adapted, exhibiting texture noise and figure distortion at scale 0.75–1.00.
                    
                    #### Hardware & Execution Constraints
                    - Running entirely in CPU float32 mode.
                    - LoRA weights verified strictly (144 / 144 tensors mapped).
                    - In-memory model caching prevents redundant reload latency.
                    """
                )

        # =====================================================================
        # Event Wiring
        # =====================================================================

        # Preset Prompt Buttons
        btn_preset1.click(lambda: "a cozy room with chairs", outputs=[prompt_input])
        btn_preset2.click(lambda: "a motorcycle parked outdoors", outputs=[prompt_input])
        btn_preset3.click(lambda: "a person standing in a kitchen", outputs=[prompt_input])
        btn_preset4.click(lambda: "a desk with electronic equipment", outputs=[prompt_input])
        btn_preset5.click(lambda: "a group of people outdoors", outputs=[prompt_input])

        # Random Seed Buttons
        btn_random_seed.click(random_seed_generator, outputs=[seed_input])
        btn_cgan_random.click(random_seed_generator, outputs=[cgan_seed_input])

        # Clear Defaults
        btn_clear.click(
            lambda: (
                DEFAULT_PROMPT,
                DEFAULT_NEGATIVE_PROMPT,
                DEFAULT_STEPS,
                DEFAULT_GUIDANCE,
                DEFAULT_LORA_SCALE,
                DEFAULT_RESOLUTION,
                DEFAULT_SEED,
            ),
            outputs=[
                prompt_input,
                negative_prompt_input,
                steps_slider,
                guidance_slider,
                lora_scale_slider,
                resolution_dropdown,
                seed_input,
            ],
        )

        # Diffusion Generate Action
        btn_generate.click(
            fn=handle_generate_diffusion,
            inputs=[
                prompt_input,
                negative_prompt_input,
                steps_slider,
                guidance_slider,
                lora_scale_slider,
                resolution_dropdown,
                seed_input,
            ],
            outputs=[
                output_image,
                status_output,
                history_gallery,
            ],
        )

        # Gallery Selection Action
        history_gallery.select(
            fn=handle_select_gallery_item,
            outputs=[
                output_image,
                prompt_input,
                negative_prompt_input,
                steps_slider,
                guidance_slider,
                lora_scale_slider,
                resolution_dropdown,
                seed_input,
                status_output,
            ],
        )

        # CGAN Generate Action
        btn_generate_cgan.click(
            fn=handle_generate_cgan,
            inputs=[
                cgan_category_dropdown,
                cgan_seed_input,
            ],
            outputs=[
                cgan_output_image,
                cgan_status_box,
            ],
        )

        # Attention Token Inspector Action
        btn_inspect_tokens.click(
            fn=handle_inspect_tokens,
            inputs=[inspect_prompt_input],
            outputs=[token_inspector_output],
        )

        # Attention Heatmap Selector Action
        if artifact_keys:
            def update_attention_map(chosen_key):
                return task5_info["artifacts"].get(chosen_key)
            artifact_selector.change(
                fn=update_attention_map,
                inputs=[artifact_selector],
                outputs=[attention_display],
            )

        # Dataset Exploration Chart Selector Action
        if chart_keys:
            def update_dataset_chart(chosen_chart):
                return task4_info["charts"].get(chosen_chart)
            chart_selector.change(
                fn=update_dataset_chart,
                inputs=[chart_selector],
                outputs=[chart_display],
            )

    return demo


def main():
    demo = build_app()
    print("\n" + "=" * 70)
    print("PIXELDREAM AI - TASK 6: Launching Unified Gradio Web Interface")
    print("=" * 70)
    print("Default Model : segmind/tiny-sd + finetune/models/tiny-sd-lora-v2")
    print("LoRA Scale    : 0.75 (Optimal)")
    print("Port          : 7860 (or next available)")
    print("=" * 70 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
