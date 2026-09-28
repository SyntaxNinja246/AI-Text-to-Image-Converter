import os
import json
import time

import gradio as gr

from generator import ImageGenerator

CUSTOM_CSS = """
/* ================================
   GLOBAL
================================ */

body {
    background: #0b0b12 !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: #0b0b12 !important;
    color: #f5f5f7 !important;
}

/* ================================
   HEADER
================================ */

.app-header {
    text-align: center;
    padding: 35px 20px 25px;
}

.app-title {
    font-size: 42px !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin-bottom: 8px;
}

.app-subtitle {
    color: #a5a5b5 !important;
    font-size: 17px !important;
}

/* ================================
   PANELS
================================ */

.panel {
    background: #14141d !important;
    border: 1px solid #292936 !important;
    border-radius: 18px !important;
    padding: 22px !important;
}

/* ================================
   INPUTS
================================ */

textarea,
input {
    background: #0f0f17 !important;
    border: 1px solid #30303d !important;
    color: #ffffff !important;
}

textarea:focus,
input:focus {
    border-color: #8b5cf6 !important;
}

/* ================================
   BUTTONS
================================ */

.generate-btn {
    height: 52px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.secondary-btn {
    height: 48px !important;
}

/* ================================
   IMAGE
================================ */

.output-image {
    border-radius: 18px !important;
    overflow: hidden !important;
}

/* ================================
   STATUS
================================ */

.status-box {
    background: #101018 !important;
    border: 1px solid #292936 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ================================
   HISTORY
================================ */

.history-title {
    margin-top: 25px !important;
}

.gallery {
    border-radius: 16px !important;
}

/* ================================
   SLIDERS
================================ */

input[type="range"] {
    accent-color: #8b5cf6 !important;
}
"""


# ==========================================
# CONFIGURATION
# ==========================================

HISTORY_FILE = "generation_history.json"
IMAGE_FOLDER = "generated_images"


# ==========================================
# START GENERATOR
# ==========================================

print("Starting Text-to-Image Generator...")
generator = ImageGenerator()


# ==========================================
# HISTORY MANAGEMENT
# ==========================================

def load_history():
    """Load saved generation metadata."""

    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)

        # Only keep entries whose images still exist
        valid_history = []

        for item in history:
            if os.path.exists(item["image"]):
                valid_history.append(item)

        return valid_history

    except (json.JSONDecodeError, OSError):
        print("Could not load history. Starting with empty history.")
        return []


def save_history():
    """Save generation metadata to disk."""

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            generation_history,
            file,
            indent=4
        )


generation_history = load_history()
def parse_resolution(resolution):
    """Convert resolution string into width and height."""

    try:
        width, height = resolution.lower().split("x")
        return int(width), int(height)

    except (ValueError, AttributeError):
        raise gr.Error("Invalid resolution selected.")


def get_gallery_images():
    """Return image paths for the Gradio gallery."""

    return [
        item["image"]
        for item in generation_history
    ]


# ==========================================
# GENERATION
# ==========================================

def generate_image(
    prompt,
    negative_prompt,
    steps,
    guidance,
    resolution,
    seed
):

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt first.")

    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()

    width,height=parse_resolution(resolution)

    start_time = time.time()

    output_path = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=width,
        height=height,
        seed=int(seed)
    )

    elapsed = time.time() - start_time

    # Store generation information
    generation_info = {
        "image": output_path,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": int(steps),
        "guidance": float(guidance),
        "resolution":resolution,
        "width":width,
        "height":height,
        "seed": int(seed)
    }

    generation_history.append(generation_info)

    # Save history permanently
    save_history()

    status = (
    f"### Generation Complete\n\n"
    f"**Time:** `{elapsed:.1f} seconds`  \n"
    f"**Resolution:** `{resolution}`  \n"
    f"**Steps:** `{int(steps)}`  \n"
    f"**Guidance:** `{float(guidance)}`  \n"
    f"**Seed:** `{int(seed)}`"
    )

    return (
        output_path,
        status,
        get_gallery_images()
    )


# ==========================================
# SELECT OLD GENERATION
# ==========================================

def select_generation(evt: gr.SelectData):

    index = evt.index

    # Gallery index should be an integer
    if isinstance(index, tuple):
        index = index[0]

    if not isinstance(index, int):
        raise gr.Error("Could not identify the selected image.")

    if index < 0 or index >= len(generation_history):
        raise gr.Error("Selected generation was not found.")

    item = generation_history[index]

    resolution=item.get(
        "resolution",
        f'{item.get("width",256)}x{item.get("height",256)}'
    )

    status = (
        " Previous generation loaded.\n\n"
        "You can modify the settings and generate again."
    )

    return (
        item["image"],
        item["prompt"],
        item["negative_prompt"],
        item["steps"],
        item["guidance"],
        resolution,
        item["seed"],
        status
    )


# ==========================================
# CLEAR CURRENT CONTROLS
# ==========================================

def clear_all():

    return (
        "",
        "blurry, distorted, low quality",
        10,
        7.5,
        "256x256",
        42,
        None,
        "Ready to generate."
    )

def random_seed():
    """Generate a random seed."""
    import random
    return random.randint(0, 2147483647)
# ==========================================
# UI
# ==========================================

with gr.Blocks(
    title="PixelDream AI",
    css=CUSTOM_CSS
) as app:

    gr.HTML(
        """
        <div class="app-header">
            <div class="app-title">
                PixelDream AI
            </div>
            
            <div class="app-subtitle">
                Turn your imagination into AI-generated images
            </div>
        </div>
        """
    )

    with gr.Row():

        # ==================================
        # LEFT SIDE
        # ==================================

        with gr.Column(scale=1,elem_classes="panel"):

            gr.Markdown("##  Generation Settings")

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want...",
                lines=5
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, distorted, low quality",
                lines=3
            )

            steps = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="Inference Steps"
            )

            guidance = gr.Slider(
                minimum=1,
                maximum=15,
                value=7.5,
                step=0.5,
                label="Guidance Scale"
            )

            resolution = gr.Dropdown(
                choices=[
                    "256x256",
                    "384x384",
                    "512x512"
                ],
                value = "256x256",
                label = "Resolution",
                info = "Higher resolution requires more CPU Time"
            )

            seed = gr.Number(
                value=42,
                precision=0,
                label="Seed"
            )

            with gr.Row():
                generate_button = gr.Button(
                    " Generate Image",
                    variant="primary",
                    elem_classes="generate-btn"
                    )

                regenerate_button = gr.Button(
                    " Regenerate",
                    elem_classes="secondary-btn"
                    )

            with gr.Row():
                random_seed_button = gr.Button(
                    " Random Seed",
                    elem_classes="secondary-btn"
                    )
                clear_button = gr.Button(
                    "Clear",
                    elem_classes="secondary-btn"
                    )

        # ==================================
        # RIGHT SIDE
        # ==================================

        with gr.Column(scale=1,elem_classes="panel"):

            gr.Markdown("## Generated Image")

            output_image = gr.Image(
                label="Generated Image",
                type="filepath",
                height=520,
                elem_classes="output-image"
            )

            status = gr.Markdown(
                "Ready to generate.",
                elem_classes="status-box"
            )

    # ==================================
    # EXAMPLES
    # ==================================

    gr.Markdown("##  Try an example")

    gr.Examples(
        examples=[
            ["A futuristic city at sunset"],
            ["A cozy cyberpunk bedroom overlooking a neon city"],
            ["A magical forest with glowing flowers at night"],
            ["A tiny astronaut exploring an alien planet"],
            ["A beautiful ancient castle on a mountain"],
        ],
        inputs=prompt
    )

    # ==================================
    # HISTORY
    # ==================================

    gr.Markdown("---")

    gr.Markdown(
        """
        ## Generation History

        **Click an image to load its settings.**
        """
    )

    history_gallery = gr.Gallery(
        label="Previous Generations",
        columns=5,
        rows=2,
        height="auto",
        object_fit="cover",
        value=get_gallery_images(),
        elem_classes="gallery"
    )

    # ==================================
    # GENERATE EVENT
    # ==================================

    generate_button.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            steps,
            guidance,
            resolution,
            seed
        ],
        outputs=[
            output_image,
            status,
            history_gallery
        ]
    )

    #===============================
    # REGENERATE EVENT
    #===========================
    regenerate_button.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            steps,
            guidance,
            resolution,
            seed
        ],
        outputs=[
            output_image,
            status,
            history_gallery
        ]
    )



    # ==================================
    # GALLERY SELECT EVENT
    # ==================================

    history_gallery.select(
        fn=select_generation,
        inputs=None,
        outputs=[
            output_image,
            prompt,
            negative_prompt,
            steps,
            guidance,
            resolution,
            seed,
            status
        ]
    )

    # ==================================
    # CLEAR EVENT
    # ==================================

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            prompt,
            negative_prompt,
            steps,
            guidance,
            resolution,
            seed,
            output_image,
            status
        ]
    )

    random_seed_button.click(
    fn=random_seed,
    inputs=None,
    outputs=seed
)


# ==========================================
# LAUNCH
# ==========================================

app.launch()