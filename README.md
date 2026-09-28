# PixelDream AI: End-to-End Multi-Modal Text-to-Image Generation & Analysis Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://github.com/huggingface/diffusers)
[![PEFT](https://img.shields.io/badge/HuggingFace-PEFT_LoRA-FF6F00?style=flat&logo=huggingface&logoColor=white)](https://github.com/huggingface/peft)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-F97316?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![Hardware](https://img.shields.io/badge/Execution-Pure_CPU_(Float32)-10B981?style=flat)](#system-requirements--execution-specifications)

---

## 1. Executive Summary & Project Overview

**PixelDream AI** is a comprehensive, production-grade text-to-image generation, fine-tuning, adversarial synthesis, and neural explainability platform. Built on pure CPU architecture using multithreaded **PyTorch** and **Hugging Face Diffusers**, PixelDream AI bridges the gap between deep generative modeling research and real-time interactive deployment.

The project encompasses a multi-phase research lifecycle across six interconnected tasks:
1. **Task 1: Latent Diffusion Fine-Tuning via LoRA**: Parameter-efficient fine-tuning (PEFT) on [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd) (323.8M parameters) across four iterations (V1–V4). Through empirical validation, **LoRA V2** (Rank 4, Alpha 4, 250 steps, Laplacian edge sharpness `418.25`) was established as the final production model, avoiding the over-adaptation observed in V4 (1,125 steps).
2. **Task 2: Category-Conditioned Generative Adversarial Networks (CGAN)**: Fast adversarial synthesis of $64 \times 64$ class-conditioned visual instances across 5 object categories (`car`, `chair`, `bottle`, `bird`, `boat`), featuring both baseline CNN and Self-Attention augmented CGAN generators.
3. **Task 3: Linguistic Representation & Dense Transformer Embeddings**: Prompt ingestion, cleaning, Byte-Pair Encoding (BPE) via CLIP Tokenizer (`ViT-L/14`), and the extraction of a dense $2.96\text{ GB}$ precomputed embedding tensor matrix $[25014, 77, 768]$ for 25,014 COCO captions.
4. **Task 4: COCO Dataset Exploration & Statistical Profiling**: In-depth exploratory data analysis across 5,000 COCO 2017 validation images, 25,014 captions, and 80 semantic categories, generating distribution profiles for aspect ratios, spatial resolutions, and caption token lengths.
5. **Task 5: Cross-Attention Spatial Attribution & Model Explainability**: Hook-based neural interception of all 18 UNet attention processors (9 self-attention, 9 cross-attention blocks) to map the spatial influence of individual conditioning tokens (e.g., `"red"`, `"car"`, `"blue"`, `"house"`) directly onto the $32 \times 32$ latent feature grid.
6. **Task 6: Unified Multi-Modal Pipeline & Dashboard**: A unified, responsive [Gradio](https://gradio.app/) web application integrating all five upstream tasks into a single platform with live prompt tokenization, dynamic LoRA scaling without reloading, history restoration, CGAN inference, and visual analytical galleries.

---

## 2. End-to-End Pipeline Architecture

The following diagram illustrates how user inputs travel through text preprocessing, tokenizer analysis, latent diffusion with verified PEFT LoRA adapters, cross-attention attribution hooks, and final VAE image decoding:

```
========================================================================================================================
                                     PIXELDREAM AI: END-TO-END PIPELINE ARCHITECTURE
========================================================================================================================

                                            +---------------------------+
                                            |     USER TEXT PROMPT      |
                                            | ("a cozy room with chairs")
                                            +-------------+-------------+
                                                          |
                                                          v
                                            +---------------------------+
                                            |    PROMPT PREPROCESSING   |
                                            |  (Validation & Cleaning)  |
                                            +-------------+-------------+
                                                          |
                 +----------------------------------------+---------------------------------------+
                 |                                                                                |
                 v                                                                                v
  +-------------------------------+                                                +-------------------------------+
  |       TASK 3 INTEGRATION      |                                                |       TASK 1 INTEGRATION      |
  |  CLIP Tokenizer (ViT-L/14)    |                                                |  Segmind/Tiny-SD Base Model   |
  |  - Active Tokens: 7 / 77      |                                                |  - Parameters: 323,816,580    |
  |  - Token IDs & Attention Mask |                                                |  - Device: CPU (Float32)      |
  +---------------+---------------+                                                +---------------+---------------+
                  |                                                                                |
                  | (Text Embeddings 77x768)                                                       | (Base UNet)
                  +-----------------------------------+--------------------------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |    V2 LoRA ADAPTER ATTACH |
                                        |  - Rank: 4, Alpha: 4      |
                                        |  - Target: q, k, v, out.0 |
                                        |  - 144 / 144 Verified     |
                                        |  - Dynamic Scale: 0.75    |
                                        +-------------+-------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |  DIFFUSION DENOISING LOOP |
                                        |  - Steps: 10 (CPU)        |
                                        |  - Guidance Scale: 7.5    |
                                        |  - Latents: 32x32x4       |
                                        +-------------+-------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
          +-------------------------------+                         +-------------------------------+
          |      TASK 5 ATTRIBUTION       |                         |         VAE DECODER           |
          |  Cross-Attention Heatmaps     |                         |  Latents (32x32) -> 256x256   |
          |  - 18 Attention Processors    |                         +---------------+---------------+
          |  - Spatial Token Attribution  |                                         |
          |  - Overall Attention Map      |                                         v
          +-------------------------------+                         +-------------------------------+
                                                                    |     GENERATED OUTPUT IMAGE    |
                                                                    |  (256x256 RGB, Clean Details) |
                                                                    +---------------+---------------+
                                                                                    |
                                                                                    v
                                                                    +-------------------------------+
                                                                    |  HISTORY & METADATA RECORDER  |
                                                                    |  - generation_history.json    |
                                                                    |  - Execution Time & Seed Log  |
                                                                    +-------------------------------+

========================================================================================================================
                                     SUPPORTING MULTI-MODAL PIPELINE COMPONENTS
========================================================================================================================

  +--------------------------------------------+            +--------------------------------------------+
  |        TASK 2: CONDITIONAL GAN (CGAN)      |            |         TASK 4: DATASET EXPLORATION        |
  |                                            |            |                                            |
  |  [Noise z (100-d)] + [Category (5 classes)]|            |  - Dataset: COCO 2017 Validation Set       |
  |                    |                       |            |  - Images Analyzed: 5,000                  |
  |                    v                       |            |  - Captions Processed: 25,014              |
  |         [Custom Generator CNN]             |            |  - Categories: 80 COCO Classes             |
  |                    |                       |            |  - Artifacts: 5 Distribution Visualizations|
  |                    v                       |            +--------------------------------------------+
  |          [64x64 Synthesized Object]        |
  +--------------------------------------------+
========================================================================================================================
```

---

## 3. How the Application Runs

PixelDream AI provides two entry points depending on operational requirements:
1. **Unified Multi-Modal Application** ([`task6/app.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/task6/app.py)): The full production platform featuring 5 interactive tabs covering diffusion generation, CGAN sampling, attention interpretability, dataset exploration, and architectural lineage.
2. **Root Diffusion Application** ([`app.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/app.py)) & **CLI Generator** ([`test_model.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/test_model.py)): Lightweight generation focused solely on diffusion inference and prompt history.

### Quick Start Guide

#### Prerequisites
- Windows OS, Linux, or macOS.
- Python 3.10, 3.11, or 3.12.
- 8 GB+ System RAM (16 GB recommended for CPU generation).

#### Step 1: Clone or Navigate to the Workspace
```powershell
cd "c:\Users\Aayat Nizam\Documents\text-to-image-generator"
```

#### Step 2: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

#### Step 3: Launch the Production Application (Task 6)
To run the full unified multi-modal platform:
```powershell
.\venv\Scripts\python.exe task6\app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

#### Step 4: (Alternative) Launch the Root Application
To run the root generator interface:
```powershell
.\venv\Scripts\python.exe app.py
```

#### Step 5: (Alternative) Run Direct CLI Inference
```powershell
.\venv\Scripts\python.exe test_model.py
```

---

## 4. Model Deep Dive & Generative Frameworks

PixelDream AI leverages two distinct generative paradigms: **Latent Diffusion** and **Adversarial Synthesis**.

### A. Latent Diffusion: `segmind/tiny-sd`
The primary text-to-image synthesis backbone is **Tiny-SD**, a distilled, parameter-efficient latent diffusion architecture:
- **Total Parameters**: 323,816,580 (~323.8M).
- **Text Encoder**: CLIP ViT-L/14 ($77 \times 768$ embedding space).
- **Autoencoder**: VAE downsampling by factor $8\times$ (input $256 \times 256 \times 3 \rightarrow$ latent $32 \times 32 \times 4$).
- **UNet Backing**: 18 total attention processors across 3 down-blocks, 1 mid-block, and 3 up-blocks.

```
                    +------------------------------------+
                    |        CLIP Text Encoder           |
                    | (ViT-L/14, 77 tokens -> 768-d)     |
                    +-----------------+------------------+
                                      | Text Conditioning
                                      v
+------------------+    +----------------------------+    +------------------+
| Latent Noise z_t | -> |         UNet 2D            | -> | Predicted Noise  |
|   (32x32x4)      |    | + PEFT LoRA (q, k, v, out) |    |      eps_theta   |
+------------------+    +----------------------------+    +------------------+
                                      | 10 Denoising Steps
                                      v
                        +----------------------------+
                        |      VAE Decoder (8x)      |
                        +--------------+-------------+
                                       |
                                       v
                        +----------------------------+
                        |   Generated Image (256x256)|
                        +----------------------------+
```

### B. Parameter-Efficient LoRA Fine-Tuning
Low-Rank Adaptation freezes the 323.8M base UNet weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where:
- $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ is initialized to zero, ensuring $\Delta W = 0$ at step 0.
- $r$ is the LoRA rank ($r = 4$ for V2).
- $\alpha$ is the scaling hyperparameter ($\alpha = 4$).
- Target modules: Attention projections `to_q`, `to_k`, `to_v`, and `to_out.0` across all UNet attention blocks.

#### The 144-Tensor PEFT Key Mapping Verification
During Task 1 development, a key naming divergence occurred between diffusers checkpoint storage and PEFT runtime state dictionaries:
- **Safetensors Storage**: `down_blocks.0.attentions.0...to_q.lora_A.weight`
- **PEFT Runtime**: `base_model.model.down_blocks.0.attentions.0...to_q.lora_A.default.weight`

[`task6/pipeline.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/task6/pipeline.py) implements automated state dictionary transformation:
```python
peft_state = {}
for key, value in saved_state.items():
    new_key = "base_model.model." + key
    new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
    new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
    peft_state[new_key] = value

pipeline.unet.load_state_dict(peft_state, strict=False)
```
Upon startup, the pipeline executes strict verification:
- **Saved LoRA tensors**: 144
- **Mapped LoRA tensors**: 144
- **Loaded LoRA keys**: 144
- **Missing keys**: 0
- **Unexpected keys**: 0

### C. LoRA Model Lineage & The Empirical Decision for V2

The development of the diffusion engine underwent four iterative cycles (V1 through V4). Rigorous quantitative and visual comparisons established **LoRA V2** as the final production model:

| Model Variant | Training Samples | Epochs | Global Steps | LoRA Rank ($r$) | LoRA Alpha ($\alpha$) | Trainable Parameters | Edge Sharpness (Laplacian Var) | Contrast (Pixel StdDev) | Final Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tiny-SD Base** | — | — | — | — | — | 0 (Frozen) | ~295.40 | 54.12 | — | Unadapted Baseline |
| **LoRA V1** | 50 | 1 | 25 | 4 | 4 | 431,616 (0.133%) | — | — | 0.824 | Smoke Test Only |
| **LoRA V2** | **500** | **2** | **250** | **4** | **4** | **431,616 (0.133%)** | **418.25** | **59.42** | **0.1142** | **FINAL PRODUCTION MODEL** |
| **LoRA V3** | 500 | 4 | 500 | 8 | 8 | 863,232 (0.266%) | ~380.10 | 58.12 | 0.1145 | Experimental (Capacity Limit) |
| **LoRA V4** | 1,500 | 3 | 1,125 | 4 | 4 | 431,616 (0.133%) | 328.41 | 60.78 | 0.0952 | Experimental (Over-Adapted) |

#### Why V2 Outperforms V3 and V4:
1. **Optimal Structural Regularization**: At 250 global steps, V2 balanced semantic adaptation with base model generalization. Higher step counts (V4 at 1,125 steps, ~4.08 hours of training) caused over-adaptation where the model generated unnatural color patches and reduced edge definition (Laplacian variance dropped from `418.25` in V2 down to `328.41` in V4).
2. **Rank Capacity Balance**: Doubling rank to $r=8$ in V3 provided no qualitative improvement while doubling parameter overhead and introducing cross-attention instability.
3. **Dynamic Scaling Behavior**: At scale **0.75**, V2 yields clean object contours and vivid contrast without high-frequency chromatic aberrations.

---

### D. Category-Conditioned GAN (Task 2 CGAN)

While diffusion excels at complex scene composition, the custom **Conditional GAN (CGAN)** provides near-instantaneous ($< 0.05\text{ s}$) generation for isolated objects across 5 COCO classes:
- **Conditioning Categories**: `car`, `chair`, `bottle`, `bird`, `boat`.
- **Input**: Latent noise $z \sim \mathcal{N}(0, I_{100})$ concatenated with a learned category embedding vector.
- **Generator CNN**: 4-stage transposed convolutional network projecting $100\text{-d} \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 64 \times 64 \times 3$ with BatchNorm, ReLU activations, and Tanh output.
- **Discriminator CNN**: Convolutional discriminator utilizing LeakyReLU and minibatch discrimination / Spectral Normalization to prevent mode collapse.
- **Self-Attention CGAN Variant** ([`cgan/attention_generator.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/cgan/attention_generator.py)): Injects a non-local self-attention block at spatial resolution $32 \times 32$ to capture long-range geometric dependencies across object silhouettes.

---

## 5. Review of Graphs, Plots, and Visual Artifacts

PixelDream AI includes an extensive suite of empirical visualizations across all tasks. Below is a detailed breakdown of each generated plot, evaluation matrix, and interpretability map.

### A. Dataset Exploration & Statistical Profiles (Task 4)
Located in [`dataset/analysis_results/`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/dataset/analysis_results/).

1. **`category_distribution.png`**:
   - **Content**: Horizontal/vertical bar chart showing annotation counts across the top 20 COCO object categories.
   - **Key Finding**: Demonstrates class imbalance in COCO 2017 validation data. `person` is the dominant category (>10,000 instances), followed by vehicles (`car`), furniture (`chair`), and everyday items. This guided the selection of balanced classes for Task 2 CGAN (`car`, `chair`, `bottle`, `bird`, `boat`).
2. **`caption_length_distribution.png`**:
   - **Content**: Histogram depicting the distribution of caption word counts across all 25,014 preprocessed captions.
   - **Key Finding**: Captions exhibit a unimodal distribution centered around a **mean of 10.45 words** and a **median of 10 words** (range: 5 to 48 words). This confirmed that CLIP's 77-token maximum sequence length comfortably accommodates $>99.9\%$ of captions without semantic truncation.
3. **`image_width_distribution.png` & `image_height_distribution.png`**:
   - **Content**: Separate histograms showing the spatial dimensions of COCO images.
   - **Key Finding**: Image widths peak around $640\text{ px}$ and heights around $480\text{ px}$ (standard $4:3$ aspect ratio). Validated the necessity of center-cropping and resizing to $256 \times 256$ for diffusion fine-tuning and $64 \times 64$ for CGAN training.
4. **`sample_images_with_captions.png`**:
   - **Content**: Qualitative $3 \times 3$ grid displaying randomly sampled COCO images alongside their ground-truth annotations, illustrating scene complexity, multi-object co-occurrences, and linguistic diversity.

---

### B. Diffusion Model Comparison Visualizations (Task 1)
Located in [`finetune/outputs/model_comparison_v2_v4/`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/finetune/outputs/model_comparison_v2_v4/).

1. **`contact_sheet.png`**:
   - **Content**: High-resolution $4 \times 5$ comparative contact sheet evaluating four model variants across five fixed benchmark prompts and seeds:
     - **Row 1**: V2 @ LoRA Scale 0.75 (Indigo badge)
     - **Row 2**: V4 @ LoRA Scale 0.50 (Cyan badge)
     - **Row 3**: V4 @ LoRA Scale 0.75 (Blue badge)
     - **Row 4**: V4 @ LoRA Scale 1.00 (Purple badge)
     - **Columns (Seeds 101–105)**:
       - Col 1: `"a motorcycle parked outdoors"` (Seed 101)
       - Col 2: `"a cozy room with chairs"` (Seed 102)
       - Col 3: `"a person standing in a kitchen"` (Seed 103)
       - Col 4: `"a desk with electronic equipment"` (Seed 104)
       - Col 5: `"a group of people outdoors"` (Seed 105)
   - **Key Finding**:
     - **V2 @ 0.75** maintains crisp object boundaries, realistic chair geometry in the cozy room, and distinct kitchen appliances.
     - **V4 @ 0.50 to 1.00** shows texture oversaturation, muddied facial features in the people prompt, and blurred electronic outlines on the desk, providing clear visual evidence of over-adaptation.
2. **`finetune/outputs/model_comparison/` (Base vs. V2 vs. V3)**:
   - Demonstrates that Base Tiny-SD produces washed-out images with weak prompt alignment.
   - V2 adds sharp focal clarity and contrast.
   - V3 introduces structural warping around high-frequency edges due to excess rank capacity ($r=8$).

---

### C. Cross-Attention Spatial Attribution Maps (Task 5)
Located in [`attention/outputs/`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/attention/outputs/).

Cross-attention layers compute affinity matrices between spatial visual tokens $Q = W_q z \in \mathbb{R}^{1024 \times d}$ and textual tokens $K = W_k c \in \mathbb{R}^{77 \times d}$:
$$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1024 \times 77}$$
Interpreted benchmark prompt: `"a red car beside a blue house"`:

1. **`attention_test_image.png`**:
   - The reference synthetic image generated by the pipeline containing a red vehicle parked beside a blue architectural structure.
2. **`attention_red.png` (Token Pos 2)**:
   - Heatmap shows concentrated activation peaks over the automotive bodywork, verifying that the adjective `"red"` specifically grounds on the vehicle pixels.
3. **`attention_car.png` (Token Pos 3)**:
   - Shows broad spatial activation encompassing the entire vehicle silhouette, wheels, and ground shadow.
4. **`attention_blue.png` (Token Pos 6)**:
   - Heatmap shifts completely away from the vehicle toward the building facade and sky region.
5. **`attention_house.png` (Token Pos 7)**:
   - Spatial activation isolates the architectural roofline, walls, and windows of the house.
6. **`overall_attention_map.png`**:
   - Summed spatial intensity showing how text conditioning guides visual composition across the entire $32 \times 32$ grid.
7. **`word_level_attention.png`**:
   - Comprehensive multi-panel figure displaying the generated test image side-by-side with colorbar-annotated attention heatmaps for all key lexical items.

---

### D. CGAN Adversarial Synthesis Visualizations (Task 2)
Located in [`cgan/samples/final_cgan/`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/cgan/samples/final_cgan/) and [`cgan/outputs/attention_comparison/`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/cgan/outputs/attention_comparison/).

1. **`cgan/samples/final_cgan/evaluation.png`**:
   - **Content**: A $5 \times 10$ evaluation grid containing 50 synthesized $64 \times 64$ images (10 random seeds $\times$ 5 object classes: car, chair, bottle, bird, boat).
   - **Key Finding**: Demonstrates consistent class conditioning. Cars display wheeled horizontal structures; chairs show vertical legs and backs; bottles display cylindrical symmetry; birds and boats show recognizable color signatures and silhouettes.
2. **`cgan/outputs/attention_comparison/baseline_vs_attention.png`**:
   - Side-by-side grid comparing baseline CGAN vs. Self-Attention CGAN.
   - Shows that Self-Attention CGAN improves global geometric coherence (e.g., straight chair legs and symmetric car bodies) over standard local convolutional filtering.
3. **Epoch Progression Grids (`epoch_1.png` to `epoch_75.png`)**:
   - Preserves the adversarial training history across 75 epochs, illustrating the progression from random chromatic noise (Epochs 1–5) to coarse object blobs (Epochs 15–30), and finally to sharp object instances (Epochs 50–75).

---

## 6. Interactive Web Application Walkthrough (Task 6)

The Task 6 dashboard ([`task6/app.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/task6/app.py)) integrates all modules into a dark-mode interface with five specialized tabs:

```
+---------------------------------------------------------------------------------------------------+
|                                        PIXELDREAM AI                                              |
|      Task 6 — Text-to-Image Pipeline: Prompt -> Text Preprocessing -> Embedding -> Generation      |
+---------------------------------------------------------------------------------------------------+
| [ Generate ] | [ Conditional GAN ] | [ Attention Analysis ] | [ Dataset & Embeddings ] | [ About ] |
+---------------------------------------------------------------------------------------------------+
```

### Tab 1: Generate (Diffusion + LoRA V2)
- **Prompt Input**: Text field supporting complex multi-subject prompts up to 77 tokens.
- **Preset Quick-Fill Buttons**: One-click benchmark prompts (`Cozy Room`, `Motorcycle`, `Kitchen`, `Desk`, `People`).
- **Interactive Controls**:
  - **Steps Slider** (1–50, default: 10): Optimal CPU speed/quality trade-off.
  - **Guidance Scale Slider** (1.0–20.0, default: 7.5): Controls prompt alignment strength.
  - **LoRA Scale Slider** (0.0–1.5, default: 0.75): Dynamically scales adapter weights on the fly without UNet re-instantiation.
  - **Seed Input & Randomizer (🎲)**: Enables exact reproducibility or random sampling.
- **Advanced Options**: Negative prompt input (`blurry, distorted, low quality`) and resolution selector (`256x256`, `128x128`, `512x512`).
- **Live Output Panel**: Displays the synthesized image along with inference metadata (latency in seconds, model name, active LoRA scale, seed, output file).
- **Recent History Gallery**: Visual carousel of past generations. **Clicking any image restores its exact prompt, seed, guidance, and LoRA scale**.

### Tab 2: Conditional GAN (Task 2)
- **Category Selector**: Dropdown to select from the 5 trained classes (`car`, `chair`, `bottle`, `bird`, `boat`).
- **Noise Seed Input & Randomizer**: Controls the 100-dimensional Gaussian noise vector $z$.
- **Instant Generation**: Synthesizes and renders a $64 \times 64$ sample in $< 0.05\text{ s}$.
- **Precomputed Evaluation Grid**: Displays the full 50-image evaluation matrix.

### Tab 3: Attention Analysis (Task 5)
- **Live CLIP Token Inspector**: Enter any custom prompt to view its BPE tokenization breakdown, sequence positions, token IDs, active token count, and padding length.
- **Precomputed Cross-Attention Explorer**: Radio selector to inspect spatial heatmaps for `"red"`, `"car"`, `"blue"`, `"house"`, and overall cross-attention density.
- **Architecture Metric Cards**: Summarizes attention configuration (18 processors, 9 cross-attention blocks, 8 heads, $32 \times 32$ spatial feature cells).

### Tab 4: Dataset & Embeddings (Tasks 3 & 4)
- **Global Metrics Banner**: Displays COCO stats (5,000 images, 25,014 captions, 80 categories, 2.96 GB embedding store).
- **Transformer Embeddings Panel**: Documents the CLIP ViT-L/14 text encoder properties, embedding shape $[25014, 77, 768]$, float16 storage, and sequence layout.
- **Distribution Plot Selector**: Dropdown to view COCO analytical plots (category counts, caption length distribution, image widths, image heights, sample image-caption pairs).

### Tab 5: About & Architecture
- **Model Lineage Table**: Comprehensive breakdown of Base, V2, V3, and V4 models.
- **Verification Audit**: Explains the 144-tensor mapping and runtime safety guarantees.
- **Hardware Profile**: Pure CPU multithreaded operation and latency benchmarks.

---

## 7. Directory Structure & Codebase Navigation

```
text-to-image-generator/
│
├── README.md                                  # Comprehensive Project Documentation (This File)
├── app.py                                     # Root Gradio Web Application
├── generator.py                               # Core Diffusion Pipeline Wrapper (ImageGenerator)
├── test_model.py                              # Interactive CLI Generation Script
├── output.png                                 # Root Quick-Test Output Image
├── generation_history.json                    # Global Persistent Generation History (57 entries)
│
├── task6/                                     # TASK 6: UNIFIED MULTI-MODAL PIPELINE
│   ├── app.py                                 # Production 5-Tab Gradio Application
│   ├── pipeline.py                            # Unified Pipeline Manager (PixelDreamPipeline)
│   ├── config.py                              # Global Hyperparameters, Paths, and Constants
│   ├── utils.py                               # Artifact Loaders & History Management
│   ├── validate_task6.py                      # Automated Verification & Safety Test Suite
│   ├── README.md                              # Task 6 Implementation Summary
│   ├── PIPELINE_SUMMARY.md                    # Technical Report & ASCII Architecture Diagram
│   └── outputs/                               # Task 6 Generated Images & CGAN Samples
│
├── finetune/                                  # TASK 1: DIFFUSION FINE-TUNING (LoRA)
│   ├── models/
│   │   ├── tiny-sd-lora-v2/                   # Production Adapter (Rank 4, Alpha 4, 250 steps)
│   │   │   └── pytorch_lora_weights.safetensors
│   │   ├── tiny-sd-lora-v3/                   # Experimental Adapter (Rank 8, Alpha 8, 500 steps)
│   │   └── tiny-sd-lora-v4/                   # Experimental Adapter (Rank 4, 1,125 steps)
│   ├── scripts/
│   │   ├── train_v2.py                        # V2 Training Script (500 samples, 2 epochs)
│   │   ├── train_v3.py                        # V3 Training Script (Rank 8 capacity test)
│   │   ├── train_v4.py                        # V4 Training Script (1,500 samples, 3 epochs)
│   │   ├── evaluate_lora_v2.py                # V2 Evaluation & Tensor Verification
│   │   ├── compare_base_v2_v3.py              # Base vs V2 vs V3 Evaluation Script
│   │   ├── compare_v2_v4.py                   # V2 vs V4 Comparative Benchmark
│   │   └── create_contact_sheet.py            # High-Res 4x5 Contact Sheet Generator
│   └── outputs/
│       ├── model_comparison_v2_v4/            # V2 vs V4 Benchmark Outputs & Contact Sheet
│       │   └── contact_sheet.png              # 4x5 Comparative Contact Sheet Image
│       ├── v2/                                # V2 Training Logs & Configuration JSON
│       ├── v3/                                # V3 Training Logs & Configuration JSON
│       └── v4/                                # V4 Training Logs & Configuration JSON
│
├── cgan/                                      # TASK 2: CONDITIONAL GAN
│   ├── config.py                              # CGAN Hyperparameters & Categories
│   ├── generator.py                           # 4-Stage Transposed Conv Generator CNN
│   ├── attention_generator.py                 # Self-Attention Augmented CGAN Generator
│   ├── discriminator.py                       # Convolutional Discriminator CNN
│   ├── discriminator_hinge.py                 # Hinge Loss Discriminator Variant
│   ├── train.py                               # CGAN Adversarial Training Loop (75 Epochs)
│   ├── evaluate.py                            # 50-Image Evaluation Grid Generator
│   ├── compare_attention.py                   # Baseline vs Attention CGAN Comparison
│   ├── models/
│   │   ├── final_cgan/generator.pth           # Final Production CGAN Checkpoint
│   │   ├── attention_cgan/                    # Self-Attention CGAN Checkpoints
│   │   └── final_4_11_diversity/              # Diversity-Tuned Checkpoints
│   ├── samples/
│   │   ├── final_cgan/evaluation.png          # 5x10 Category Evaluation Grid
│   │   └── final_4_11_diversity/              # Epoch-by-Epoch Progress Images (1-75)
│   └── outputs/
│       └── attention_comparison/              # Baseline vs Attention Comparison Images
│
├── dataset/                                   # TASKS 3 & 4: DATASET & EMBEDDINGS
│   ├── analyze_coco.py                        # COCO Exploratory Data Analysis Script
│   ├── preprocess_captions.py                 # Caption Cleaning & Normalization
│   ├── tokenize_captions.py                   # BPE Tokenization via CLIP Tokenizer
│   ├── generate_embeddings.py                 # Dense CLIP Embedding Extraction
│   ├── annotations/annotations/               # COCO Annotations (captions_val2017.json)
│   ├── images/val2017/                        # 5,000 COCO Validation Images
│   └── analysis_results/
│       ├── category_distribution.png          # Top Category Frequencies Chart
│       ├── caption_length_distribution.png    # Caption Length Histogram
│       ├── image_width_distribution.png       # Width Distribution Chart
│       ├── image_height_distribution.png      # Height Distribution Chart
│       ├── sample_images_with_captions.png    # Qualitative Sample Grid
│       ├── dataset_summary.txt                # Textual Statistical Summary
│       └── embeddings/
│           ├── embedding_metadata.json        # Embedding Shape & Dtype Metadata
│           └── caption_embeddings.npy         # 2.96 GB Dense Embedding Array [25014, 77, 768]
│
├── attention/                                 # TASK 5: CROSS-ATTENTION EXPLAINABILITY
│   ├── visualize_attention.py                 # Word-Level Attention Heatmap Generator
│   ├── token_inspection.py                    # Token Position & ID Inspector
│   ├── attention_analysis.py                  # UNet Attention Processor Interceptor
│   └── outputs/
│       ├── attention_test_image.png           # Benchmark Reference Output Image
│       ├── attention_red.png                  # 'red' Token Spatial Heatmap
│       ├── attention_car.png                  # 'car' Token Spatial Heatmap
│       ├── attention_blue.png                 # 'blue' Token Spatial Heatmap
│       ├── attention_house.png                # 'house' Token Spatial Heatmap
│       ├── overall_attention_map.png          # Summed Cross-Attention Density Map
│       ├── word_level_attention.png           # Multi-Panel Combined Heatmap Figure
│       └── attention_analysis.json            # UNet Attention Processor Breakdown
│
└── generated_images/                          # Global History Image Store (68 images)
```

---

## 8. Quantitative Evaluation & Benchmark Results

### A. Diffusion LoRA Adaptation Metrics

Evaluated across the 5 standard benchmark prompts with fixed seeds (101–105) at 10 denoising steps and guidance scale 7.5:

| Metric | Base Tiny-SD | LoRA V2 (Scale 0.75) | LoRA V4 (Scale 0.75) | LoRA V4 (Scale 1.0) |
| :--- | :---: | :---: | :---: | :---: |
| **Edge Sharpness (Laplacian Var)** | 295.40 | **418.25** | 328.41 | 312.18 |
| **Contrast (StdDev)** | 54.12 | **59.42** | 60.78 | 61.45 |
| **Prompt Alignment (Visual)** | Moderate | **High (Accurate)** | Over-Saturated | Artifact-Prone |
| **Average Step Latency (CPU)** | ~1.65 s | **~1.70 s** | ~1.72 s | ~1.72 s |
| **Total Inference Time (10 steps)** | ~16.8 s | **~17.2 s** | ~17.4 s | ~17.4 s |

### B. CGAN Generation Statistics (64x64 Samples)

Computed across 50 samples generated with seed 42:

| Metric | Baseline CGAN | Self-Attention CGAN |
| :--- | :---: | :---: |
| **Pixel Intensity Mean** | -0.0482 | **-0.0315** |
| **Pixel Intensity StdDev** | 0.5821 | **0.5947** |
| **Inference Latency per Sample** | **0.038 s** | 0.044 s |
| **Structural Coherence Score** | Standard | **Enhanced (Global Attention)** |

---

## 9. System Requirements & Execution Specifications

- **Execution Environment**: Pure **CPU** execution (`torch.device("cpu")`).
- **Precision**: Full **Float32** precision for both base model UNet and LoRA weights to guarantee numerical stability and eliminate underflow on CPU architectures.
- **Multithreading**: Automatically detects and allocates all available CPU cores:
  ```python
  cpu_threads = os.cpu_count() or 1
  torch.set_num_threads(cpu_threads)
  ```
- **Memory Footprint**:
  - Tiny-SD UNet + VAE + Text Encoder: ~1.8 GB RAM.
  - CGAN Generator: ~25 MB RAM.
  - In-Memory Model Caching: Pipelines remain resident in memory after initialization, preventing repetitive ~15-second reload penalties across interactive user requests.

---

## 10. Verification & Quality Assurance

To ensure absolute system integrity, PixelDream AI includes an automated verification script ([`task6/validate_task6.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/task6/validate_task6.py)):

```powershell
.\venv\Scripts\python.exe task6\validate_task6.py
```

### Verification Checklist:
- [x] **144/144 LoRA Tensors Loaded**: 0 missing keys, 0 unexpected keys.
- [x] **Dynamic Scale Adjustment**: Verified scale modifications from 0.0 to 1.5 without UNet reload.
- [x] **End-to-End Diffusion Generation**: Synthesized test image and verified file output.
- [x] **CGAN Conditioning**: Generated valid $64 \times 64$ class-conditioned images across all 5 classes.
- [x] **CLIP Token Inspection**: Verified active token extraction and attention mask handling.
- [x] **History Persistence**: Verified append operations to [`generation_history.json`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/generation_history.json).
- [x] **Root App Backward Compatibility**: Verified that root [`app.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/app.py) and [`generator.py`](file:///c:/Users/Aayat%20Nizam/Documents/text-to-image-generator/generator.py) remain fully functional.

---

## 11. Contributors & Acknowledgments

- **Model Foundations**: [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd) developed by Segmind; [OpenAI CLIP](https://github.com/openai/CLIP) developed by OpenAI.
- **Dataset**: [Microsoft COCO (Common Objects in Context)](https://cocodataset.org/) 2017 Challenge dataset.
- **Frameworks**: [PyTorch](https://pytorch.org/), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), and [Gradio](https://gradio.app/).
