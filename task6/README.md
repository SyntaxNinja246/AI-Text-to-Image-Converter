# PixelDream AI - Task 6: Unified Multi-Modal Pipeline

## 1. Overview and Objective
Task 6 integrates all previous components of the PixelDream AI project into a cohesive, production-grade multi-modal AI generation and analysis platform. It demonstrates how:
1. **Fine-Tuned Diffusion Models (Task 1)** provide high-fidelity text-to-image generation.
2. **Category-Conditioned GANs (Task 2)** offer fast class-specific 64×64 synthesis.
3. **Transformer Preprocessing & CLIP Embeddings (Task 3)** manage linguistic representations and sequence tokenization.
4. **Dataset Exploration & Metrics (Task 4)** anchor the pipeline with COCO distribution statistics.
5. **Cross-Attention Spatial Attribution (Task 5)** visually explains token influence across the diffusion UNet.

---

## 2. Model Lineage: Why V2 is the Final Production Model
A critical finding of Task 1 research is the empirical performance of LoRA adaptation on `segmind/tiny-sd`:

| Model Variant | Training Samples | Global Steps | LoRA Rank | Edge Sharpness (Laplacian Var) | Contrast (StdDev) | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LoRA V2** | **500** | **250** | **4** | **418.25** | **59.42** | **FINAL / DEFAULT PRODUCTION MODEL** |
| LoRA V3 | 500 | 250 | 8 | ~380.10 | 58.12 | Experimental Only (tested rank capacity limits) |
| LoRA V4 | 1,500 | 1,125 | 4 | 328.41 | 60.78 | Experimental Only (over-adapted at 1,125 steps) |

### Key Reasons V2 is the Default:
- **Optimal Regularization**: 250 global steps prevented catastrophic cross-attention drift on Tiny-SD's 323M parameter UNet.
- **Superior Visual Quality**: Delivers the highest structural definition and sharpest edge details (418.25 Laplacian variance vs. 328.41 for V4).
- **Scale Stability**: At scale **0.75**, V2 produces clean geometry without color bleeding or high-frequency chromatic noise.
- **Strictly Verified**: All 144 LoRA tensors are dynamically mapped and verified upon initialization.

---

## 3. Architecture & Task Integration

```
[ User Input / Web Interface ]
              │
              ├──► 1. Text-to-Image Generation (Task 1)
              │         ├── CLIP Tokenizer & Text Encoder
              │         ├── Segmind/Tiny-SD UNet (Float32, CPU)
              │         ├── Verified V2 LoRA Adapter (144 Tensors, Rank 4, Alpha 4)
              │         └── Dynamic LoRA Scaling (0.0 to 1.5, Default: 0.75)
              │
              ├──► 2. Conditional GAN (Task 2)
              │         ├── Noise Vector z ~ N(0, I_100)
              │         ├── Class Embedding (5 Classes: car, chair, bottle, bird, boat)
              │         └── Generator CNN (Output: 64×64 RGB)
              │
              ├──► 3. Transformer & Prompt Tokenization (Task 3)
              │         ├── Live CLIP Tokenizer Breakdown (IDs, Position, Padding)
              │         └── 2.96 GB Dense Embedding Array Metadata [25014, 77, 768]
              │
              ├──► 4. Dataset Exploration (Task 4)
              │         ├── 5,000 COCO 2017 Validation Images
              │         ├── 25,014 Preprocessed Captions across 80 Categories
              │         └── Distribution Visualizations (Aspect Ratios, Category Counts)
              │
              └──► 5. Cross-Attention Spatial Attribution (Task 5)
                        ├── 18 Attention Processors (9 Self, 9 Cross-Attention)
                        ├── 8 Attention Heads, 32×32 Spatial Resolution (1024 cells)
                        └── Token Attribution Heatmaps ("red", "car", "blue", "house")
```

---

## 4. How to Launch Task 6

To launch the unified Task 6 Gradio application, execute the following command in PowerShell from the project root:

```powershell
.\venv\Scripts\python.exe task6\app.py
```

Access the interface in your browser at:
```
http://127.0.0.1:7860
```

---

## 5. CPU Limitations & Operational Guidelines
- **Device & Precision**: All models run on **CPU** using **float32** precision.
- **Inference Latency**:
  - Tiny-SD (10 steps, 256×256): ~15–20 seconds per generation.
  - CGAN (64×64): < 0.05 seconds per generation.
- **Memory Optimization**: The diffusion pipeline and CGAN generator remain resident in memory to avoid reload overhead across consecutive generations.
- **Recommended Denoising Steps**: 10 steps provides the optimal trade-off between generation speed and visual fidelity on CPU.

---

## 6. Example Prompts
- `"a cozy room with chairs"` (Seed: 102, LoRA scale: 0.75)
- `"a motorcycle parked outdoors"` (Seed: 101, LoRA scale: 0.75)
- `"a person standing in a kitchen"` (Seed: 103, LoRA scale: 0.75)
- `"a desk with electronic equipment"` (Seed: 104, LoRA scale: 0.75)
- `"a group of people outdoors"` (Seed: 105, LoRA scale: 0.75)

---

## 7. Verification & Quality Assurance
The pipeline is verified using `task6/validate_task6.py`:
- **Tensor Count**: 144 saved, 144 mapped, 144 loaded, 0 missing, 0 unexpected.
- **Dynamic Scaling**: Scale adjustments applied without re-instantiating UNet.
- **End-to-End Diffusion**: Verified image synthesis and history update.
- **CGAN Compatibility**: Category conditioning successfully validated.
- **Historical Consistency**: `generation_history.json` preserved and extended.
- **Root App Protection**: Original `app.py` and `generator.py` remain intact and functional.
