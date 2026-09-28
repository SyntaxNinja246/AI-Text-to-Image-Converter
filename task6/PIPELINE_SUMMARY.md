# PixelDream AI: Pipeline Architecture & Technical Summary

## 1. ASCII Pipeline Architecture Diagram

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

## 2. Technical Summary for Internship Report

### A. Architectural Synergy
The PixelDream AI system represents a full-lifecycle text-to-image pipeline:
1. **Linguistic Representation (Task 3)**: High-dimensional semantic representation using a pretrained CLIP ViT-L/14 text encoder producing $[25014, 77, 768]$ embeddings.
2. **Generative Modeling Approaches**:
   - **Diffusion Synthesis (Task 1)**: Latent diffusion via `segmind/tiny-sd` with low-rank adaptation for complex multi-subject prompts.
   - **Adversarial Synthesis (Task 2)**: Class-conditional generative adversarial networks producing fast $64 \times 64$ class instances.
3. **Interpretability & Explainability (Task 5)**: Cross-attention interception using hook-based recorders to extract attention probability matrices, quantifying spatial attribution per prompt token.
4. **Empirical Validation (Tasks 1, 4, 6)**: Rigorous comparison across iterations (V1–V4), identifying **V2** (250 steps, rank 4, optimal Laplacian sharpness: `418.25`) as the final production model, while demonstrating the risks of over-training (V4 at 1,125 steps).

### B. Deployment Specifications
- **Framework**: PyTorch 2.x + Diffusers + PEFT + Gradio 6.x
- **Hardware Architecture**: Pure CPU (Float32 precision, multithreaded)
- **Production Adapter**: `finetune/models/tiny-sd-lora-v2` (144 verified tensors)
- **Launch Command**: `.\venv\Scripts\python.exe task6\app.py`
