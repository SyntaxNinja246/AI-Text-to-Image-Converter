import os
import time
import torch
from diffusers import DiffusionPipeline


class ImageGenerator:

    def __init__(self):
        print("Loading Stable Diffusion model...")

        # Your computer is currently using CPU
        self.device = "cpu"
        cpu_threads = os.cpu_count() or 1
        torch.set_num_threads(cpu_threads)
        print(f"CPU threads: {cpu_threads}")

        self.model_id = "segmind/tiny-sd"

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32
        )

        self.pipe = self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        # Store generated images here
        self.output_folder = "generated_images"
        os.makedirs(self.output_folder, exist_ok=True)

        print(f"Model: {self.model_id}")
        print(f"Device: {self.device}")
        print("Image Generator ready!")

    def generate(
        self,
        prompt,
        negative_prompt="",
        num_inference_steps=10,
        guidance_scale=7.5,
        width=256,
        height=256,
        seed=None
    ):

        print("\n------------------------------")
        print("Starting image generation")

        # ==========================================
        # INPUT VALIDATION
        # ==========================================

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        prompt = prompt.strip()
        negative_prompt = negative_prompt.strip()

        num_inference_steps = int(num_inference_steps)
        guidance_scale = float(guidance_scale)
        width = int(width)
        height = int(height)

        if num_inference_steps < 1:
            raise ValueError("Inference steps must be at least 1.")

        if guidance_scale < 0:
            raise ValueError("Guidance scale cannot be negative.")

        if width < 64 or height < 64:
            raise ValueError("Image dimensions must be at least 64x64.")

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError("Image dimensions must be divisible by 8")

        if width*height > 512 * 512:
            raise ValueError("Maximum supported Resolution is 512x512")

        # ==========================================
        # DISPLAY SETTINGS
        # ==========================================

        print(f"Prompt: {prompt}")

        if negative_prompt:
            print(f"Negative prompt: {negative_prompt}")

        print(f"Steps: {num_inference_steps}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Size: {width}x{height}")

        # ==========================================
        # SEED
        # ==========================================

        if seed is None:
            seed = torch.randint(
                0,
                2**32 - 1,
                (1,)
            ).item()

        seed = int(seed)

        print(f"Seed: {seed}")

        # ==========================================
        # REPRODUCIBLE RANDOM GENERATOR
        # ==========================================

        generator = torch.Generator(
            device=self.device
        ).manual_seed(seed)

        # ==========================================
        # GENERATION
        # ==========================================

        start_time = time.time()

        with torch.inference_mode():

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )

        image = result.images[0]
        del result

        # ==========================================
        # SAVE IMAGE
        # ==========================================

        timestamp = int(time.time() * 1000)

        filename = (
            f"image_{timestamp}_{seed}.png"
        )

        output_path = os.path.join(
            self.output_folder,
            filename
        )

        image.save(output_path)

        # ==========================================
        # GENERATION TIME
        # ==========================================

        generation_time = time.time() - start_time

        print(f"Image saved to: {output_path}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print("------------------------------")

        return output_path