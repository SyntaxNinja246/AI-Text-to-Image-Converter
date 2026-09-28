from generator import ImageGenerator


def get_integer(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return default

    try:
        return int(value)
    except ValueError:
        print(f"Invalid number. Using default: {default}")
        return default


def get_float(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return default

    try:
        return float(value)
    except ValueError:
        print(f"Invalid number. Using default: {default}")
        return default


print("=" * 45)
print("       TEXT-TO-IMAGE GENERATOR")
print("=" * 45)

prompt = input("\nEnter your prompt: ").strip()

if not prompt:
    print("You must enter a prompt.")
    exit()

negative_prompt = input(
    "Negative prompt [blurry, distorted, low quality]: "
).strip()

if not negative_prompt:
    negative_prompt = "blurry, distorted, low quality"

steps = get_integer(
    "Inference steps",
    10
)

guidance = get_float(
    "Guidance scale",
    7.5
)

seed = get_integer(
    "Seed",
    42
)

print("\nLoading generator...\n")

generator = ImageGenerator()

output = generator.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=steps,
    guidance_scale=guidance,
    width=256,
    height=256,
    seed=seed
)

print("\n" + "=" * 45)
print("GENERATION COMPLETE!")
print("=" * 45)
print(f"Image: {output}")