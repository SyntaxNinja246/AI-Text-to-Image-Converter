import torch
from transformers import CLIPTokenizer


MODEL_ID = "openai/clip-vit-base-patch32"


def main():

    prompt = "a red car beside a blue house"

    print("=" * 60)
    print("PIXELDREAM AI - TASK 5")
    print("TEXT TOKEN INSPECTION")
    print("=" * 60)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)

    tokens = tokenizer.tokenize(prompt)
    token_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).input_ids[0]

    print("\nPrompt:")
    print(prompt)

    print("\nTokens:")
    print(tokens)

    print("\nToken positions and IDs:")
    print("-" * 40)

    for position, token_id in enumerate(token_ids):

        token = tokenizer.decode([token_id])

        print(
            f"Position {position:2d} | "
            f"ID {token_id.item():5d} | "
            f"Token: {repr(token)}"
        )

    print("\n" + "=" * 60)
    print("TOKEN INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()