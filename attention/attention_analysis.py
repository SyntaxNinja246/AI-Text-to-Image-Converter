import os
import json
import torch
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0


MODEL_ID = "segmind/tiny-sd"
OUTPUT_DIR = "attention/outputs"


class CrossAttentionRecorder(AttnProcessor2_0):
    """
    Records the attention probability information from
    a cross-attention layer.
    """

    def __init__(self):
        super().__init__()
        self.attention_maps = []

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs
    ):

        # Cross-attention happens when encoder_hidden_states
        # contains the text-conditioning information.
        if encoder_hidden_states is not None:

            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(
                    hidden_states,
                    temb
                )

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape

                hidden_states = hidden_states.view(
                    batch_size,
                    channel,
                    height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(
                    attention_mask,
                    sequence_length,
                    batch_size
                )

                attention_mask = attention_mask.view(
                    batch_size,
                    attn.heads,
                    -1,
                    attention_mask.shape[-1]
                )

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            # Query comes from image features
            query = attn.to_q(hidden_states)

            # Key and value come from text features
            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )

            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            # Reshape into attention heads
            query = query.view(
                batch_size,
                -1,
                attn.heads,
                query.shape[-1] // attn.heads
            ).transpose(1, 2)

            key = key.view(
                batch_size,
                -1,
                attn.heads,
                key.shape[-1] // attn.heads
            ).transpose(1, 2)

            value = value.view(
                batch_size,
                -1,
                attn.heads,
                value.shape[-1] // attn.heads
            ).transpose(1, 2)

            # Calculate attention scores
            attention_scores = torch.matmul(
                query,
                key.transpose(-1, -2)
            )

            attention_scores = attention_scores * (
                query.shape[-1] ** -0.5
            )

            if attention_mask is not None:
                attention_scores = (
                    attention_scores + attention_mask
                )

            # Convert scores into probabilities
            attention_probs = torch.softmax(
                attention_scores,
                dim=-1
            )

            # Store a CPU copy for analysis
            self.attention_maps.append(
                attention_probs.detach().cpu()
            )

            # Continue normal attention computation
            hidden_states = torch.matmul(
                attention_probs,
                value
            )

            hidden_states = hidden_states.transpose(
                1, 2
            ).reshape(
                batch_size,
                -1,
                attn.heads * (
                    query.shape[-1]
                )
            )

            hidden_states = hidden_states.to(
                query.dtype
            )

            hidden_states = attn.to_out[0](
                hidden_states
            )

            hidden_states = attn.to_out[1](
                hidden_states
            )

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(
                    -1, -2
                ).reshape(
                    batch_size,
                    channel,
                    height,
                    width
                )

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

        # Self-attention uses the normal implementation
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs
        )


def main():

    print("=" * 60)
    print("PIXELDREAM AI - TASK 5")
    print("TOKEN-LEVEL CROSS-ATTENTION ANALYSIS")
    print("=" * 60)

    print("\nLoading Tiny-SD...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    print("Model loaded successfully.")

    recorder = CrossAttentionRecorder()

    # Select the first cross-attention layer
    selected_layer = None

    for name in pipe.unet.attn_processors.keys():

        if ".attn2." in name:
            selected_layer = name
            break

    if selected_layer is None:
        raise RuntimeError(
            "No cross-attention layer found."
        )

    print("\nSelected cross-attention layer:")
    print(selected_layer)

    # Replace only one attention processor
    processors = pipe.unet.attn_processors.copy()

    processors[selected_layer] = recorder

    pipe.unet.set_attn_processor(
        processors
    )

    prompt = "a red car beside a blue house"

    print("\nPrompt:")
    print(prompt)

    print("\nGenerating image...")

    with torch.no_grad():

        result = pipe(
            prompt=prompt,
            num_inference_steps=5,
            guidance_scale=7.5,
            height=256,
            width=256
        )

    image = result.images[0]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    image_path = os.path.join(
        OUTPUT_DIR,
        "token_attention_test.png"
    )

    image.save(image_path)

    print("\nImage saved:")
    print(image_path)

    print("\nAttention maps captured:")
    print(len(recorder.attention_maps))

    # Save the actual attention maps
    attention_array = np.stack(
        [
            attention.numpy()
            for attention in recorder.attention_maps
        ]
    )

    attention_map_path = os.path.join(
        OUTPUT_DIR,
        "token_attention_maps.npz"
    )

    np.savez_compressed(
        attention_map_path,
        attention_maps=attention_array
    )

    print("\nActual attention maps saved:")
    print(attention_map_path)

    print("Saved attention array shape:")
    print(attention_array.shape)

    if len(recorder.attention_maps) == 0:
        raise RuntimeError(
            "No attention maps were captured."
        )

    # Examine the first captured attention map
    first_map = recorder.attention_maps[0]

    print("\nFirst attention map shape:")
    print(tuple(first_map.shape))

    # Save basic attention statistics
    attention_statistics = {
        "model": MODEL_ID,
        "prompt": prompt,
        "selected_layer": selected_layer,
        "number_of_attention_maps": len(
            recorder.attention_maps
        ),
        "first_attention_map_shape": list(
            first_map.shape
        ),
        "description": (
            "Cross-attention probabilities between "
            "image features and text conditioning."
        )
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "token_attention_analysis.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            attention_statistics,
            f,
            indent=4
        )

    print("\nAnalysis saved:")
    print(json_path)

    print("\n" + "=" * 60)
    print("TOKEN-LEVEL ATTENTION CAPTURE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()