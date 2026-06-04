import os
from typing import List, Optional

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline


def load_anchor_embeddings(file_path: str, device: str):
    """
    Load trained anchor embeddings and validate them.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find anchor embedding file: {file_path}"
        )

    checkpoint = torch.load(file_path, map_location=device)

    if "anchor_embeds" not in checkpoint:
        raise KeyError(
            "Expected key 'anchor_embeds' not found in checkpoint."
        )

    anchor_embeds = checkpoint["anchor_embeds"].to(device)

    print("\n=== Anchor Embedding Statistics ===")
    print("Shape:", anchor_embeds.shape)
    print("Dtype:", anchor_embeds.dtype)
    print("Min:", anchor_embeds.min().item())
    print("Max:", anchor_embeds.max().item())
    print("Mean:", anchor_embeds.mean().item())
    print("Std:", anchor_embeds.std().item())

    if torch.isnan(anchor_embeds).any():
        raise ValueError("Anchor embeddings contain NaNs.")

    if torch.isinf(anchor_embeds).any():
        raise ValueError("Anchor embeddings contain Infs.")

    return anchor_embeds


def normalize_anchor_embeddings(
    anchor_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
):
    """
    Match anchor embedding scale to target prompt scale.
    Helps prevent black outputs from embedding explosions.
    """

    anchor_embeds = anchor_embeds.float()

    anchor_std = anchor_embeds.std()
    target_std = target_embeds.float().std()

    print("\n=== Embedding Scale Check ===")
    print("Target std:", target_std.item())
    print("Anchor std:", anchor_std.item())

    if anchor_std > 0:
        anchor_embeds = (
            anchor_embeds / anchor_std
        ) * target_std

    anchor_embeds = torch.clamp(anchor_embeds, -10, 10)

    return anchor_embeds.to(target_embeds.dtype)


def create_comparison_grid(
    comparison_rows,
    output_path,
):
    """
    Create one large image containing all comparisons.
    """

    if len(comparison_rows) == 0:
        print("No images to save.")
        return

    sample_img = comparison_rows[0]["target"]

    img_w, img_h = sample_img.size

    label_height = 40
    rows = len(comparison_rows)

    canvas = Image.new(
        "RGB",
        (
            img_w * 2,
            rows * (img_h + label_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for row_idx, item in enumerate(comparison_rows):

        y_offset = row_idx * (img_h + label_height)

        canvas.paste(
            item["target"],
            (0, y_offset + label_height),
        )

        canvas.paste(
            item["anchor"],
            (img_w, y_offset + label_height),
        )

        draw.text(
            (10, y_offset + 10),
            f"Seed {item['seed']} | Target",
            fill="black",
        )

        draw.text(
            (img_w + 10, y_offset + 10),
            f"Seed {item['seed']} | Anchor",
            fill="black",
        )

    canvas.save(output_path)

    print(f"\nSaved comparison grid:")
    print(output_path)


def run_inference(
    anchor_embed_path: str,
    target_prompt: str,
    output_dir: str = "output_images",
    seeds: Optional[List[int]] = None,
    num_images_per_seed: int = 1,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
    device: str = "cuda",
    save_individual_images: bool = False,
):

    os.makedirs(output_dir, exist_ok=True)

    model_id = "CompVis/stable-diffusion-v1-4"

    print(f"\nLoading {model_id} ...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    # --------------------------------------------------
    # Load anchor embeddings
    # --------------------------------------------------

    anchor_embeds = load_anchor_embeddings(
        anchor_embed_path,
        device=device,
    )

    # --------------------------------------------------
    # Encode target prompt
    # --------------------------------------------------

    print("\nEncoding target prompt...")

    target_embeds, negative_embeds = pipe.encode_prompt(
        prompt=target_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )

    # Match batch dimension
    if anchor_embeds.shape[0] != target_embeds.shape[0]:
        anchor_embeds = anchor_embeds[:1]

    # Normalize scale
    anchor_embeds = normalize_anchor_embeddings(
        anchor_embeds,
        target_embeds,
    )

    print(
        "Target shape:",
        target_embeds.shape,
    )

    print(
        "Anchor shape:",
        anchor_embeds.shape,
    )

    print(
        "Negative shape:",
        negative_embeds.shape,
    )

    if seeds is None or len(seeds) == 0:
        seeds = [42]

    comparison_rows = []

    latent_height = 64
    latent_width = 64

    print(
        f"\nRunning {len(seeds)} seeds..."
    )

    for seed in seeds:

        for img_idx in range(num_images_per_seed):

            print(
                f"\nSeed={seed} "
                f"Image={img_idx+1}/{num_images_per_seed}"
            )

            generator = torch.Generator(
                device=device
            ).manual_seed(seed * 1000 + img_idx)

            latents = torch.randn(
                (
                    1,
                    pipe.unet.config.in_channels,
                    latent_height,
                    latent_width,
                ),
                generator=generator,
                device=device,
                dtype=target_embeds.dtype,
            )

            # ------------------------------------------
            # Target image
            # ------------------------------------------

            target_image = pipe(
                prompt_embeds=target_embeds,
                negative_prompt_embeds=negative_embeds,
                latents=latents.clone(),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]

            # ------------------------------------------
            # Anchor image
            # ------------------------------------------

            anchor_image = pipe(
                prompt_embeds=anchor_embeds,
                negative_prompt_embeds=negative_embeds,
                latents=latents.clone(),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images[0]

            comparison_rows.append(
                {
                    "seed": f"{seed}-{img_idx}",
                    "target": target_image,
                    "anchor": anchor_image,
                }
            )

            if save_individual_images:

                target_path = os.path.join(
                    output_dir,
                    f"seed_{seed}_{img_idx}_target.png",
                )

                anchor_path = os.path.join(
                    output_dir,
                    f"seed_{seed}_{img_idx}_anchor.png",
                )

                target_image.save(target_path)
                anchor_image.save(anchor_path)

    # --------------------------------------------------
    # Save giant comparison image
    # --------------------------------------------------

    grid_path = os.path.join(
        output_dir,
        "all_results.png",
    )

    create_comparison_grid(
        comparison_rows,
        grid_path,
    )

    print("\nFinished.")


if __name__ == "__main__":

    ANCHOR_PATH = (
        "anchor-embeds/anchor_Shiba_Inu_steps100.pt"
    )

    TARGET_PROMPT = "Shiba Inu"

    USER_SEEDS = [
        42,
        2026,
        999,
    ]

    run_inference(
        anchor_embed_path=ANCHOR_PATH,
        target_prompt=TARGET_PROMPT,
        output_dir="anchor_vs_target_results",
        seeds=USER_SEEDS,
        num_images_per_seed=2,
        guidance_scale=3.0,
        num_inference_steps=50,
        device="cuda:0",
        save_individual_images=False,
    )