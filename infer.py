import argparse
import random
from copy import deepcopy
from math import ceil
from pathlib import Path

import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from nudenet import NudeDetector

TARGET_LABELS = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
CONF_THRESHOLD = 0.2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esd_inference_sd.py",
        description="Generate images with original SD weights and ESD-trained checkpoint.",
    )
    parser.add_argument("--basemodel_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--esd_path", type=str)
    parser.add_argument("--prompt", type=str, default="image of a cowboy drinking a beer")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single fixed seed. Mutually exclusive with --num_samples.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of seeds, e.g. --seeds 42 123 999.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of random seeds to sample when neither --seed nor --seeds is given.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=None,
        help="Columns in the combined grid image. Defaults to number of seeds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_dir", type=str, default="images/sd_inference")
    parser.add_argument("--prefix", type=str, default="sample")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "esd", "both"],
        default="both",
        help="Inference mode: original | esd | both (default).",
    )
    parser.add_argument("--watermark", type=bool, default=True)
    return parser


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Return the final list of seeds based on the three seed-related args."""
    if args.seeds is not None:
        return args.seeds
    if args.seed is not None:
        return [args.seed]
    return [random.randint(0, 2**15) for _ in range(args.num_samples)]


def make_generator(seed: int, device: str) -> torch.Generator:
    gen_device = "cuda" if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=gen_device).manual_seed(seed)


def generate_one(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    device: str,
) -> Image.Image:
    generator = make_generator(seed, device)
    return pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]


def make_grid(
    images: list[Image.Image],
    cols: int,
    cell_w: int,
    cell_h: int,
    padding: int = 4,
    bg_color: tuple[int, int, int] = (30, 30, 30),
) -> Image.Image:
    """
    Arrange *images* into a grid with *cols* columns.

    Each cell is (cell_w × cell_h). Rows with a label strip are NOT added here —
    callers that want labels should annotate images before passing them in.
    """
    rows = ceil(len(images) / cols)
    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), color=bg_color)

    for idx, img in enumerate(images):
        col = idx % cols
        row = idx // cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        grid.paste(img.resize((cell_w, cell_h)), (x, y))

    return grid


def make_comparison_grid(
    originals: list[Image.Image],
    trained: list[Image.Image],
    seeds: list[int],
    cell_w: int,
    cell_h: int,
    cols: int,
) -> Image.Image:
    """
    Build a two-row-per-seed grid:
      Row A (original): one cell per seed
      Row B (trained):  one cell per seed

    Columns = min(cols, len(seeds)) so the layout stays readable.
    """
    assert len(originals) == len(trained) == len(seeds)
    # Interleave: [orig_0, orig_1, ..., trained_0, trained_1, ...]
    # Better for comparison: pair them side by side per seed in groups of 2 cols
    paired: list[Image.Image] = []
    for o, t in zip(originals, trained):
        paired.append(o)
        paired.append(t)

    # Two images per seed → double the column count so each seed pair sits together
    return make_grid(paired, cols=cols * 2, cell_w=cell_w, cell_h=cell_h)

def detect_nsfw_regions(detector, image):
    """
    Detect ONLY sensitive exposed regions.

    Returns:
        list[(x1,y1,x2,y2)]
    """

    image_np = np.array(image)

    results = detector.detect(image_np)

    boxes = []

    for r in results:

        label = r["class"]
        score = r["score"]

        # --- Filter unwanted detections ---
        if label not in TARGET_LABELS:
            continue

        if score < CONF_THRESHOLD:
            continue

        x, y, w, h = r["box"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        boxes.append((x1, y1, x2, y2))

    return boxes

def watermark_regions(
    image: Image.Image,
    boxes,
):
    """
    Draw black censor boxes with a single large white '*' centered.
    """

    draw = ImageDraw.Draw(image)

    for (x1, y1, x2, y2) in boxes:

        # Draw solid black rectangle
        draw.rectangle(
            [x1, y1, x2, y2],
            fill=(0, 0, 0),
        )

        # Compute region size
        w = x2 - x1
        h = y2 - y1

        # Choose font size based on box size
        font_size = int(min(w, h) * 0.7)

        try:
            font = ImageFont.truetype(
                "DejaVuSans-Bold.ttf",
                font_size,
            )
        except:
            font = ImageFont.load_default()

        text = "*"

        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Center the '*'
        text_x = x1 + (w - text_w) // 2
        text_y = y1 + (h - text_h) // 2

        draw.text(
            (text_x, text_y),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    return image

def main() -> None:
    args = build_parser().parse_args()
    torch.set_grad_enabled(False)

    seeds = resolve_seeds(args)
    print(f"Running {len(seeds)} seed(s): {seeds}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    
    if args.watermark:
        detector = NudeDetector()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.basemodel_id,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,
    ).to(args.device)

    original_weights = deepcopy(pipe.unet.state_dict())
    esd_weights = None
    if args.mode in ["esd", "both"]:
        esd_weights = load_file(args.esd_path, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        device=args.device,
    )

    original_images: list[Image.Image] = []
    trained_images: list[Image.Image] = []

    for seed in seeds:
        if args.mode in ["original", "both"]:
            pipe.unet.load_state_dict(original_weights, strict=False)
            orig = generate_one(pipe, seed=seed, **gen_kwargs)
            
            # Detect NSFW
            boxes = detect_nsfw_regions(detector ,orig)
            if boxes:
                orig = watermark_regions(orig, boxes)

            original_images.append(orig)

        if args.mode in ["esd", "both"]:
            pipe.unet.load_state_dict(esd_weights, strict=False)
            trained = generate_one(pipe, seed=seed, **gen_kwargs)
            
            # Detect NSFW
            boxes = detect_nsfw_regions(detector, trained)
            if boxes:
                trained = watermark_regions(trained, boxes)

            trained_images.append(trained)  
    
        # Per-seed saves
        # orig.save(output_dir / f"{args.prefix}_original_seed{seed}.png")
        # trained.save(output_dir / f"{args.prefix}_trained_seed{seed}.png")

        # Per-seed side-by-side comparison
        if args.mode == "both":
            compare = Image.new("RGB", (args.width * 2, args.height))
            compare.paste(orig, (0, 0))
            compare.paste(trained, (args.width, 0))
        # compare.save(output_dir / f"{args.prefix}_compare_seed{seed}.png")

        print(f"  Seed {seed} done.")

    # ------------------------------------------------------------------ #
    # Combined grid: originals on top rows, trained below, paired by seed #
    # ------------------------------------------------------------------ #
    cols = args.grid_cols if args.grid_cols is not None else len(seeds)
    cols = max(1, min(cols, len(seeds)))

    if args.mode == "original":
        orig_grid = make_grid(
            original_images,
            cols=cols,
            cell_w=args.width,
            cell_h=args.height,
        )

        combined_path = output_dir / f"{args.prefix}_original_n{len(seeds)}.png"
        orig_grid.save(combined_path)

    elif args.mode == "esd":
        trained_grid = make_grid(
            trained_images,
            cols=cols,
            cell_w=args.width,
            cell_h=args.height,
        )

        combined_path = output_dir / f"{args.prefix}_esd_n{len(seeds)}.png"
        trained_grid.save(combined_path)

    else:  # both
        orig_grid = make_grid(
            original_images,
            cols=cols,
            cell_w=args.width,
            cell_h=args.height,
        )

        trained_grid = make_grid(
            trained_images,
            cols=cols,
            cell_w=args.width,
            cell_h=args.height,
        )

        combined = Image.new(
            "RGB",
            (orig_grid.width, orig_grid.height + trained_grid.height),
        )

        combined.paste(orig_grid, (0, 0))
        combined.paste(trained_grid, (0, orig_grid.height))

        combined_path = output_dir / f"{args.prefix}_combined_n{len(seeds)}.png"
        combined.save(combined_path)

    print(f"\nAll done. Output saved to: {combined_path}")


if __name__ == "__main__":
    main()