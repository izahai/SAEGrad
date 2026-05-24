import argparse
import sys

import torch

sys.path.append(".")
from trainer import ESDConfig, run_esd_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TrainESD for SD",
        description="Fine-tune a Stable Diffusion UNet to erase concepts.",
    )
    parser.add_argument(
        "--basemodel_id",
        help="HF model id for any Stable Diffusion-compatible diffusers pipeline",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument("--erase_concept", help="concept to erase", type=str, required=True)
    parser.add_argument("--erase_from", help="target concept to erase from", type=str, default=None)
    parser.add_argument("--num_inference_steps", help="number of denoising steps", type=int, default=50)
    parser.add_argument("--guidance_scale", help="guidance scale used to sample xt", type=float, default=3)
    parser.add_argument(
        "--train_method",
        help="train method (esd-x, esd-u, esd-all, esd-x-strict, selfattn, specific-layer). Legacy aliases still work.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target_layers",
        help="module paths to train when --train_method specific-layer is used",
        nargs="+",
        default=None,
    )
    parser.add_argument("--iterations", help="number of optimization steps", type=int, default=200)
    parser.add_argument(
        "--lr",
        help="learning rate. If omitted, a family-specific default is used.",
        type=float,
        default=None,
    )
    parser.add_argument("--batchsize", help="batch size for prompt embeddings", type=int, default=1)
    parser.add_argument(
        "--resolution",
        help="training resolution. Defaults to the base model native size.",
        type=int,
        default=None,
    )
    parser.add_argument("--negative_guidance", help="negative guidance value", type=float, default=2)
    parser.add_argument("--save_path", help="directory to save checkpoints", type=str, default="esd-models/sd/")
    parser.add_argument("--device", help="device to train on", type=str, default="cuda:0")
    parser.add_argument(
        "--save_gradient",
        help="List of 2D matrix weight layer names to save gradients for",
        nargs="+",
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ESDConfig(
        family="sd",
        base_model_id=args.basemodel_id,
        erase_concept=args.erase_concept,
        erase_from=args.erase_from,
        train_method=args.train_method,
        iterations=args.iterations,
        lr=args.lr,
        negative_guidance=args.negative_guidance,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        batch_size=args.batchsize,
        resolution=args.resolution,
        save_path=args.save_path,
        device=args.device,
        torch_dtype=torch.bfloat16,
        target_layers=args.target_layers,
        save_gradient=args.save_gradient,
    )
    checkpoint_path = run_esd_training(config)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
    gradients = torch.load("esd-models/sd/components_mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj_weight.pt")

    print(gradients.shape)  # Output: torch.Size([iterations, flattened_weights])
    print(gradients.dtype)  # Matches your training precision (e.g., torch.float32 or torch.bfloat16)
    print(gradients.device) # Extracted to 'cpu' for safe cross-device loading
