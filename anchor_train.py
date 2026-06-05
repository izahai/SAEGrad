import argparse
import torch
import sys

# Import the config and training function from your training script
# Assuming the file provided previously is saved as "anchor_trainer.py"
try:
    from anchor_train.config import AnchorConfig
    from anchor_train.train import (
    run_anchor_step_training, run_anchor_trajectory_training, run_anchor_side_training
    )
except ImportError:
    print("Error: Could not import anchor_train. Run this from the repository root or add the repository root to PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train Anchor Embeddings for Stable Diffusion 1.4")
    
    # Text Input
    parser.add_argument("--target_prompt", type=str, default="Golden Retriever", help="The text prompt you want to anchor.")
    parser.add_argument("--method", type=str, choices=["step", "trajectory", "side"], default="step")
    
    # Training Hyperparameters
    parser.add_argument("--iterations", type=int, default=101, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating embeddings.")
    
    # SD Inference Settings
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Total inference steps for the scheduler.")
    parser.add_argument("--train_till_timestep", type=int, default=5, help="Total train steps for the training.")
    parser.add_argument("--anchor_noise", type=float, default=1.0, help="Noise level for anchor embeddings.")
    
    # Hardware & File System
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--save_path", type=str, default="anchor-embeds", help="Directory where the trained tensor will be saved.")

    args = parser.parse_args()

    print("=== Initializing Anchor Embeddings Training ===")
    print(f"Target Prompt:   '{args.target_prompt}'")
    print(f"Guidance Scale:  {args.guidance_scale}")
    print(f"Method:          {args.method}")
    print(f"Iterations:      {args.iterations}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Train till time step: {args.train_till_timestep}")
    print(f"Device:          {args.device}")
    print("===============================================")

    # 1. Instantiate the Configuration (including new sigmoid attributes)
    config = AnchorConfig(
        target_prompt=args.target_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        iterations=args.iterations,
        lr=args.lr,
        batch_size=args.batch_size,
        torch_dtype=torch.bfloat16, 
        device=args.device,
        anchor_save_path=args.save_path,
        train_till_timestep=args.train_till_timestep,
        anchor_noise=args.anchor_noise
    )
    
    # 2. Run the Training Loop
    try:
        if args.method == "step":
            result_message = run_anchor_step_training(config)
        elif args.method == "trajectory":
            result_message = run_anchor_trajectory_training(config)
        elif args.method == "side":
            result_message = run_anchor_side_training(config)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        print("\n=== Training Complete ===")
        print(result_message)
    except Exception as e:
        print(f"\n[!] Training failed with error: {e}")

if __name__ == "__main__":
    main()
