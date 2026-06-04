import argparse
import torch
import sys

# Import the config and training function from your training script
# Assuming the file provided previously is saved as "anchor_trainer.py"
try:
    from anchor_trainer import AnchorConfig, run_anchor_training
except ImportError:
    print("Error: Could not import anchor_trainer. Make sure your training script is named 'anchor_trainer.py' and is in the same directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train Anchor Embeddings for Stable Diffusion 1.4")
    
    # Text Input
    parser.add_argument("--target_prompt", type=str, default="Golden Retriever", help="The text prompt you want to anchor.")
    
    # Training Hyperparameters
    parser.add_argument("--iterations", type=int, default=100, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating embeddings.")
    parser.add_argument("--margin_hyperpara", type=float, default=1.0, help="Distance margin hyperparameter for small timesteps.")
    
    # SD Inference Settings
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Total inference steps for the scheduler.")
    
    # Loss & Smoothing Configuration
    # Added "sigmoid" to choices
    parser.add_argument("--smooth_function", type=str, choices=["linear", "bell", "sigmoid"], default="linear", help="Smoothing function to use for loss weighting.")
    
    # Gaussian (Bell) Specific Hyperparameters
    parser.add_argument("--center_t", type=float, default=30.0, help="Center t parameter (mu) if using the 'bell' smooth function.")
    parser.add_argument("--sigma", type=float, default=10.0, help="Sigma parameter if using the 'bell' smooth function.")
    
    # Sigmoid Specific Hyperparameters
    parser.add_argument("--sigmoid_mid", type=float, default=24.5, help="Midpoint of the 0-49 step range for the 'sigmoid' function.")
    parser.add_argument("--sigmoid_k", type=float, default=0.40, help="Steepness factor for the 'sigmoid' function.")
    
    # Hardware & File System
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--save_path", type=str, default="anchor-embeds", help="Directory where the trained tensor will be saved.")

    args = parser.parse_args()

    print("=== Initializing Anchor Embeddings Training ===")
    print(f"Target Prompt:   '{args.target_prompt}'")
    print(f"Iterations:      {args.iterations}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Smooth Function: {args.smooth_function}")
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
        margin_hyperpara=args.margin_hyperpara,
        smooth_function=args.smooth_function,
        center_t=args.center_t,
        sigma=args.sigma,
        sigmoid_mid=args.sigmoid_mid,  
        sigmoid_k=args.sigmoid_k      
    )
    
    # 2. Run the Training Loop
    try:
        result_message = run_anchor_training(config)
        print("\n=== Training Complete ===")
        print(result_message)
    except Exception as e:
        print(f"\n[!] Training failed with error: {e}")

if __name__ == "__main__":
    main()