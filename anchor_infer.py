import os
import torch
from typing import List, Optional
from diffusers import StableDiffusionPipeline

def load_anchor_embeddings(file_path: str, device: str = "cuda") -> torch.Tensor:
    """
    Loads the trained anchor embeddings tensor from disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find anchor embedding file at: {file_path}")
        
    checkpoint = torch.load(file_path, map_location=device)
    anchor_embeds = checkpoint["anchor_embeds"].to(device)
    print(f"Successfully loaded anchor embeddings from {file_path} with shape: {anchor_embeds.shape}")
    return anchor_embeds

def run_inference(
    anchor_embed_path: str,
    target_prompt: str,
    output_dir: str = "output_images",
    num_images_per_seed: int = 1,
    seeds: Optional[List[int]] = None,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0"
):
    """
    Loads SD 1.4, generates images using both the original target prompt and 
    the optimized anchor embeddings, and saves them for visual comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Stable Diffusion Pipeline
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading pipeline: {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype
    ).to(device)
    
    # 2. Load the trained anchor embeddings
    anchor_embeds = load_anchor_embeddings(anchor_embed_path, device=device)
    
    # Adjust batch size match if anchor was saved with a specific batch size configuration
    # pipe.encode_prompt handles building the unconditioned (negative) embeds for us
    print("Encoding target and negative prompts...")
    target_embeds, negative_embeds = pipe.encode_prompt(
        prompt=target_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )
    
    # Default to a random seed if none are provided
    if not seeds:
        seeds = [int(torch.randint(0, 2**15, (1,)).item())]
        
    print(f"Starting inference across {len(seeds)} unique seed(s)...")
    
    for seed in seeds:
        for idx in range(num_images_per_seed):
            print(f"--- Generating Generation Set [Seed: {seed} | Image: {idx+1}/{num_images_per_seed}] ---")
            
            # --- 3. Target Prompt Inference ---
            generator = torch.Generator(device=device).manual_seed(seed)
            target_image = pipe(
                prompt_embeds=target_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            target_path = os.path.join(output_dir, f"seed_{seed}_img_{idx}_target.png")
            target_image.save(target_path)
            
            # --- 4. Anchor Embedding Inference ---
            # Reset generator state to ensure fair random noise generation matching the target pass
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # Match prompt_embeds dimensions if anchor embeddings were saved with a batch dimension size mismatch
            if anchor_embeds.shape[0] != target_embeds.shape[0]:
                # Slice or duplicate to match target shape [1, 77, 768]
                current_anchor_embeds = anchor_embeds[0:1]
            else:
                current_anchor_embeds = anchor_embeds

            anchor_image = pipe(
                prompt_embeds=current_anchor_embeds,
                negative_prompt_embeds=negative_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            anchor_path = os.path.join(output_dir, f"seed_{seed}_img_{idx}_anchor.png")
            anchor_image.save(anchor_path)
            
            print(f"Saved: \n  -> {target_path}\n  -> {anchor_path}")

if __name__ == "__main__":
    # Example Configuration Parameters
    ANCHOR_PATH = "anchor-embeds/anchor__your_prompt_here__steps500.pt" # Replace with your saved .pt file path
    TARGET_PROMPT = "a professional photograph of an astronaut riding a horse" # Replace with your original target prompt
    
    # Controlled Generation Inputs
    USER_SEEDS = [42, 2026, 999]  # Add as many custom seeds as desired
    IMAGES_PER_SEED = 2            # Number of iterations to perform per seed
    
    run_inference(
        anchor_embed_path=ANCHOR_PATH,
        target_prompt=TARGET_PROMPT,
        output_dir="anchor_vs_target_results",
        num_images_per_seed=IMAGES_PER_SEED,
        seeds=USER_SEEDS,
        guidance_scale=3.0,          # Ensure this matches your training configuration guidance scale
        num_inference_steps=50,      # Ensure this matches your training step count configuration
        torch_dtype=torch.bfloat16,  # Enforce precision consistency
        device="cuda:0"
    )