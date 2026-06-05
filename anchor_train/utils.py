import os
import torch

def save_anchor_embeddings(context: dict, config: AnchorConfig) -> str:
    """
    Extracts, detaches, and saves the trained anchor embeddings tensor 
    using the provided run configuration.
    
    Args:
        context (dict): The training context containing 'anchor_embeds'.
        config (AnchorConfig): The configuration for the training run.
        
    Returns:
        str: The absolute or relative file path where the embeddings were saved.
    """
    # 1. Ensure output directory exists
    os.makedirs(config.anchor_save_path, exist_ok=True)
    
    # 2. Extract and clean the target tensor
    anchor_embeds = context["anchor_embeds"]
    clean_embeds = anchor_embeds.detach().cpu()
    
    # 3. Construct a safe filename using config properties
    safe_prompt_name = "".join([c if c.isalnum() else "_" for c in config.target_prompt[:20]])
    filename = f"anchor_{config.method}_{safe_prompt_name}_steps{config.iterations}.pt"
    save_path = os.path.join(config.anchor_save_path, filename)
    
    # 4. Save to disk
    torch.save({"anchor_embeds": clean_embeds}, save_path)
    print(f"Anchor embeddings successfully saved to: {save_path}")
    
    return save_path

def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)