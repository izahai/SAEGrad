import os
import torch

def save_anchor_embeddings(anchor_embeds: torch.Tensor, save_dir: str, filename: str = "anchor_embeds.pt") -> str:
    """Detaches, moves to CPU, and saves the trained anchor embeddings tensor."""
    os.makedirs(save_dir, exist_ok=True)
    clean_embeds = anchor_embeds.detach().cpu()
    save_path = os.path.join(save_dir, filename)
    torch.save({"anchor_embeds": clean_embeds}, save_path)
    print(f"Anchor embeddings successfully saved to: {save_path}")
    return save_path

def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)