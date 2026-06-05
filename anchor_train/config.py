import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class AnchorConfig:
    # --- NON-DEFAULT FIELDS (Must go first) ---
    target_prompt: str
    method: str 
    guidance_scale: float 
    num_inference_steps: int 
    iterations: int
    lr: Optional[float]
    batch_size: int
    train_till_timestep: int # k: sample target timestep index from [0, k-1]
    anchor_noise: float
    
    # --- DEFAULT FIELDS (Must go last) ---
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda:0"
    anchor_save_path: str = "anchor-embeds"
    
@dataclass
class StepResult:
    target_noise: torch.Tensor
    anchor_noise: torch.Tensor
    timestep_index: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class TrajectoryStep:
    latent_model_input: torch.Tensor  # x_t actually fed to the UNet (scaled, detached / no-grad)
    timestep: torch.Tensor            # scheduler timestep value (e.g. 981)
    timestep_index: int               # index into scheduler.timesteps
    target_noise: torch.Tensor        # cached UNet noise pred under the target prompt (no-grad)
    metrics: Dict[str, Any] = field(default_factory=dict)