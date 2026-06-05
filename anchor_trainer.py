import os
import torch
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline

from utils.sd_utils import esd_sd_call

def save_anchor_embeddings(anchor_embeds: torch.Tensor, save_dir: str, filename: str = "anchor_embeds.pt") -> str:
    """
    Detaches, moves to CPU, and saves the trained anchor embeddings tensor.
    
    Args:
        anchor_embeds (torch.Tensor): The optimized anchor embedding tensor.
        save_dir (str): Directory where the embedding should be saved.
        filename (str): Name of the file (defaults to 'anchor_embeds.pt').
        
    Returns:
        str: Absolute path to the saved file.
    """
    # 1. Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Detach from graph and move to CPU to ensure clean loading later
    clean_embeds = anchor_embeds.detach().cpu()
    
    # 3. Construct save path
    save_path = os.path.join(save_dir, filename)
    
    # 4. Save the tensor
    torch.save({"anchor_embeds": clean_embeds}, save_path)
    
    print(f"Anchor embeddings successfully saved to: {save_path}")
    return save_path
    
@dataclass
class AnchorConfig:
    # --- NON-DEFAULT FIELDS (Must go first) ---
    target_prompt: str
    guidance_scale: float 
    num_inference_steps: int 
    iterations: int
    lr: Optional[float]
    batch_size: int
    margin_hyperpara: float # distance margin
    smooth_function: str # "linear", "bell"
    center_t: Optional[float]  # if using bell smooth function, 35
    sigma: Optional[float] # if using bell smooth function, 5
    sigmoid_k: Optional[float] # if using sigmoid smooth function, 0.4
    sigmoid_mid: Optional[float] # if using sigmoid smooth function, 24.5
    train_till_timestep: int # k: sample target timestep index from [0, k-1]
    
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
    
def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)

class SDAnchorTrainer():
    def __init__(self, config: AnchorConfig):
        self.config = config
        self.default_base_model_id = "CompVis/stable-diffusion-v1-4"
        
    def prepare_context(self, pipe, config: AnchorConfig) -> Dict[str, Any]:
        with torch.no_grad():
            target_embeds, null_embeds = pipe.encode_prompt(
                prompt=config.target_prompt,
                device=config.device,
                num_images_per_prompt=config.batch_size,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
            target_embeds = target_embeds.to(config.device)
            null_embeds = null_embeds.to(config.device)

            timestep_cond = None
            if pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(config.batch_size)
                timestep_cond = pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=pipe.unet.config.time_cond_proj_dim,
                ).to(device=config.device, dtype=config.torch_dtype)
            else: 
                print("SD1.4 doen not have timestep cond proj!")
                
        alpha = 0.3  # 30% noise, 70% target

        # Generate noise that matches the scale of the target
        noise = torch.randn_like(target_embeds) * target_embeds.std()

        # Interpolate: (1 - alpha) * target + alpha * noise
        mixed_embeds = (1 - alpha) * target_embeds.clone().detach() + alpha * noise
        anchor_embeds = mixed_embeds.requires_grad_(True)
        
        # noise_scale = 0.1
        # noise = torch.randn_like(target_embeds) * noise_scale
        
        # anchor_embeds = (target_embeds.clone().detach() + noise).requires_grad_(True)
        # anchor_embeds = torch.randn_like(target_embeds, requires_grad=True)
                
        return {
            "target_embeds": target_embeds,
            "anchor_embeds": anchor_embeds,
            "null_embeds": null_embeds,
            "timestep_cond": timestep_cond,
        }
    
    def training_step(self, pipe, context: Dict[str, Any], config: AnchorConfig) -> StepResult:
        
        run_till_timestep = random.randint(0, config.train_till_timestep - 1)
        seed = random.randint(0, 2**15)
        
        with torch.no_grad():
            xt = esd_sd_call(
                pipe,
                prompt_embeds=context["target_embeds"],
                negative_prompt_embeds=context["null_embeds"],
                num_images_per_prompt=1,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                run_till_timestep=run_till_timestep,
                generator=make_sampling_generator(config.device, seed),
                output_type="latent",
            ).images
            
            # pipe.scheduler.timesteps: tensor([981, 961, 941, 921, 901,  ...,  81,  61,  41,  21,   1])
            # we choose the index (run_till_timestep) in that array
            timestep = pipe.scheduler.timesteps[run_till_timestep]
            
            # Calculate the predicted noise of target prompt
            noise_pred_target = pipe.unet(
                xt,
                timestep,
                encoder_hidden_states=context["target_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            
        
        # Calculate the predicted noise of anchor prompt
        # Allow gradients
        noise_pred_anchor = pipe.unet(
            xt,
            timestep,
            encoder_hidden_states=context["anchor_embeds"],
            timestep_cond=context["timestep_cond"],
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        
        return StepResult(target_noise=noise_pred_target, anchor_noise=noise_pred_anchor, timestep_index=run_till_timestep)
        
    def get_linear_loss_weights(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get alpha, (1-alpha) at timestep t
        """
        
        T = self.config.num_inference_steps
        
        alpha = t.float() / T # alpha(t)
        beta = (1 - alpha) # (1 - alpha(t))
        
        return alpha, beta
    
    def get_bell_loss_weights(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.config.center_t
        sigma = self.config.sigma
        
        bell_weight = torch.exp(-((t.float() - mu) ** 2) / (2 * (sigma ** 2)))
        
        return 1.0 - bell_weight, bell_weight
    
    def get_sigmoid_loss_weights(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get alpha, (1-alpha) at timestep index t using a parameterized sigmoid function.
        Assumes t ranges from 0 to (num_inference_steps - 1).
        """
        # Retrieve hyperparameters from config
        t_mid = self.config.sigmoid_mid if self.config.sigmoid_mid is not None else 24.5
        k = self.config.sigmoid_k if self.config.sigmoid_k is not None else 0.4
        
        # Calculate alpha(t) using the sigmoid formula
        # t.float() ensures we don't hit integer division issues
        alpha = 1.0 / (1.0 + torch.exp(-k * (t.float() - t_mid)))
        beta = 1.0 - alpha
        
        return alpha, beta
    
    def get_simple_loss(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get static weights (1.0, 0.0) regardless of the timestep.
        """
        # torch.ones_like and zeros_like ensure the output matches 
        # the device (CPU/GPU) and shape of the input tensor 't'
        alpha = torch.ones_like(t, dtype=torch.float32)
        beta = torch.zeros_like(t, dtype=torch.float32)
        
        return alpha, beta
        
    def compute_loss(
        self,
        predicted_noise_target: torch.Tensor, # BCWH
        predicted_noise_anchor: torch.Tensor, # BCWH
        t: torch.Tensor # B
        ) -> torch.Tensor:
        
        if self.config.smooth_function == "bell":
            alpha, beta = self.get_bell_loss_weights(t)
        elif self.config.smooth_function == "linear":
            alpha, beta = self.get_linear_loss_weights(t)
        elif self.config.smooth_function == "sigmoid":
            alpha, beta = self.get_sigmoid_loss_weights(t)
        elif self.config.smooth_function == "simple":
            beta, alpha = self.get_simple_loss(t) # beta = 1.0, alpha = 0.0
            
        dims_to_extend = len(predicted_noise_target.shape) - 1 # (BCWH) -> (4-1)=3
        
        # Unsqueeze dim from (B,) -> B111
        for _ in range(dims_to_extend):
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            
        mse_distance = torch.mean(
            (predicted_noise_target - predicted_noise_anchor) ** 2, 
            dim=list(range(1, len(predicted_noise_target.shape))) # dim=[1,2,3]
        ) # (B,)
        
        return mse_distance
    
    
def run_anchor_training(config: AnchorConfig) -> str:
    """
    Executes the anchor embedding optimization loop.
    
    Args:
        config (AnchorConfig): The configuration for the training run.
        
    Returns:
        str: A status message indicating completion and final loss.
    """
    trainer = SDAnchorTrainer(config)
        
    # 2. Load the pipeline
    print(f"Loading Stable Diffusion pipeline: {trainer.default_base_model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        trainer.default_base_model_id,
        torch_dtype=config.torch_dtype
    ).to(config.device)
    
    # Freeze the pipeline parameters since we are only optimizing the anchor embeddings
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    
    # 3. Prepare the embeddings context
    context = trainer.prepare_context(pipe, config)
    
    # 4. Initialize the optimizer
    # We only pass the anchor_embeds to the optimizer since that's what we're tuning
    optimizer = optim.Adam([context["anchor_embeds"]], lr=config.lr)
    
    print(f"Starting anchor training for {config.iterations} iterations on {config.device}...")
    
    # 5. The Training Loop
    # We leave the models in eval mode since they are frozen, though the gradients 
    # will flow back to the anchor_embeds leaf tensor.
    loss_history = []
    for i in tqdm(range(config.iterations), desc="Optimizing Anchor Embeds"):
        optimizer.zero_grad()
        
        # Forward pass / get step result
        step_result = trainer.training_step(pipe, context, config)
        
        # Format the timestep index as a batched tensor for compute_loss
        t_tensor = torch.tensor([step_result.timestep_index], device=config.device)
        
        # Compute loss
        loss_tensor = trainer.compute_loss(
            predicted_noise_target=step_result.target_noise,
            predicted_noise_anchor=step_result.anchor_noise,
            t=t_tensor
        )
        
        # Aggregate loss across the batch (assuming batch size of 1, mean is safe)
        loss = loss_tensor.mean()
        
        # Backpropagation
        loss.backward()
        
        # Update embeddings
        optimizer.step()

        # Record the loss for later export and visualization
        loss_history.append(loss.item())
        
        # Optional: Print loss every 50 steps
        if i % 50 == 0 or i == config.iterations - 1:
            tqdm.write(f"Iteration {i:04d} | Timestep: {step_result.timestep_index:03d} | Loss: {loss.item():.4f}")

    vis_dir = "visualization"
    os.makedirs(vis_dir, exist_ok=True)
    safe_prompt_name = "".join([c if c.isalnum() else "_" for c in config.target_prompt[:20]])

    loss_save_path = os.path.join(vis_dir, f"loss_values_{safe_prompt_name}_steps{config.iterations}.pt")
    torch.save({"loss_history": loss_history}, loss_save_path)
    print(f"Loss values successfully saved to: {loss_save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color="royalblue", alpha=0.8, label="Loss")
    plt.title(f"Anchor Optimization Loss Curve\nPrompt: '{config.target_prompt[:40]}...'")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plot_path = os.path.join(vis_dir, f"loss_curve_{safe_prompt_name}_steps{config.iterations}.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Loss visualization plot saved completely to: {plot_path}")

    final_anchor_embeds = context["anchor_embeds"]
    safe_prompt_name = "".join([c if c.isalnum() else "_" for c in config.target_prompt[:20]])
    filename = f"anchor_{safe_prompt_name}_steps{config.iterations}.pt"
    saved_at = save_anchor_embeddings(final_anchor_embeds, config.anchor_save_path, filename)
    
    # 6. Return Completion Status
    return f"Success! Anchor training completed {config.iterations} iterations. Final loss: {loss.item():.4f}."