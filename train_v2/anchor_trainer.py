import os
import torch
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import torch.optim as optim
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# NOTE: esd_sd_call is no longer needed — we roll out the trajectory ourselves (no CFG).

def save_anchor_embeddings(anchor_embeds: torch.Tensor, save_dir: str, filename: str = "anchor_embeds.pt") -> str:
    """Detaches, moves to CPU, and saves the trained anchor embeddings tensor."""
    os.makedirs(save_dir, exist_ok=True)
    clean_embeds = anchor_embeds.detach().cpu()
    save_path = os.path.join(save_dir, filename)
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
    margin_hyperpara: float          # distance margin
    smooth_function: str             # "linear", "bell", "sigmoid", "simple"
    center_t: Optional[float]        # if using bell smooth function, 35
    sigma: Optional[float]           # if using bell smooth function, 5
    sigmoid_k: Optional[float]       # if using sigmoid smooth function, 0.4
    sigmoid_mid: Optional[float]     # if using sigmoid smooth function, 24.5
    train_till_timestep: int # k: sample target timestep index from [0, k-1]

    # --- DEFAULT FIELDS (Must go last) ---
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda:0"
    anchor_save_path: str = "anchor-embeds"


@dataclass
class TrajectoryStep:
    latent_model_input: torch.Tensor  # x_t actually fed to the UNet (scaled, detached / no-grad)
    timestep: torch.Tensor            # scheduler timestep value (e.g. 981)
    timestep_index: int               # index into scheduler.timesteps
    target_noise: torch.Tensor        # cached UNet noise pred under the target prompt (no-grad)
    metrics: Dict[str, Any] = field(default_factory=dict)


def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


class SDAnchorTrainer:
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
            null_embeds = null_embeds.to(config.device)  # kept for reference; unused without CFG

            timestep_cond = None
            if pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(config.batch_size)
                timestep_cond = pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=pipe.unet.config.time_cond_proj_dim,
                ).to(device=config.device, dtype=config.torch_dtype)
            else:
                print("SD1.4 does not have timestep cond proj!")

        # Anchor init: 70% target + 30% scale-matched noise
        alpha = 0.3
        noise = torch.randn_like(target_embeds) * target_embeds.std()
        mixed_embeds = (1 - alpha) * target_embeds.clone().detach() + alpha * noise
        anchor_embeds = mixed_embeds.requires_grad_(True)

        return {
            "target_embeds": target_embeds,
            "anchor_embeds": anchor_embeds,
            "null_embeds": null_embeds,
            "timestep_cond": timestep_cond,
        }

    @torch.no_grad()
    def generate_trajectory(
        self,
        pipe,
        context: Dict[str, Any],
        config: AnchorConfig,
        run_till_timestep: int,
        seed: int,
    ) -> List[TrajectoryStep]:
        """
        Roll out the diffusion trajectory from index 0 up to (and including)
        `run_till_timestep`, conditioning ONLY on the target prompt (no CFG).
        Caches (latent_model_input, timestep, target_noise) at every step.
        """
        device = config.device
        generator = make_sampling_generator(device, seed)

        # Fresh scheduler state for this rollout
        pipe.scheduler.set_timesteps(config.num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps  # e.g. tensor([981, 961, ..., 21, 1])

        # Initial latent noise x_T
        num_channels_latents = pipe.unet.config.in_channels
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        latents = pipe.prepare_latents(
            config.batch_size,
            num_channels_latents,
            height,
            width,
            context["target_embeds"].dtype,
            device,
            generator,
            None,
        )

        trajectory: List[TrajectoryStep] = []

        for i in range(run_till_timestep + 1):
            t = timesteps[i]
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)

            # CFG-free conditional prediction (target prompt only)
            noise_pred_target = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=context["target_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            trajectory.append(
                TrajectoryStep(
                    latent_model_input=latent_model_input,
                    timestep=t,
                    timestep_index=i,
                    target_noise=noise_pred_target,
                )
            )

            # Advance x_t -> x_{t-1} with the (CFG-free) target prediction
            latents = pipe.scheduler.step(noise_pred_target, t, latents, return_dict=False)[0]

        return trajectory

    # ---- Loss weighting (unchanged) ----
    def get_linear_loss_weights(self, t: torch.Tensor):
        T = self.config.num_inference_steps
        alpha = t.float() / T
        beta = 1 - alpha
        return alpha, beta

    def get_bell_loss_weights(self, t: torch.Tensor):
        mu = self.config.center_t
        sigma = self.config.sigma
        bell_weight = torch.exp(-((t.float() - mu) ** 2) / (2 * (sigma ** 2)))
        return 1.0 - bell_weight, bell_weight

    def get_sigmoid_loss_weights(self, t: torch.Tensor):
        t_mid = self.config.sigmoid_mid if self.config.sigmoid_mid is not None else 24.5
        k = self.config.sigmoid_k if self.config.sigmoid_k is not None else 0.4
        alpha = 1.0 / (1.0 + torch.exp(-k * (t.float() - t_mid)))
        beta = 1.0 - alpha
        return alpha, beta

    def get_simple_loss(self, t: torch.Tensor):
        alpha = torch.ones_like(t, dtype=torch.float32)
        beta = torch.zeros_like(t, dtype=torch.float32)
        return alpha, beta

    def compute_loss(
        self,
        predicted_noise_target: torch.Tensor,  # BCWH
        predicted_noise_anchor: torch.Tensor,  # BCWH
        t: torch.Tensor,                        # B
    ) -> torch.Tensor:
        if self.config.smooth_function == "bell":
            alpha, beta = self.get_bell_loss_weights(t)
        elif self.config.smooth_function == "linear":
            alpha, beta = self.get_linear_loss_weights(t)
        elif self.config.smooth_function == "sigmoid":
            alpha, beta = self.get_sigmoid_loss_weights(t)
        elif self.config.smooth_function == "simple":
            beta, alpha = self.get_simple_loss(t)  # beta = 1.0, alpha = 0.0

        dims_to_extend = len(predicted_noise_target.shape) - 1  # (BCWH) -> 3
        for _ in range(dims_to_extend):
            alpha = alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        mse_distance = torch.mean(
            (predicted_noise_target - predicted_noise_anchor) ** 2,
            dim=list(range(1, len(predicted_noise_target.shape))),  # [1,2,3]
        )  # (B,)

        mse = beta * mse_distance
        # D_mse = alpha * (self.config.margin_hyperpara - mse_distance)
        total_loss = mse  # + D_mse
        return total_loss


def run_anchor_training(config: AnchorConfig) -> str:
    """Executes the trajectory-cached anchor embedding optimization loop with immediate online updates."""
    trainer = SDAnchorTrainer(config)

    print(f"Loading Stable Diffusion pipeline: {trainer.default_base_model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        trainer.default_base_model_id,
        torch_dtype=config.torch_dtype,
    ).to(config.device)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    context = trainer.prepare_context(pipe, config)
    optimizer = optim.Adam([context["anchor_embeds"]], lr=config.lr)

    print(f"Starting anchor training for {config.iterations} iterations on {config.device}...")

    # Array to track absolutely every loss calculated per backprop step
    all_backprop_losses = []
    last_loss = 0.0

    for i in tqdm(range(config.iterations), desc="Optimizing Anchor Embeds"):
        # 1. Sample a target timestep index + seed for this iteration
        # run_till_timestep = random.randint(0, config.train_till_timestep - 1)
        run_till_timestep = config.train_till_timestep - 1
        seed = random.randint(0, 2 ** 15)

        # 2. Roll out and cache the trajectory (no grad, no CFG, target prompt only)
        trajectory = trainer.generate_trajectory(
            pipe, context, config, run_till_timestep, seed
        )

        num_pairs = len(trajectory)
        iter_loss = 0.0

        # 3. Replay each cached pair, backprop, and step immediately (online update)
        for step in trajectory:
            optimizer.zero_grad()  # Clear gradients for the current pair

            noise_pred_anchor = pipe.unet(
                step.latent_model_input,           # cached x_t (same input used for target)
                step.timestep,
                encoder_hidden_states=context["anchor_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            t_tensor = torch.tensor([step.timestep_index], device=config.device)
            loss = trainer.compute_loss(
                predicted_noise_target=step.target_noise,
                predicted_noise_anchor=noise_pred_anchor,
                t=t_tensor,
            ).mean()

            # Backprop the raw loss and update the weights immediately
            loss.backward()
            optimizer.step()
            
            # Record individual step loss
            loss_val = loss.item()
            all_backprop_losses.append(loss_val)
            iter_loss += loss_val

        # 4. Clear the cache before the next iteration
        trajectory.clear()
        del trajectory

        # Calculate average loss across the trajectory for logging purposes
        last_loss = iter_loss / num_pairs if num_pairs > 0 else 0.0
        
        if i % 50 == 0 or i == config.iterations - 1:
            tqdm.write(
                f"Iteration {i:04d} | t_idx: {run_till_timestep:03d} "
                f"| pairs: {num_pairs:02d} | Avg Trajectory Loss: {last_loss:.4f}"
            )

    # 5. Generate and save the loss curve chart
    vis_dir = "visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    safe_prompt_name = "".join([c if c.isalnum() else "_" for c in config.target_prompt[:20]])
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_backprop_losses, color='royalblue', alpha=0.7, label='Per-step Loss')
    plt.title(f"Anchor Optimization Loss Curve\nPrompt: '{config.target_prompt[:40]}...'")
    plt.xlabel("Global Backprop Steps (Every Pair)")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plot_filename = f"loss_curve_{safe_prompt_name}_steps{config.iterations}.png"
    plot_path = os.path.join(vis_dir, plot_filename)
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Loss visualization plot saved completely to: {plot_path}")

    # 6. Save final optimized weights
    final_anchor_embeds = context["anchor_embeds"]
    filename = f"anchor_{safe_prompt_name}_steps{config.iterations}.pt"
    save_anchor_embeddings(final_anchor_embeds, config.anchor_save_path, filename)

    return f"Success! Anchor training completed {config.iterations} iterations. Final avg loss: {last_loss:.4f}."