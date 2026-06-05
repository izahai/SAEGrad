import os
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import torch.optim as optim
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

from utils.sd_utils import esd_sd_call
from anchor_train.utils import make_sampling_generator, save_anchor_embeddings
from anchor_train.config import AnchorConfig, TrajectoryStep


class SDAnchorTrajectoryTrainer:
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

        noise = torch.randn_like(target_embeds) * target_embeds.std()
        mixed_embeds = (1 - config.anchor_noise) * target_embeds.clone().detach() + config.anchor_noise * noise
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

    def compute_loss(
        self,
        predicted_noise_target: torch.Tensor,  # BCWH
        predicted_noise_anchor: torch.Tensor,  # BCWH
        t: torch.Tensor,                        # B
    ) -> torch.Tensor:

        mse_distance = torch.mean(
            (predicted_noise_target - predicted_noise_anchor) ** 2,
            dim=list(range(1, len(predicted_noise_target.shape))),  # [1,2,3]
        )  # (B,)

        return mse_distance