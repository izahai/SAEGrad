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
from anchor_train.utils import make_sampling_generator
from anchor_train.config import AnchorConfig, StepResult

class SDAnchorSideTrainer():
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

        # Generate noise that matches the scale of the target
        noise = torch.randn_like(target_embeds) * target_embeds.std()

        # Interpolate: (1 - alpha) * target + alpha * noise
        mixed_embeds = (1 - config.anchor_noise) * target_embeds.clone().detach() + config.anchor_noise * noise
        anchor_embeds = mixed_embeds.requires_grad_(True)
        
        return {
            "target_embeds": target_embeds,
            "anchor_embeds": anchor_embeds,
            "null_embeds": null_embeds,
            "timestep_cond": timestep_cond,
        }
    
    def training_step(
        self, pipe, context: Dict[str, Any], config: AnchorConfig, is_even: bool,
        ) -> StepResult:
        
        if is_even:
            run_till_timestep = 0
        else:
            run_till_timestep = config.train_till_timestep - 1
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
        
    def compute_loss(
        self,
        predicted_noise_target: torch.Tensor, # BCWH
        predicted_noise_anchor: torch.Tensor, # BCWH
        t: torch.Tensor # B
        ) -> torch.Tensor:
    
        mse_distance = torch.mean(
            (predicted_noise_target - predicted_noise_anchor) ** 2, 
            dim=list(range(1, len(predicted_noise_target.shape))) # dim=[1,2,3]
        ) # (B,)
        
        return mse_distance