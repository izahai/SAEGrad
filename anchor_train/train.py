import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random
from tqdm import tqdm

from diffusers import StableDiffusionPipeline

from anchor_train.stepTrainer import SDAnchorStepTrainer
from anchor_train.trajectoryTrainer import SDAnchorTrajectoryTrainer
from anchor_train.config import AnchorConfig
from anchor_train.utils import save_anchor_embeddings


def run_anchor_step_training(config: AnchorConfig) -> str:
    """
    Executes the anchor embedding optimization loop.
    
    Args:
        config (AnchorConfig): The configuration for the training run.
        
    Returns:
        str: A status message indicating completion and final loss.
    """
    trainer = SDAnchorStepTrainer(config)
        
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

def run_anchor_trajectory_training(config: AnchorConfig) -> str:
    """Executes the trajectory-cached anchor embedding optimization loop with immediate online updates."""
    trainer = SDAnchorTrajectoryTrainer(config)

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

def run_anchor_side_training(config: AnchorConfig) -> str:
    """
    Executes the anchor embedding optimization loop.
    
    Args:
        config (AnchorConfig): The configuration for the training run.
        
    Returns:
        str: A status message indicating completion and final loss.
    """
    trainer = SDAnchorStepTrainer(config)
        
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
        is_even = i % 2 == 0
        step_result = trainer.training_step(pipe, context, config, is_even)
        
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