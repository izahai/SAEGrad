import torch
import torch.optim as optim
import random
from tqdm import tqdm

from diffusers import StableDiffusionPipeline

from anchor_train.stepTrainer import SDAnchorStepTrainer
from anchor_train.trajectoryTrainer import SDAnchorTrajectoryTrainer
from anchor_train.sideTrainer import SDAnchorSideTrainer
from anchor_train.config import AnchorConfig
from anchor_train.utils import save_anchor_embeddings
from anchor_train.vis import LossVisualizer


def run_training_pipeline(config: AnchorConfig) -> str:
    try:
        if config.method == "step":
            result_message = run_anchor_step_training(config)
        elif config.method == "trajectory":
            result_message = run_anchor_trajectory_training(config)
        elif config.method == "side":
            result_message = run_anchor_side_training(config)
        else:
            raise ValueError(f"Unknown method: {config.method}")
        print("\n=== Training Complete ===")
        print(result_message)
    except Exception as e:
        print(f"\n[!] Training failed with error: {e}")


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
    visualizer = LossVisualizer(config)
    
    print(f"Starting anchor training for {config.iterations} iterations on {config.device}...")
    
    # 5. The Training Loop
    # We leave the models in eval mode since they are frozen, though the gradients 
    # will flow back to the anchor_embeds leaf tensor.
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
        visualizer.record(loss.item())
        
        # Optional: Print loss every 50 steps
        if i % 50 == 0 or i == config.iterations - 1:
            tqdm.write(f"Iteration {i:04d} | Timestep: {step_result.timestep_index:03d} | Loss: {loss.item():.4f}")

    visualizer.save_data()
    visualizer.save_plot()

    save_anchor_embeddings(context, config)
    
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
    visualizer = LossVisualizer(config)

    print(f"Starting anchor training for {config.iterations} iterations on {config.device}...")

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
            visualizer.record(loss_val)
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
    visualizer.save_data()
    visualizer.save_plot()

    # 6. Save final optimized weights
    save_anchor_embeddings(context, config)

    return f"Success! Anchor training completed {config.iterations} iterations. Final avg loss: {last_loss:.4f}."

def run_anchor_side_training(config: AnchorConfig) -> str:
    """
    Executes the anchor embedding optimization loop.
    
    Args:
        config (AnchorConfig): The configuration for the training run.
        
    Returns:
        str: A status message indicating completion and final loss.
    """
    trainer = SDAnchorSideTrainer(config)
        
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
    
    visualizer = LossVisualizer(config)
    
    print(f"Starting anchor training for {config.iterations} iterations on {config.device}...")
    
    # 5. The Training Loop
    # We leave the models in eval mode since they are frozen, though the gradients 
    # will flow back to the anchor_embeds leaf tensor.
    for i in tqdm(range(config.iterations), desc="Optimizing Anchor Embeds"):
        optimizer.zero_grad()
        
        # Forward pass / get step result
        is_even = i % 2 == 0
        step_result = trainer.training_step(pipe, context, config, is_even)
        
        # Format the timestep index as a batched tensor for compute_loss
        t_tensor = torch.tensor([step_result.timestep_index], device=config.device)
        
        # Compute loss
        if is_even:
            loss_tensor = trainer.compute_dw_loss(
                predicted_noise_target=step_result.target_noise,
                predicted_noise_anchor=step_result.anchor_noise,
                t=t_tensor
            )
        else:
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
        visualizer.record(loss.item())
        
        # Optional: Print loss every 50 steps
        if i % 50 == 0 or i == config.iterations - 1:
            tqdm.write(f"Iteration {i:04d} | Timestep: {step_result.timestep_index:03d} | Loss: {loss.item():.4f}")

    visualizer.save_data()
    visualizer.save_plot()
    
    # Save final optimized weights
    saved_at = save_anchor_embeddings(context, config)
    
    # 6. Return Completion Status
    return f"Success! Anchor training completed {config.iterations} iterations. Final loss: {loss.item():.4f}."
