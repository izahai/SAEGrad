import os
import torch
import matplotlib.pyplot as plt

from anchor_train.config import AnchorConfig


class LossVisualizer:
    """Handles tracking, exporting, and plotting training loss history using a configuration object."""
    
    def __init__(self, config: AnchorConfig, output_dir: str = "visualization"):
        self.iterations = config.iterations
        self.output_dir = output_dir
        self.loss_history = []
        
        # Pull prompt configurations from the config object
        target_prompt = config.target_prompt
        
        # Generate a clean, safe filename prefix based on the prompt
        safe_prompt = "".join([c if c.isalnum() else "_" for c in target_prompt[:20]])
        self.file_prefix = f"{config.method}_t_{config.train_till_timestep}_{safe_prompt}_steps{self.iterations}"
        
        # Truncate prompt for the plot title presentation
        self.display_prompt = f"{target_prompt[:40]}..." if len(target_prompt) > 40 else target_prompt

    def record(self, loss_value: float):
        """Records a single iteration's loss value."""
        self.loss_history.append(loss_value)

    def save_data(self):
        """Saves the raw loss history array as a PyTorch file."""
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"loss_values_{self.file_prefix}.pt")
        torch.save({"loss_history": self.loss_history}, save_path)
        print(f"Loss values successfully saved to: {save_path}")

    def save_plot(self):
        """Generates and saves the loss curve matplotlib plot."""
        os.makedirs(self.output_dir, exist_ok=True)
        plot_path = os.path.join(self.output_dir, f"loss_curve_{self.file_prefix}.png")
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, color="royalblue", alpha=0.8, label="Loss")
        plt.title(f"Anchor Optimization Loss Curve\nPrompt: '{self.display_prompt}'")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Loss visualization plot saved completely to: {plot_path}")