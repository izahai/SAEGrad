from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize


@dataclass
class LossCurveVisualizer:
    """Collect and render training loss values with timestep coloring."""

    num_inference_steps: int
    output_dir: str = "visualization"
    file_prefix: str = "esd_single_step"
    loss_history: list[float] = field(default_factory=list)
    timestep_history: list[int] = field(default_factory=list)

    def record(self, loss_value: float, timestep_index: int) -> None:
        """Store one loss value together with its timestep index."""
        self.loss_history.append(float(loss_value))
        self.timestep_history.append(int(timestep_index))

    def extend(self, records: Iterable[tuple[float, int]]) -> None:
        """Append multiple (loss, timestep) pairs."""
        for loss_value, timestep_index in records:
            self.record(loss_value, timestep_index)

    def save_data(self) -> str:
        """Persist the raw loss and timestep history to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"loss_values_{self.file_prefix}.pt")
        torch.save(
            {
                "loss_history": self.loss_history,
                "timestep_history": self.timestep_history,
                "num_inference_steps": self.num_inference_steps,
            },
            save_path,
        )
        return save_path

    def save_plot(self, title: str | None = None) -> str:
        """Render a loss curve where point color tracks the sampled timestep."""
        if not self.loss_history:
            raise ValueError("No loss values were recorded, so there is nothing to plot.")

        os.makedirs(self.output_dir, exist_ok=True)
        plot_path = os.path.join(self.output_dir, f"loss_curve_{self.file_prefix}.png")

        steps = np.arange(len(self.loss_history))
        losses = np.asarray(self.loss_history, dtype=np.float32)
        timesteps = np.asarray(self.timestep_history, dtype=np.float32)

        timestep_max = max(1, int(self.num_inference_steps))
        norm = Normalize(vmin=0, vmax=timestep_max)
        cmap = LinearSegmentedColormap.from_list("timestep_red_green", ["#ff3b30", "#34c759"])

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        ax.plot(steps, losses, color="#3b82f6", alpha=0.35, linewidth=1.5, zorder=1)
        scatter = ax.scatter(
            steps,
            losses,
            c=timesteps,
            cmap=cmap,
            norm=norm,
            s=36,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            zorder=2,
        )

        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("run_till_timestep", rotation=90)
        cbar.set_ticks([0, timestep_max])
        cbar.set_ticklabels(["0", str(timestep_max)])

        ax.set_title(title or "Single-Step ESD Loss Curve", pad=14)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.35)

        lower_label = f"red = 0"
        upper_label = f"green = {self.num_inference_steps}"
        ax.text(
            0.01,
            0.01,
            f"Color scale: {lower_label}  ->  {upper_label}",
            transform=ax.transAxes,
            fontsize=9,
            color="dimgray",
            va="bottom",
        )

        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        return plot_path
