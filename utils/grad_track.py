import os
import warnings
from typing import Dict

import torch
import torch.nn

class GradientTracker:
    def __init__(self, layer_names: list[str]):
        self.layer_names = layer_names
        # Structure: {layer_name: [step_0_grad_vector, step_1_grad_vector, ...]}
        self.gradient_history: Dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
        self.hooks = []

    def register_hooks(self, model: torch.nn.Module):
        # Create a mapping of named modules for quick lookup
        named_modules = dict(model.named_modules())
        
        for name in self.layer_names:
            if name not in named_modules:
                warnings.warn(f"Layer '{name}' not found in the model. Skipping gradient tracking.")
                continue
                
            module = named_modules[name]
            
            # Define hook to capture input activations and error signals (grad_output)
            def create_hook(layer_name):
                def hook(mod, grad_input, grad_output):
                    # grad_output[0] is the error signal delta
                    # mod.saved_input is cached from a forward hook or extracted via custom setup.
                    # Alternatively, PyTorch calculates the direct weight gradient for us:
                    if mod.weight.grad is not None:
                        # Flatten the 2D gradient matrix into a 1D vector
                        grad_vector = mod.weight.grad.detach().cpu().flatten()
                        self.gradient_history[layer_name].append(grad_vector)
                return hook

            # Using full weight gradient captures exactly (input activation * error signal)
            # resolved over the batch dimension
            handle = module.register_backward_hook(create_hook(name))
            self.hooks.append(handle)

    def save_history(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for name, grads in self.gradient_history.items():
            if not grads:
                continue
            # Stack all steps together: [num_iterations, weight_dim]
            stacked_grads = torch.stack(grads)
            sanitized_name = name.replace(".", "_")
            save_file = os.path.join(save_dir, f"grads_{sanitized_name}.pt")
            torch.save(stacked_grads, save_file)
            print(f"Saved gradient history for {name} to {save_file}")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()