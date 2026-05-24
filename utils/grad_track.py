import os
import warnings
from typing import Dict

import torch
import torch.nn

class GradientTracker:
    def __init__(self, layer_names: list[str]):
        self.layer_names = layer_names
        # Maps layer -> list of concatenated [activation, error_signal] vectors
        self.gradient_history: Dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
        self.hooks = []
        # Temporary cache to pass activations from forward to backward pass
        self._current_activations = {}

    def register_hooks(self, model: torch.nn.Module):
        named_modules = dict(model.named_modules())
        
        for name in self.layer_names:
            # --- Clean trailing weights/biases if provided ---
            clean_name = name
            if name.endswith(".weight") or name.endswith(".bias"):
                clean_name = ".".join(name.split(".")[:-1])
            
            if clean_name not in named_modules:
                warnings.warn(f"Layer '{name}' not found in the model. Skipping tracker.")
                continue
                
            module = named_modules[clean_name]
            
            # 1. Forward hook to capture input activation (X) reduced to the feature dimension
            def create_forward_hook(layer_name):
                def forward_hook(mod, m_input, m_output):
                    X = m_input[0].detach().cpu()
                    # X shape is typically [Batch, Tokens, Input_Channels] (e.g., [1, 288, 1280])
                    # Average across Batch (dim 0) and Tokens (dim 1) to isolate the 1280 features
                    if X.ndim == 3:
                        X_reduced = X.mean(dim=(0, 1)) 
                    elif X.ndim == 2:
                        X_reduced = X.mean(dim=0)
                    else:
                        X_reduced = X.flatten() # Fallback
                        
                    self._current_activations[layer_name] = X_reduced
                return forward_hook

            # 2. Backward hook to capture error signal (delta) reduced to the feature dimension
            def create_backward_hook(layer_name):
                def backward_hook(mod, grad_input, grad_output):
                    if grad_output[0] is not None:
                        delta = grad_output[0].detach().cpu()
                        # grad_output shape matches the layer's output: [Batch, Tokens, Output_Channels] (e.g., [1, 288, 10240])
                        # Average across Batch and Tokens to isolate the 10240 features
                        if delta.ndim == 3:
                            delta_reduced = delta.mean(dim=(0, 1))
                        elif delta.ndim == 2:
                            delta_reduced = delta.mean(dim=0)
                        else:
                            delta_reduced = delta.flatten()
                            
                        X_reduced = self._current_activations.get(layer_name)
                        
                        if X_reduced is not None:
                            # Both are now 1D vectors: X_reduced is [1280], delta_reduced is [10240]
                            # Concatenate them into a single vector of size [11520]
                            combined_vector = torch.cat([delta_reduced, X_reduced]) 
                            self.gradient_history[layer_name].append(combined_vector)
                return backward_hook

            # Register both hooks safely using PyTorch's modern full backward hook
            self.hooks.append(module.register_forward_hook(create_forward_hook(name)))
            self.hooks.append(module.register_full_backward_hook(create_backward_hook(name)))

    def save_history(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for name, vectors in self.gradient_history.items():
            if not vectors:
                continue
            
            # Since sequence lengths or dynamic batches could theoretically shift sizes, 
            # stacking requires uniform vectors. If they are identical, stack them into a 2D matrix.
            try:
                stacked_data = torch.stack(vectors)
            except RuntimeError:
                # Fallback if dimensions vary across dynamic steps
                stacked_data = vectors

            sanitized_name = name.replace(".", "_")
            save_file = os.path.join(save_dir, f"components_{sanitized_name}.pt")
            torch.save(stacked_data, save_file)
            print(f"Saved memory-efficient components for {name} to {save_file}")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self._current_activations.clear()