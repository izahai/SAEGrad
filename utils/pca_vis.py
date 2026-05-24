import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_gradient_pca(save_dir: str, layer_name: str = "mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj_weight"):
    """
    Loads the saved [200, 11520] tensor, performs PCA down to 2 dimensions,
    and plots the trajectory across training steps.
    """
    # 1. Locate and load the saved PyTorch tensor
    file_path = os.path.join(save_dir, f"components_{layer_name}.pt")
    
    if not os.path.exists(file_path):
        # Fallback search if name formatting differs slightly
        possible_files = glob.glob(os.path.join(save_dir, f"*{layer_name}*.pt"))
        if possible_files:
            file_path = possible_files[0]
        else:
            raise FileNotFoundError(f"Could not find saved component file at {file_path}. "
                                    f"Please check your save_path or layer name.")

    print(f"Loading data from: {file_path}")
    # Load tensor and convert to numpy array
    data_tensor = torch.load(file_path, map_location="cpu")
    
    if isinstance(data_tensor, list):
        # If the tracker fell back to a list of tensors instead of a stacked matrix
        data_tensor = torch.stack(data_tensor)
        
    data_np = data_tensor.float().numpy()
    
    print(f"Loaded tensor shape: {data_np.shape}") # Should be [200, 11520]
    
    # 2. Apply PCA to reduce from 11520 dimensions down to 2 dimensions
    n_components = 2
    pca = PCA(n_components=n_components)
    data_2d = pca.fit_transform(data_np)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA complete. Variance explained by PC1: {explained_variance[0]:.2%}, PC2: {explained_variance[1]:.2%}")

    # 3. Plotting the 2D Trajectory
    num_steps = data_np.shape[0]
    steps = np.arange(num_steps) # Array from 0 to 199 for color mapping

    plt.figure(figsize=(10, 8), dpi=120)
    
    # Draw line tracing the trajectory through optimization space
    plt.plot(data_2d[:, 0], data_2d[:, 1], color='gray', linestyle='-', alpha=0.3, zorder=1)
    
    # Scatter plot with 'viridis' colormap to represent time progression
    scatter = plt.scatter(
        data_2d[:, 0], 
        data_2d[:, 1], 
        c=steps, 
        cmap='viridis', 
        s=45, 
        edgecolors='black', 
        linewidths=0.5,
        zorder=2
    )
    
    # Highlight Start and End points explicitly
    plt.scatter(data_2d[0, 0], data_2d[0, 1], c='crimson', marker='X', s=150, label='Start (Step 0)', zorder=3)
    plt.scatter(data_2d[-1, 0], data_2d[-1, 1], c='cyan', marker='P', s=150, label='End (Step 200)', zorder=3)

    # Cosmetics & Labels
    plt.title(f"2D PCA Trajectory of Activations & Gradients\nLayer: {layer_name}", fontsize=12, pad=15)
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1%})", fontsize=10)
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1%})", fontsize=10)
    
    # Colorbar to decode which color matches which training iteration
    cbar = plt.colorbar(scatter)
    cbar.set_label('Training Optimization Step', fontsize=10)
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save chart and display
    output_img_path = os.path.join(save_dir, f"pca_{layer_name}.png")
    plt.savefig(output_img_path)
    print(f"Successfully saved 2D visualization plot to: {output_img_path}")
    plt.show()

if __name__ == "__main__":
    # Replace this string path with your actual config.save_path folder
    # e.g., "esd-models/sd/"
    OUTPUT_DIRECTORY = "esd-models/sd/" 
    
    # Run the function
    try:
        visualize_gradient_pca(save_dir=OUTPUT_DIRECTORY)
    except Exception as e:
        print(f"Error executing visualization: {e}")
        print("\nMake sure your pipeline run finished completely and saved the '.pt' files into your specified save directory.")