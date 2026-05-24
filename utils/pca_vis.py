import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_multiple_gradients_pca(save_dir: str, file_pattern: str = "components_*.pt"):
    """
    Loads all matching gradient history tensors from save_dir, runs a unified PCA 
    to reduce them to a shared 2D space, and plots them to observe clustering.
    """
    # 1. Find all matching tensor files
    search_path = os.path.join(save_dir, file_pattern)
    file_paths = sorted(glob.glob(search_path))
    
    if not file_paths:
        raise FileNotFoundError(f"No files found matching pattern: {search_path}")
        
    print(f"Found {len(file_paths)} layer tensor files to analyze.")

    all_data_list = []
    layer_metadata = [] # Tracks which rows belong to which layer
    
    # 2. Load and stack all data into one giant matrix
    for path in file_paths:
        filename = os.path.basename(path)
        # Extract clean layer name from filename
        layer_name = filename.replace("components_", "").replace(".pt", "")
        
        print(f"Loading: {filename}...")
        data_tensor = torch.load(path, map_location="cpu")
        
        if isinstance(data_tensor, list):
            data_tensor = torch.stack(data_tensor)
            
        data_np = data_tensor.float().numpy()
        
        # Guard rail: ensure it's a 2D matrix [steps, features]
        if data_np.ndim != 2:
            print(f"Skipping {filename}: expected 2D matrix shape, got {data_np.shape}")
            continue
            
        num_steps, num_features = data_np.shape
        
        # Append to our global dataset matrix
        all_data_list.append(data_np)
        
        # Keep track of where this layer lives in the stacked matrix
        layer_metadata.append({
            "name": layer_name,
            "num_steps": num_steps,
            "num_features": num_features
        })

    if not all_data_list:
        print("No valid 2D tensor data loaded. Exiting.")
        return

    # Combine everything into shape [Total_Layers * 200, 11520]
    combined_matrix = np.vstack(all_data_list)
    print(f"\nUnified dataset matrix built with shape: {combined_matrix.shape}")

    # 3. Fit a single, joint PCA space for absolute comparison
    print("Fitting unified PCA model...")
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined_matrix)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA complete! Total variance explained -> PC1: {explained_variance[0]:.2%}, PC2: {explained_variance[1]:.2%}")

    # 4. Plotting the results
    plt.figure(figsize=(12, 9), dpi=120)
    
    # Available color maps for distinct layers (fades from dark to light/vibrant)
    colormaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'PuRd', 'GnBu']
    
    current_row_idx = 0
    
    # Unpack each layer sequence out of the shared PCA coordinates
    for i, meta in enumerate(layer_metadata):
        start_idx = current_row_idx
        end_idx = start_idx + meta["num_steps"]
        current_row_idx = end_idx # increment pointer
        
        # Isolate this layer's 2D path
        layer_2d = combined_2d[start_idx:end_idx]
        steps = np.arange(meta["num_steps"])
        
        # Select a cyclic colormap choice from our list
        cmap_choice = colormaps[i % len(colormaps)]
        
        # Draw sequential trajectory connecting lines
        plt.plot(layer_2d[:, 0], layer_2d[:, 1], linestyle='-', alpha=0.25, color='gray')
        
        # Scatter points with a step gradient (Dark = Step 0, Bright/Saturated = Step 200)
        scatter = plt.scatter(
            layer_2d[:, 0], 
            layer_2d[:, 1], 
            c=steps, 
            cmap=cmap_choice, 
            s=40, 
            edgecolors='black', 
            linewidths=0.3,
            alpha=0.85,
            label=meta["name"]
        )
        
        # Highlight start and end landmarks for this specific cluster trajectory
        plt.scatter(layer_2d[0, 0], layer_2d[0, 1], marker='X', s=90, color='black', alpha=0.7)
        plt.scatter(layer_2d[-1, 0], layer_2d[-1, 1], marker='o', s=80, facecolors='none', edgecolors='red', linewidths=1.5)

    # Styling and Cluster interpretation
    plt.title("Unified PCA Space: Layer Clustering & Gradient Trajectories", fontsize=14, pad=15)
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1%})", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1%})", fontsize=11)
    
    # Use a custom legend handling strings smoothly
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Tracked Layers\n(Color fades Dark → Light over time)", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Add an explanatory note on the plot
    plt.figtext(0.15, 0.02, "* Note: 'X' marks training Step 0. Red circles mark training Step 200.", 
                fontsize=9, style='italic', color='dimgray')
    
    plt.tight_layout()
    
    # Save unified map
    output_img_path = os.path.join(save_dir, "unified_layer_clusters_pca.png")
    plt.savefig(output_img_path, bbox_inches='tight')
    print(f"\n[Success] Unified visualization plot saved to: {output_img_path}")
    plt.show()

if __name__ == "__main__":
    # Point this to your actual training logs directory
    TARGET_DIR = "esd-models/sd/"
    
    try:
        visualize_multiple_gradients_pca(save_dir=TARGET_DIR)
    except Exception as e:
        print(f"\nExecution failed: {e}")