import torch
from diffusers import UNet2DConditionModel

def collect_linear_layers(model_id: str = "CompVis/stable-diffusion-v1-4"):
    """
    Loads the SD 1.4 UNet and identifies all linear layers and their shapes.
    """
    print(f"Loading UNet from {model_id}...")
    try:
        # Loading on CPU by default to save VRAM during inspection
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    except Exception as e:
        print(f"Error: Failed to load model - {e}")
        return None

    linear_info = []
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            # weight.shape is typically [out_features, in_features]
            shape = list(module.weight.shape)
            linear_info.append((name, shape))

    return linear_info

def print_linear_report(layers):
    if not layers:
        print("No linear layers found.")
        return

    print(f"\n{'='*110}")
    print(f"{'SD 1.4 UNet Linear Layer Analysis':^110}")
    print(f"{'='*110}")
    print(f"{'Layer Name':<85} | {'Weight Shape (Out, In)':<20}")
    print(f"{'-'*110}")

    for name, shape in layers:
        print(f"{name:<85} | {str(shape):<20}")

    print(f"{'-'*110}")
    print(f"Total Linear Layers Identified: {len(layers)}")
    print(f"{'='*110}\n")

if __name__ == "__main__":
    layers = collect_linear_layers()
    print_linear_report(layers)