import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from mesd.hook import parent_module, linear_backward_hook, linear_forward_hook

def test_fake_gradient_hook():
    layer = nn.Linear(320, 1280).cuda()
    layer.zero_grad()
    
    # Register hooks
    layer.register_forward_hook(linear_forward_hook)
    layer.register_full_backward_hook(linear_backward_hook)
    
    dummy_input = torch.randn(2, 16, 320).cuda()
    
    # Forward Pass
    output = layer(dummy_input)
    
    # CHECK 1: Verify forward hook (Check the WEIGHT, not the layer)
    if not hasattr(layer.weight, "__x__"):
        raise RuntimeError("Forward hook failed to set __x__ on layer.weight!")
    
    # Backward Pass
    loss = output.sum()
    loss.backward()
    
    # CHECK 2: Verify backward hook (Check the WEIGHT, not the layer)
    if not hasattr(layer.weight, "__delta__"):
        raise RuntimeError("Backward hook failed to set __delta__ on layer.weight!")
    
    # 6. Manual Gradient Calculation
    # Access from .weight where the hook saved them
    x = layer.weight.__x__        
    delta = layer.weight.__delta__ 
    
    # [Batch, Seq, In] and [Batch, Seq, Out] -> [Out, In]
    manual_grad = torch.einsum('bni,bnj->ji', x, delta)
    
    pytorch_grad = layer.weight.grad
    error = (manual_grad - pytorch_grad).abs().max()
    
    print(f"Max Difference: {error.item()}")
    assert torch.allclose(manual_grad, pytorch_grad, atol=1e-3)
    print("✅ Test Passed!")
    
def test_sd14_gradient_hook():
    print("🚀 Loading Stable Diffusion 1.4 UNet in full FP32...")
    # Force the model to load in float32
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        subfolder="unet",
        torch_dtype=torch.float32
    ).cuda()
    
    target_layer_path = "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2"
    
    # 1. Resolve the layer
    target_layer = parent_module(unet, target_layer_path)
    if isinstance(target_layer, (nn.ModuleList, nn.Sequential)):
        for sub_mod in target_layer:
            if isinstance(sub_mod, nn.Linear):
                target_layer = sub_mod
                break
    
    # 2. Attach hooks
    target_layer.register_forward_hook(linear_forward_hook)
    target_layer.register_full_backward_hook(linear_backward_hook)
    
    # 3. Ensure everything is in FP32
    target_layer.zero_grad()
    unet.float() 

    # 4. Dummy Inputs (Strictly FP32)
    batch_size = 1
    sample = torch.randn(batch_size, 4, 64, 64).cuda().float()
    timestep = torch.tensor([1]).cuda().float()
    encoder_hidden_states = torch.randn(batch_size, 77, 768).cuda().float()

    # 5. Forward & Backward
    print("🏃 Running UNet forward pass (FP32)...")
    output = unet(sample, timestep, encoder_hidden_states).sample
    
    print("🔙 Running UNet backward pass (FP32)...")
    loss = output.sum() # dummy loss
    loss.backward()

    # 6. Extract
    x = target_layer.weight.__x__        # Already float32
    delta = target_layer.weight.__delta__ # Already float32
    
    # 7. Manual Gradient Calculation
    # [B, N, In] @ [B, N, Out] -> [Out, In]
    manual_grad = torch.einsum('bni,bnj->ji', x, delta)
    pytorch_grad = target_layer.weight.grad

    # 8. Final Validation
    error = torch.abs(manual_grad - pytorch_grad).max()
    print(f"📊 Max Difference (Pure FP32): {error.item()}")
    
    # In pure FP32, the difference should be near zero (1e-6 or less)
    assert torch.allclose(manual_grad, pytorch_grad, atol=1e-5), f"Mismatch even in FP32! Error: {error.item()}"
    print("✨ SUCCESS: Manual gradient is identical to PyTorch Autograd in FP32.")

    del unet
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    test_fake_gradient_hook()
    test_sd14_gradient_hook()