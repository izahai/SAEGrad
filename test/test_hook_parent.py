import pytest
import torch
from diffusers import UNet2DConditionModel
from mesd.hook import parent_module

@pytest.fixture(scope="module")
def sd_unet():
    # Load SD 1.4 UNet (CPU is enough for structural tests)
    return UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    )

def test_parent_module_shallow(sd_unet):
    # pname = "conv_in" -> parent should be the unet itself
    parent = parent_module(sd_unet, "conv_in")
    assert parent is sd_unet
    assert hasattr(parent, "conv_in")

def test_parent_module_nested_list(sd_unet):
    # Testing digit indexing: down_blocks[0].resnets[0]
    pname = "down_blocks.0.resnets.0.conv1"
    parent = parent_module(sd_unet, pname)
    
    expected_parent = sd_unet.down_blocks[0].resnets[0]
    assert parent is expected_parent
    assert hasattr(parent, "conv1")

def test_parent_module_cross_attention(sd_unet):
    # Deeply nested attention projection
    pname = "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k"
    parent = parent_module(sd_unet, pname)
    
    expected_parent = sd_unet.down_blocks[1].attentions[0].transformer_blocks[0].attn2
    assert parent is expected_parent
    assert hasattr(parent, "to_k")

def test_parent_module_invalid_path(sd_unet):
    with pytest.raises(RuntimeError, match="Couldn't find child module"):
        parent_module(sd_unet, "down_blocks.0.non_existent_layer.weight")

def test_parent_module_missing_leaf(sd_unet):
    # Path is valid until the last element
    with pytest.raises(AssertionError):
        parent_module(sd_unet, "down_blocks.0.resnets.0.wrong_leaf")

def test_parent_module_parameter_leaf(sd_unet):
    # Testing access where the leaf is a weight parameter, not a submodule
    pname = "conv_in.weight"
    parent = parent_module(sd_unet, pname)
    
    assert parent is sd_unet.conv_in
    assert isinstance(getattr(parent, "weight"), torch.nn.Parameter)
    
def test_parent_module_sequential_and_dict():
    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = torch.nn.Sequential(torch.nn.Linear(10, 10))
            self.m_dict = torch.nn.ModuleDict({"layer_a": torch.nn.Linear(5, 5)})

    model = CustomModel()
    
    # Test Sequential indexing
    assert parent_module(model, "seq.0") is model.seq
    
    # Test ModuleDict key access
    assert parent_module(model, "m_dict.layer_a") is model.m_dict
    
def test_parent_module_single_component(sd_unet):
    # Path with no dots should return the model itself as parent
    parent = parent_module(sd_unet, "conv_in")
    assert parent is sd_unet

def test_parent_module_empty_string(sd_unet):
    # Test behavior on empty input
    with pytest.raises(Exception): # Usually IndexError or AssertionError
        parent_module(sd_unet, "")

def test_parent_module_double_dot(sd_unet):
    # Test malformed path strings
    pname = "down_blocks..resnets"
    with pytest.raises(RuntimeError):
        parent_module(sd_unet, pname)
        
def test_parent_module_patching_capability(sd_unet):
    pname = "down_blocks.0.resnets.0.conv1"
    parent = parent_module(sd_unet, pname)
    
    # Simulate a MEND/Edit patch
    original_layer = getattr(parent, "conv1")
    mock_layer = torch.nn.Identity()
    
    setattr(parent, "conv1", mock_layer)
    assert sd_unet.down_blocks[0].resnets[0].conv1 is mock_layer
    
    # Restore (cleanup)
    setattr(parent, "conv1", original_layer)