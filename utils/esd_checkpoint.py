import os
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ESD_CHECKPOINT_FORMAT = "erasing-esd-v2"


def save_esd_checkpoint(
    tensors: Dict[str, torch.Tensor],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    serialized_tensors = {}
    for name, tensor in tensors.items():
        tensor_to_save = tensor.detach()
        if tensor_to_save.device.type != "cpu":
            tensor_to_save = tensor_to_save.cpu()
        serialized_tensors[name] = tensor_to_save.contiguous()

    final_metadata = {"format": ESD_CHECKPOINT_FORMAT}
    if metadata is not None:
        final_metadata.update({key: str(value) for key, value in metadata.items() if value is not None})

    save_file(serialized_tensors, filename, metadata=final_metadata)


def load_esd_checkpoint(filename: str, device: str = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    with safe_open(filename, framework="pt", device=device) as handle:
        metadata = handle.metadata() or {}

    tensors = load_file(filename, device=device)
    return tensors, metadata


def infer_checkpoint_component(
    pipe,
    checkpoint_tensors: Dict[str, torch.Tensor],
    metadata_component: Optional[str] = None,
) -> str:
    if metadata_component is not None:
        component = getattr(pipe, metadata_component, None)
        if component is not None:
            return metadata_component

    candidate_components = []
    checkpoint_keys = set(checkpoint_tensors.keys())
    for component_name in ("unet", "transformer"):
        component = getattr(pipe, component_name, None)
        if component is None:
            continue

        overlap = len(checkpoint_keys.intersection(component.state_dict().keys()))
        if overlap > 0:
            candidate_components.append((overlap, component_name))

    if not candidate_components:
        raise ValueError(
            "Could not infer which pipeline component to update. "
            "Pass an explicit component name or save the checkpoint with metadata."
        )

    candidate_components.sort(reverse=True)
    return candidate_components[0][1]


def apply_esd_checkpoint(pipe, filename: str, device: str = "cpu", component_name: Optional[str] = None):
    checkpoint_tensors, metadata = load_esd_checkpoint(filename, device=device)
    resolved_component_name = infer_checkpoint_component(
        pipe,
        checkpoint_tensors,
        metadata_component=component_name or metadata.get("component"),
    )

    component = getattr(pipe, resolved_component_name)
    load_result = component.load_state_dict(checkpoint_tensors, strict=False)
    matched_keys = len(checkpoint_tensors) - len(load_result.unexpected_keys)
    if matched_keys == 0:
        raise ValueError(
            f"Checkpoint '{filename}' did not match any parameters on pipeline.{resolved_component_name}."
        )

    return metadata, resolved_component_name, load_result
