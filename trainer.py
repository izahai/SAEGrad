from __future__ import annotations

import gc
import os
import random
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from diffusers import FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps as retrieve_flux_timesteps
except ModuleNotFoundError:
    FluxPipeline = None
    StableDiffusionPipeline = None
    StableDiffusionXLPipeline = None
    calculate_shift = None
    retrieve_flux_timesteps = None

from utils.esd_checkpoint import save_esd_checkpoint
from utils.grad_track import GradientTracker

try:
    from utils.sd_utils import esd_sd_call
except ModuleNotFoundError:
    esd_sd_call = None

TARGET_MODULE_TYPES = {
    "Linear",
    "Conv2d",
    "LoRACompatibleLinear",
    "LoRACompatibleConv",
}

@dataclass
class ESDConfig:
    family: str
    base_model_id: str
    erase_concept: str
    erase_from: Optional[str]
    train_method: str
    iterations: int
    lr: Optional[float]
    negative_guidance: float
    num_inference_steps: int
    guidance_scale: float
    batch_size: int
    resolution: Optional[int]
    save_path: str
    device: str = "cuda:0"
    torch_dtype: torch.dtype = torch.bfloat16
    inference_guidance_scale: Optional[float] = None
    max_sequence_length: int = 77
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    target_layers: Optional[list[str]] = None
    save_gradient: Optional[list[str]] = None

    @property
    def erase_from_effective(self) -> str:
        return self.erase_from if self.erase_from is not None else self.erase_concept


@dataclass
class StepResult:
    model_pred: torch.Tensor
    target: torch.Tensor
    timestep_index: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class PreparedComponent:
    def __init__(
        self,
        component: torch.nn.Module,
        student_params: "OrderedDict[str, torch.nn.Parameter]",
        base_params: "OrderedDict[str, torch.nn.Parameter]",
    ) -> None:
        self.component = component
        self.student_params = student_params
        self.base_params = base_params

    def use_base(self) -> None:
        for name, param in self.base_params.items():
            set_module(self.component, name, param)

    def use_student(self) -> None:
        for name, param in self.student_params.items():
            set_module(self.component, name, param)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.student_params.values()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().contiguous()
            for name, param in self.student_params.items()
        }


def set_module(module: torch.nn.Module, module_name, new_module) -> None:
    if isinstance(module_name, str):
        module_name = module_name.split(".")

    if len(module_name) == 1:
        setattr(module, module_name[0], new_module)
        return

    child_module = getattr(module, module_name[0])
    set_module(child_module, module_name[1:], new_module)


def resolve_default_resolution(pipe, fallback_component: Optional[str] = None) -> int:
    default_sample_size = getattr(pipe, "default_sample_size", None)
    if default_sample_size is None and fallback_component is not None:
        component = getattr(pipe, fallback_component)
        default_sample_size = component.config.sample_size

    if isinstance(default_sample_size, (tuple, list)):
        default_sample_size = default_sample_size[0]

    return int(default_sample_size) * pipe.vae_scale_factor


def select_parameter_names(
    component: torch.nn.Module,
    module_selector,
) -> list[str]:
    selected_names = []
    seen = set()
    for module_name, module in component.named_modules():
        if module.__class__.__name__ not in TARGET_MODULE_TYPES:
            continue
        if not module_selector(module_name):
            continue

        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if full_name in seen:
                continue
            seen.add(full_name)
            selected_names.append(full_name)

    return selected_names


def prepare_component(
    component: torch.nn.Module,
    parameter_names: list[str],
    trainable_dtype: Optional[torch.dtype] = None,
) -> PreparedComponent:
    if not parameter_names:
        raise ValueError("No trainable parameters were selected for this configuration.")

    named_params = dict(component.named_parameters())
    component.requires_grad_(False)

    student_params: "OrderedDict[str, torch.nn.Parameter]" = OrderedDict()
    base_params: "OrderedDict[str, torch.nn.Parameter]" = OrderedDict()
    for parameter_name in parameter_names:
        if parameter_name not in named_params:
            raise KeyError(f"Parameter '{parameter_name}' was not found on the target component.")

        param = named_params[parameter_name]
        if trainable_dtype is not None and param.dtype != trainable_dtype:
            student_param = torch.nn.Parameter(
                param.detach().to(dtype=trainable_dtype).clone(),
                requires_grad=True,
            )
        else:
            param.requires_grad_(True)
            student_param = param

        student_params[parameter_name] = student_param
        base_params[parameter_name] = torch.nn.Parameter(param.detach().clone(), requires_grad=False)

    return PreparedComponent(component, student_params, base_params)


def sanitize_checkpoint_name(text: str) -> str:
    return text.replace(" ", "_")


def clear_device_cache(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _suppress_transformers_pipeline_load_noise() -> Iterator[None]:
    """Hide benign Transformers 5.x chatter during diffusers `from_pretrained` (SD/SDXL/…).

    The "… LOAD REPORT" tables are emitted at WARNING on the **`transformers.modeling_utils`**
    logger (see `log_state_dict_report(..., logger=logger)` in Transformers), not on
    `transformers.utils.loading_report`. Lowering the whole library verbosity for this
    block reliably suppresses those messages plus legacy CLIP config warnings.
    """
    try:
        from transformers import logging as transformers_logging
    except ImportError:
        yield
        return
    previous = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(previous)


def offload_modules_to_cpu(device: str, *modules: Optional[torch.nn.Module]) -> None:
    for module in modules:
        if module is not None:
            module.to("cpu")
    clear_device_cache(device)
    gc.collect()


def make_sampling_generator(device: str, seed: int) -> torch.Generator:
    target_device = torch.device(device)
    if target_device.type == "cuda" and torch.cuda.is_available():
        return torch.Generator(device=target_device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


class BaseESDAdapter:
    family = ""
    component_attr = ""
    default_base_model_id = ""
    default_save_path = ""

    def normalize_train_method(self, train_method: str) -> str:
        raise NotImplementedError

    def default_lr_for_method(self, train_method: str) -> float:
        raise NotImplementedError

    def load_pipeline(self, config: ESDConfig):
        raise NotImplementedError

    def trainable_param_dtype(self, config: ESDConfig) -> Optional[torch.dtype]:
        return None

    def select_parameter_names(
        self,
        component: torch.nn.Module,
        train_method: str,
        config: Optional[ESDConfig] = None,
    ) -> list[str]:
        raise NotImplementedError

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        raise NotImplementedError

    def training_step(
        self,
        pipe,
        prepared: PreparedComponent,
        context: Dict[str, Any],
        config: ESDConfig,
    ) -> StepResult:
        raise NotImplementedError

    def resolve_resolution(self, pipe, config: ESDConfig) -> int:
        if config.resolution is not None:
            return config.resolution
        return resolve_default_resolution(pipe, fallback_component=self.component_attr)

    def resolve_learning_rate(self, config: ESDConfig) -> float:
        if config.lr is not None:
            return config.lr
        return self.default_lr_for_method(config.train_method)

    def build_metadata(self, config: ESDConfig) -> Dict[str, str]:
        metadata = {
            "family": self.family,
            "component": self.component_attr,
            "base_model_id": config.base_model_id,
            "train_method": config.train_method,
            "erase_concept": config.erase_concept,
            "erase_from": config.erase_from or "",
            "num_inference_steps": str(config.num_inference_steps),
            "guidance_scale": str(config.guidance_scale),
            "negative_guidance": str(config.negative_guidance),
            "batch_size": str(config.batch_size),
        }
        if config.resolution is not None:
            metadata["resolution"] = str(config.resolution)
        if config.target_layers:
            metadata["target_layers"] = ",".join(config.target_layers)
        return metadata

    def build_checkpoint_path(self, config: ESDConfig) -> str:
        method_suffix = config.train_method.replace("-", "")
        filename = (
            f"esd-{sanitize_checkpoint_name(config.erase_concept)}"
            f"-from-{sanitize_checkpoint_name(config.erase_from_effective)}"
            f"-{method_suffix}.safetensors"
        )
        return os.path.join(config.save_path, filename)

    def create_prepared_component(self, pipe, train_method: str, config: ESDConfig) -> PreparedComponent:
        component = getattr(pipe, self.component_attr)
        parameter_names = self.select_parameter_names(component, train_method, config)
        return prepare_component(component, parameter_names, trainable_dtype=self.trainable_param_dtype(config))


class StableDiffusionESDAdapter(BaseESDAdapter):
    family = "sd"
    component_attr = "unet"
    default_base_model_id = "CompVis/stable-diffusion-v1-4"
    default_save_path = "esd-models/sd/"

    def normalize_train_method(self, train_method: str) -> str:
        aliases = {
            "xattn": "esd-x",
            "noxattn": "esd-u",
            "full": "esd-all",
            "xattn-strict": "esd-x-strict",
            "selfattn": "selfattn",
            "esd-x": "esd-x",
            "esd-u": "esd-u",
            "esd-all": "esd-all",
            "esd-x-strict": "esd-x-strict",
            "specific-layer": "specific-layer",
        }
        normalized = aliases.get(train_method)
        if normalized is None:
            raise ValueError(f"Unsupported SD train method: {train_method}")
        return normalized

    def default_lr_for_method(self, train_method: str) -> float:
        return 5e-5

    def load_pipeline(self, config: ESDConfig):
        if StableDiffusionPipeline is None:
            raise ModuleNotFoundError("diffusers is required to load Stable Diffusion pipelines.")
        pipe = StableDiffusionPipeline.from_pretrained(
            config.base_model_id,
            torch_dtype=config.torch_dtype,
            use_safetensors=True,
        ).to(config.device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        if pipe.safety_checker is not None:
            pipe.safety_checker.requires_grad_(False)
        return pipe

    def select_parameter_names(
        self,
        component: torch.nn.Module,
        train_method: str,
        config: Optional[ESDConfig] = None,
    ) -> list[str]:
        target_layers = config.target_layers if config is not None else None

        def selector(module_name: str) -> bool:
            if train_method == "esd-x":
                return "attn2" in module_name
            if train_method == "esd-u":
                return "attn2" not in module_name
            if train_method == "esd-all":
                return True
            if train_method == "esd-x-strict":
                return "attn2.to_k" in module_name or "attn2.to_v" in module_name
            if train_method == "selfattn":
                return "attn1" in module_name
            if train_method == "specific-layer":
                if not target_layers:
                    raise ValueError("train_method='specific-layer' requires config.target_layers to be set.")
                return any(
                    module_name == target_layer or module_name.startswith(f"{target_layer}.")
                    for target_layer in target_layers
                )
            return False

        return select_parameter_names(component, selector)

    def prepare_context(self, pipe, config: ESDConfig) -> Dict[str, Any]:
        resolution = self.resolve_resolution(pipe, config)
        with torch.no_grad():
            erase_embeds, null_embeds = pipe.encode_prompt(
                prompt=config.erase_concept,
                device=config.device,
                num_images_per_prompt=config.batch_size,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
            erase_embeds = erase_embeds.to(config.device)
            null_embeds = null_embeds.to(config.device)

            erase_from_embeds = None
            if config.erase_from is not None:
                erase_from_embeds, _ = pipe.encode_prompt(
                    prompt=config.erase_from,
                    device=config.device,
                    num_images_per_prompt=config.batch_size,
                    do_classifier_free_guidance=False,
                    negative_prompt="",
                )
                erase_from_embeds = erase_from_embeds.to(config.device)

            timestep_cond = None
            if pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(config.guidance_scale - 1).repeat(config.batch_size)
                timestep_cond = pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor,
                    embedding_dim=pipe.unet.config.time_cond_proj_dim,
                ).to(device=config.device, dtype=config.torch_dtype)

        offload_modules_to_cpu(config.device, pipe.vae, pipe.text_encoder, pipe.safety_checker)

        return {
            "resolution": resolution,
            "erase_embeds": erase_embeds,
            "null_embeds": null_embeds,
            "erase_from_embeds": erase_from_embeds,
            "sample_prompt_embeds": erase_embeds if erase_from_embeds is None else erase_from_embeds,
            "sample_negative_prompt_embeds": null_embeds,
            "student_prompt_embeds": erase_embeds if erase_from_embeds is None else erase_from_embeds,
            "timestep_cond": timestep_cond,
        }

    def training_step(self, pipe, prepared: PreparedComponent, context: Dict[str, Any], config: ESDConfig) -> StepResult:
        if esd_sd_call is None:
            raise ModuleNotFoundError("diffusers is required to run Stable Diffusion ESD training steps.")
        run_till_timestep = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)

        prepared.use_base()
        prepared.component.eval()
        with torch.no_grad():
            xt = esd_sd_call(
                pipe,
                prompt_embeds=context["sample_prompt_embeds"],
                negative_prompt_embeds=context["sample_negative_prompt_embeds"],
                num_images_per_prompt=1,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                run_till_timestep=run_till_timestep,
                generator=make_sampling_generator(config.device, seed),
                output_type="latent",
                height=context["resolution"],
                width=context["resolution"],
            ).images

            timestep = pipe.scheduler.timesteps[run_till_timestep]
            noise_pred_erase = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=context["erase_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred_null = prepared.component(
                xt,
                timestep,
                encoder_hidden_states=context["null_embeds"],
                timestep_cond=context["timestep_cond"],
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0]

            if context["erase_from_embeds"] is not None:
                noise_pred_erase_from = prepared.component(
                    xt,
                    timestep,
                    encoder_hidden_states=context["erase_from_embeds"],
                    timestep_cond=context["timestep_cond"],
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                noise_pred_erase_from = noise_pred_erase

        prepared.use_student()
        prepared.component.train()
        model_pred = prepared.component(
            xt,
            timestep,
            encoder_hidden_states=context["student_prompt_embeds"],
            timestep_cond=context["timestep_cond"],
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]

        target = noise_pred_erase_from - config.negative_guidance * (noise_pred_erase - noise_pred_null)
        return StepResult(model_pred=model_pred, target=target, timestep_index=run_till_timestep)


ADAPTERS = {
    "sd": StableDiffusionESDAdapter(),
}


def get_adapter(family: str) -> BaseESDAdapter:
    try:
        return ADAPTERS[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported ESD family: {family}") from exc


def run_esd_training(config: ESDConfig) -> str:
    adapter = get_adapter(config.family)
    config.train_method = adapter.normalize_train_method(config.train_method)
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    with _suppress_transformers_pipeline_load_noise():
        pipe = adapter.load_pipeline(config)
    pipe.set_progress_bar_config(disable=True)
    component = getattr(pipe, adapter.component_attr)
    if config.gradient_checkpointing and hasattr(component, "enable_gradient_checkpointing"):
        component.enable_gradient_checkpointing()

    prepared = adapter.create_prepared_component(pipe, config.train_method, config)
    prepared.use_student()
    
    
    # --- Track Gradients ---
    tracker = None
    if config.save_gradient:
        tracker = GradientTracker(config.save_gradient)
        tracker.register_hooks(prepared.component)

    learning_rate = adapter.resolve_learning_rate(config)
    optimizer = torch.optim.Adam(prepared.parameters(), lr=learning_rate)
    context = adapter.prepare_context(pipe, config)

    pbar = tqdm(range(config.iterations), desc=f"Training ESD ({adapter.family})")
    for _ in pbar:
        optimizer.zero_grad(set_to_none=True)
        step_result = adapter.training_step(pipe, prepared, context, config)
        loss = F.mse_loss(step_result.model_pred.float(), step_result.target.float())
        loss.backward()
        optimizer.step()

        postfix = {"esd_loss": f"{loss.item():.4f}", "timestep": step_result.timestep_index}
        postfix.update({key: str(value) for key, value in step_result.metrics.items()})
        pbar.set_postfix(postfix)
        
    # --- Save Gradients and Clean Up ---
    if tracker:
        tracker.save_history(config.save_path, config.erase_concept)
        tracker.remove_hooks()

    prepared.use_student()
    checkpoint_path = adapter.build_checkpoint_path(config)
    save_esd_checkpoint(prepared.state_dict(), checkpoint_path, metadata=adapter.build_metadata(config))
    return checkpoint_path
