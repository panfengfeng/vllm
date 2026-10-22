# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
) -> ModelState:
    cls = resolve_model_state_cls(vllm_config, model)

    # Reject enable_prompt_embeds for states that would silently ignore it.
    if vllm_config.model_config.enable_prompt_embeds and not cls.supports_prompt_embeds:
        raise ValueError(f"--enable-prompt-embeds not supported with {cls.__name__}.")

    return cls(vllm_config, model, encoder_cache, device)


def resolve_model_state_cls(
    vllm_config: VllmConfig, model: nn.Module
) -> type[ModelState]:
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    config = vllm_config.model_config
    if config.is_encoder_decoder or config.is_hybrid or config.is_attention_free:
        raise ValueError("This build only supports the DeepSeek V4.1 decoder state.")
    get_state_cls = getattr(model, "get_model_state_cls", None)
    if get_state_cls is None or get_state_cls() is not DeepseekV41ModelState:
        raise ValueError("This build requires DeepseekV41ModelState.")
    return DeepseekV41ModelState
