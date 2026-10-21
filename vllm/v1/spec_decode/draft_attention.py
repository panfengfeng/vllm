# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve draft attention requirements before constructing model layers."""

from typing import Any

_SLIDING_ATTENTION = "sliding_attention"


def _dflash_layer_causal(config: Any, layer_idx: int) -> bool:
    """Resolve explicit causality before falling back to legacy layer defaults."""
    is_causal = getattr(config, "is_causal", None)
    if is_causal is not None:
        return bool(is_causal)
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return bool(override)
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return False
    return layer_types[layer_idx] == _SLIDING_ATTENTION


def dflash_has_any_non_causal(config: Any) -> bool:
    """Whether the draft needs a non-causal-capable backend, resolved from config
    (config mirror of the model's ``get_draft_attn_causal``, usable pre-build)."""
    return not all(
        _dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)
    )
