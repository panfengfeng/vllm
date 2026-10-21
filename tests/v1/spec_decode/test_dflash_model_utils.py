# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.qwen3_dflash import (
    _add_global_draft_layer_exclusions,
)


def test_draft_quant_exclusions_include_global_layer_indices():
    quant_config = SimpleNamespace(
        exclude_modules=[
            "layers.0.mlp_conv*",
            "*layers.4.self_attn.q_proj",
            "layers.88.already_global",
            "lilicorr.layers.0.mlp.0",
        ]
    )

    _add_global_draft_layer_exclusions(quant_config, 88, 5)

    assert "layers.88.mlp_conv*" in quant_config.exclude_modules
    assert "*layers.92.self_attn.q_proj" in quant_config.exclude_modules
    assert quant_config.exclude_modules.count("layers.88.already_global") == 1
    assert "lilicorr.layers.88.mlp.0" not in quant_config.exclude_modules


def test_context_kv_uses_quantized_projection_fallback(monkeypatch):
    from torch import nn

    from vllm.model_executor.models import qwen3_dflash

    class Projection(nn.Module):
        def __init__(self, packed_weight):
            super().__init__()
            self.register_buffer("packed_weight", packed_weight)
            self.quant_method = object()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.packed_weight), None

    monkeypatch.setattr(
        qwen3_dflash.ops,
        "rms_norm",
        lambda output, hidden_states, weight, eps: output.copy_(hidden_states),
    )
    context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    projections = [
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        ),
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        ),
    ]
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        _rms_norm_eps=1e-6,
    )
    layers_attn = [
        SimpleNamespace(
            qkv_proj=projection,
            q_size=2,
            k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        )
        for projection in projections
    ]
    qwen3_dflash.DFlashQwen3Model._build_context_kv_buffers(
        model,
        layers_attn,
        has_bias=False,
    )
    assert model._fused_kv_weight is None
    assert all(not hasattr(projection, "weight") for projection in projections)

    actual_k, actual_v = qwen3_dflash.DFlashQwen3Model._project_context_kv(
        model,
        context_states,
        num_ctx=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
    )

    expected_k = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]
    ).unsqueeze(2)
    expected_v = torch.tensor(
        [[[3.0, -1.0], [7.0, -1.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(2)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert [projection.calls for projection in projections] == [1, 1]
