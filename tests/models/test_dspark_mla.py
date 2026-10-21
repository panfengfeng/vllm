# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.dspark import DSparkMarkovHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)
    head.markov_w2.quant_method.process_weights_after_loading(head.markov_w2)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "scale_dtype", [torch.uint8, torch.float8_e8m0fnu, torch.float32]
)
@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_module", "shard_id"),
    [
        ("mtp.0.attn.wq_a.scale", "model.layers.0.attn.fused_wqa_wkv", 0),
        ("mtp.0.attn.wkv.scale", "model.layers.0.attn.fused_wqa_wkv", 1),
        ("mtp.0.main_proj.scale", "model.main_proj", None),
        (
            "mtp.0.ffn.shared_experts.w1.scale",
            "model.layers.0.ffn.shared_experts.gate_up_proj",
            0,
        ),
        (
            "mtp.0.ffn.shared_experts.w2.scale",
            "model.layers.0.ffn.shared_experts.down_proj",
            None,
        ),
    ],
)
def test_v41_dspark_loads_linear_scales(
    monkeypatch, scale_dtype, checkpoint_name, runtime_module, shard_id
):
    """Checkpoint ``.scale`` maps to the quant method's scale parameter and
    loads untouched. MXFP8 block-scale expansion lives in the KMxfp8Static
    loader (see tests/quantization/test_modelopt.py), not in load_weights."""
    from vllm.models.deepseek_v41.nvidia import dspark

    mxfp8 = scale_dtype != torch.float32
    scale_name = "weight_scale" if mxfp8 else "weight_scale_inv"
    runtime_name = f"{runtime_module}.{scale_name}"
    raw = torch.tensor([[120, 127], [128, 130]], dtype=torch.uint8)
    checkpoint_scale = raw.view(scale_dtype) if mxfp8 else raw.float()
    param = nn.Parameter(torch.empty_like(checkpoint_scale), requires_grad=False)
    shards = []

    def load_scale(param, weight, *args):
        shards.append(args)
        assert weight.dtype == checkpoint_scale.dtype
        param.copy_(weight)

    param.weight_loader = load_scale
    draft = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, n_routed_experts=1),
        quant_config=SimpleNamespace(
            weight_block_size=[32, 32] if mxfp8 else [128, 128]
        ),
        linear_scale_name=scale_name,
        pad_shared_expert=False,
        model=SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_native_mega_moe=False))],
            confidence_head=None,
        ),
        named_parameters=lambda: [(runtime_name, param)],
        process_weights_after_loading=lambda: None,
    )
    draft._remap_dspark_name = lambda name: (
        dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name(draft, name)
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )

    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(
        draft, [(checkpoint_name, checkpoint_scale)]
    )

    assert loaded == {runtime_name}
    assert shards == [() if shard_id is None else (shard_id,)]
    torch.testing.assert_close(param, checkpoint_scale)
