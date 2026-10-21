# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check draft dispatch, shared helpers, and head construction without GPUs.

Production modules run with dependency stubs. These tests do not execute GPU
operators or validate numerical model, attention, or sampling implementations.
"""

import importlib.util
import sys
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

DISPATCH_PATH = (
    Path(__file__).resolve().parents[2] / "vllm/v1/worker/gpu/spec_decode/__init__.py"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def stub_module(monkeypatch):
    def create(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return create


@pytest.mark.parametrize(
    "method,module_name,class_name",
    [
        ("dspark", "dspark.speculator", "DSparkSpeculator"),
        ("dflash", "dflash.speculator", "DFlashSpeculator"),
        ("mtp", "mtp.speculator", "MTPSpeculator"),
        ("eagle", "eagle.speculator", "EagleSpeculator"),
        ("draft_model", "standalone_ar.speculator", "StandaloneARSpeculator"),
        ("ngram", "ngram.speculator", "NgramGPUSpeculator"),
        (
            "extract_hidden_states",
            "extract_hidden_states",
            "ExtractHiddenStatesSpeculator",
        ),
    ],
)
def test_dispatch_needs_no_removed_model_helpers(
    stub_module, method, module_name, class_name
):
    stub_module("torch", device=object)
    stub_module("vllm", __path__=[])
    stub_module("vllm.config", VllmConfig=object)
    stub_module(
        f"vllm.v1.worker.gpu.spec_decode.{module_name}",
        **{class_name: lambda *args: args},
    )
    spec = importlib.util.spec_from_file_location("speculator_dispatch", DISPATCH_PATH)
    assert spec is not None and spec.loader is not None
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            use_multi_module_mtp=lambda: False,
            use_eagle=lambda: method == "eagle",
            uses_draft_model=lambda: method == "draft_model",
            use_ngram=lambda: method == "ngram",
        )
    )
    device, req_states = object(), object()
    result = dispatch.init_speculator(config, device, req_states)
    expected_args = (
        (config, device, req_states) if method == "ngram" else (config, device)
    )
    assert result == expected_args


class _Backend(Enum):
    SPARSE_MLA = 1
    FLASHINFER_SPARSE = 2


@pytest.fixture
def dspark_utils(stub_module):
    torch = stub_module("torch", __path__=[])
    torch.nn = stub_module("torch.nn", Module=object)
    stub_module("vllm", __path__=[])
    stub_module(
        "vllm.config", ParallelConfig=object, VllmConfig=object, replace=replace
    )
    logger = SimpleNamespace(
        info_once=lambda *args: None, warning_once=lambda *args: None
    )
    stub_module("vllm.logger", init_logger=lambda _: logger)
    stub_module("vllm.model_executor.model_loader.utils", get_draft_load_config=None)
    stub_module("vllm.v1.attention.backends.registry", AttentionBackendEnum=_Backend)
    stub_module(
        "vllm.v1.worker.gpu.spec_decode.utils", get_pp_safe_draft_load_config=None
    )
    spec = importlib.util.spec_from_file_location(
        "dspark_utils", DISPATCH_PATH.parent / "dspark/utils.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target_backend", [None, *_Backend])
@pytest.mark.parametrize("draft_backend", [None, *_Backend])
def test_dspark_retains_target_cache_backend_unless_overridden(
    dspark_utils, target_backend, draft_backend
):
    expected = draft_backend if draft_backend is not None else target_backend
    assert (
        dspark_utils._resolve_dspark_attention_backend(draft_backend, target_backend)
        is expected
    )


@pytest.fixture
def draft_attention():
    spec = importlib.util.spec_from_file_location(
        "draft_attention", REPO_ROOT / "vllm/v1/spec_decode/draft_attention.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"layer_types": []}, True),
        ({"layer_types": ["sliding_attention", "sliding_attention"]}, False),
        ({"layer_types": ["sliding_attention", "full_attention"]}, True),
        ({"dflash_config": {"causal": True}}, False),
        (
            {
                "layer_types": ["sliding_attention", "sliding_attention"],
                "dflash_config": {"causal": False},
            },
            True,
        ),
        ({"is_causal": False, "dflash_config": {"causal": True}}, True),
        ({"is_causal": True, "dflash_config": {"causal": False}}, False),
    ],
)
def test_shared_draft_attention_preserves_override_precedence(
    draft_attention, overrides, expected
):
    config = SimpleNamespace(num_hidden_layers=2, **overrides)
    assert draft_attention.dflash_has_any_non_causal(config) is expected


def test_shared_dspark_heads_keep_checkpoint_names_and_quantization(stub_module):
    """Import shared heads without Qwen models and retain their weight layout."""
    torch = stub_module("torch", __path__=[], Tensor=object, float32=object())

    def layer(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs)

    torch.nn = stub_module("torch.nn", Module=object, Embedding=layer)
    stub_module("vllm", __path__=[])
    stub_module("vllm.model_executor.layers.linear", ReplicatedLinear=layer)
    stub_module("vllm.model_executor.layers.logits_processor", LogitsProcessor=object)
    stub_module("vllm.model_executor.layers.quantization", QuantizationConfig=object)
    stub_module(
        "vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils",
        dequantize_to_dtype=None,
        nvfp4_gathered_bias=None,
    )
    stub_module(
        "vllm.model_executor.layers.vocab_parallel_embedding", ParallelLMHead=layer
    )
    stub_module(
        "vllm.model_executor.models.utils",
        maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name,
    )
    stub_module("vllm.platforms", current_platform=None)
    spec = importlib.util.spec_from_file_location(
        "dspark_heads", REPO_ROOT / "vllm/model_executor/layers/dspark.py"
    )
    assert spec is not None and spec.loader is not None
    heads = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(heads)

    quant_config = object()
    markov = heads.DSparkMarkovHead(
        128, 128, 16, "draft.markov_head", quant_config, retain_weight_for_gather=True
    )
    assert markov.markov_w1.args == (128, 16)
    assert markov.markov_w2.args == (128, 16)
    assert markov.markov_w2.kwargs == {
        "bias": False,
        "quant_config": quant_config,
        "prefix": "draft.markov_head.markov_w2",
        "disable_tp": True,
    }
    assert markov.markov_w2._retain_weight_for_gather
    assert not markov.markov_w2.is_w4a16_nvfp4

    confidence = heads.DSparkConfidenceHead(80, "draft.confidence_head")
    assert confidence.with_markov
    assert confidence.proj.args == (80, 1)
    assert confidence.proj.kwargs == {
        "bias": False,
        "return_bias": False,
        "params_dtype": torch.float32,
        "prefix": "draft.confidence_head.proj",
    }
