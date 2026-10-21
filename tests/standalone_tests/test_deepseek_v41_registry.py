# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise registry policy without installing Torch or the GPU runtime.

The complete production registry module is loaded with dependency stubs. These
tests cover architecture selection and registration, not real model imports,
model metadata, subprocess isolation, CUDA execution or numerical correctness.

Run with --confcutdir=tests/standalone_tests to exclude the GPU test fixtures.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SUPPORTED_ARCHITECTURES = ("DeepseekV41ForCausalLM", "DSparkV41DraftModel")
REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "vllm/model_executor/models/registry.py"
)


@pytest.fixture
def registry(monkeypatch):
    def stub_module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class Module:
        pass

    stub_module("torch", __path__=[])
    stub_module("torch.nn", Module=Module)
    stub_module("vllm", __path__=[])
    stub_module("vllm.envs", VLLM_CACHE_ROOT="unused-model-cache")
    stub_module("vllm.config", ModelConfig=SimpleNamespace)
    stub_module("vllm.logger", init_logger=logging.getLogger)
    stub_module("vllm.logging_utils", logtime=lambda **kwargs: lambda fn: fn)
    stub_module("vllm.tasks", ScoreType=str)
    stub_module("vllm.utils", __path__=[])
    stub_module("vllm.utils.hashing", safe_hash=None)
    stub_module("vllm.model_executor", __path__=[])
    stub_module("vllm.model_executor.models", __path__=[])
    stub_module(
        "vllm.platforms",
        current_platform=SimpleNamespace(verify_model_arch=lambda arch: None),
    )
    stub_module(
        "vllm.model_executor.models.interfaces",
        **dict.fromkeys(
            (
                "has_inner_state",
                "has_noops",
                "is_attention_free",
                "is_hybrid",
                "requires_raw_input_tokens",
                "supports_mamba_prefix_caching",
                "supports_multimodal",
                "supports_multimodal_encoder_tp_data",
                "supports_multimodal_raw_input_only",
                "supports_pp",
                "supports_replayssm",
                "supports_transcription",
            ),
            lambda cls: False,
        ),
    )
    stub_module(
        "vllm.model_executor.models.interfaces_base",
        **dict.fromkeys(
            (
                "get_attn_type",
                "get_default_seq_pooling_type",
                "get_default_tok_pooling_type",
                "get_score_type",
                "is_pooling_model",
                "is_text_generation_model",
            ),
            lambda cls: False,
        ),
    )
    stub_module("vllm.models", __path__=[])
    model_classes = {
        "DeepseekV41ForCausalLM": type("DeepseekV41ForCausalLM", (Module,), {}),
        "DSparkV41DraftModel": type("DSparkDeepseekV4ForCausalLM", (Module,), {}),
    }
    stub_module(
        "vllm.models.deepseek_v41",
        **{cls.__name__: cls for cls in model_classes.values()},
    )

    name = "vllm.model_executor.models.registry"
    spec = importlib.util.spec_from_file_location(name, REGISTRY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_run_in_subprocess", lambda fn: fn())
    return SimpleNamespace(module=module, models=model_classes)


def test_only_target_and_bundled_draft_are_advertised(registry):
    assert set(registry.module.ModelRegistry.get_supported_archs()) == set(
        SUPPORTED_ARCHITECTURES
    )


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
@pytest.mark.parametrize("model_impl", ["auto", "vllm"])
def test_target_and_bundled_draft_resolve_without_fallback(
    registry, architecture, model_impl
):
    config = SimpleNamespace(model_impl=model_impl)
    model_class, resolved = registry.module.ModelRegistry.resolve_model_cls(
        architecture, config
    )
    assert model_class is registry.models[architecture]
    assert resolved == architecture
    info, inspected = registry.module.ModelRegistry.inspect_model_cls(
        [architecture], config
    )
    assert info.architecture == model_class.__name__
    assert inspected == architecture


@pytest.mark.parametrize("operation", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize(
    "architectures",
    [
        ["LlamaForCausalLM"],
        ["DeepseekV4ForCausalLM"],
        ["DSparkDraftModel"],
        ["Qwen3DSparkModel"],
        ["TransformersForCausalLM"],
        ["Terratorch"],
        ["DeepseekV41Model"],
        ["LlamaForCausalLM", "DeepseekV41ForCausalLM"],
    ],
)
def test_other_architectures_are_rejected_before_loading(
    registry, monkeypatch, operation, architectures
):
    def unexpected_load(*args):
        pytest.fail("An unsupported request must not import or inspect a model")

    monkeypatch.setattr(
        registry.module._LazyRegisteredModel, "load_model_cls", unexpected_load
    )
    monkeypatch.setattr(
        registry.module._LazyRegisteredModel, "inspect_model_cls", unexpected_load
    )
    with pytest.raises(ValueError, match="DeepSeek V4.1-only"):
        getattr(registry.module.ModelRegistry, operation)(
            architectures, SimpleNamespace(model_impl="auto")
        )


@pytest.mark.parametrize("operation", ["inspect_model_cls", "resolve_model_cls"])
@pytest.mark.parametrize("model_impl", ["transformers", "terratorch"])
def test_alternative_implementations_cannot_replace_target(
    registry, operation, model_impl
):
    with pytest.raises(ValueError, match="alternative model implementations"):
        getattr(registry.module.ModelRegistry, operation)(
            "DeepseekV41ForCausalLM", SimpleNamespace(model_impl=model_impl)
        )


@pytest.mark.parametrize("operation", ["inspect_model_cls", "resolve_model_cls"])
def test_missing_architectures_are_rejected(registry, operation):
    with pytest.raises(ValueError, match="No model architectures"):
        getattr(registry.module.ModelRegistry, operation)(
            [], SimpleNamespace(model_impl="auto")
        )


@pytest.mark.parametrize("architecture", ["LlamaForCausalLM", *SUPPORTED_ARCHITECTURES])
@pytest.mark.parametrize("lazy", [False, True])
def test_external_registration_cannot_extend_or_override_builtins(
    registry, architecture, lazy
):
    model = (
        "external_plugin:Model" if lazy else registry.models["DeepseekV41ForCausalLM"]
    )
    with pytest.raises(ValueError, match="fixed model registry"):
        registry.module.ModelRegistry.register_model(architecture, model)


def test_registry_mapping_cannot_be_extended_directly(registry):
    with pytest.raises(TypeError):
        registry.module.ModelRegistry.models["LlamaForCausalLM"] = (
            registry.module.ModelRegistry.models["DeepseekV41ForCausalLM"]
        )


@pytest.mark.parametrize(
    "operation,loader,error",
    [
        ("inspect_model_cls", "inspect_model_cls", "failed to be inspected"),
        ("resolve_model_cls", "load_model_cls", "failed to be loaded"),
    ],
)
def test_broken_builtin_reports_failure_without_an_alternative(
    registry, monkeypatch, operation, loader, error
):
    def broken_load(*args):
        raise ImportError("Built-in model dependency is missing")

    monkeypatch.setattr(registry.module._LazyRegisteredModel, loader, broken_load)
    with pytest.raises(ValueError, match=error):
        getattr(registry.module.ModelRegistry, operation)(
            "DeepseekV41ForCausalLM", SimpleNamespace(model_impl="auto")
        )
