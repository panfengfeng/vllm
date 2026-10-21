# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise complete architecture conversion and V4.1 config modules on CPU.

Torch, Transformers and the architecture-result dataclass use dependency stubs.
These tests cover conversion control flow, not real HF validation or GPU/model
loading. Run with --confcutdir=tests/standalone_tests.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def converters(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    class PreTrainedConfig(SimpleNamespace):
        def get_text_config(self):
            return getattr(self, "text_config", self)

    def getattr_iter(obj, names, default=None, default_factory=None):
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return default_factory() if default_factory else default

    metadata: dict[str, dict[str, str]] = {}
    envs = SimpleNamespace(VLLM_MLA_DISABLE=False)
    stub("torch", dtype=str, float32="float32")
    stub("safetensors.torch", _TYPES={"BF16": "bfloat16", "F32": "float32"})
    stub("transformers", PreTrainedConfig=PreTrainedConfig)
    stub("vllm", __path__=[], envs=envs)
    stub("vllm.config.model_arch", ModelArchitectureConfig=SimpleNamespace)
    stub("vllm.config.utils", getattr_iter=getattr_iter)
    stub("vllm.logger", init_logger=logging.getLogger)
    stub(
        "vllm.transformers_utils.config",
        ConfigFormat=str,
        get_safetensors_params_metadata=lambda *a, **kw: metadata,
    )
    stub(
        "vllm.utils.torch_utils",
        common_broadcastable_dtype=lambda dtypes: (
            "float32" if "float32" in dtypes else "bfloat16"
        ),
    )
    module = load(
        "vllm.transformers_utils.model_arch_config_convertor",
        "vllm/transformers_utils/model_arch_config_convertor.py",
    )
    config_module = load(
        "vllm.transformers_utils.configs.deepseek_v41",
        "vllm/transformers_utils/configs/deepseek_v41.py",
    )
    return SimpleNamespace(
        module=module,
        config_class=config_module.DeepseekV41Config,
        envs=envs,
        metadata=metadata,
        hf_class=PreTrainedConfig,
    )


def make_hf(converters, vision=False):
    return converters.config_class(
        architectures=["DeepseekV41ForCausalLM"],
        text_config=dict(
            hidden_size=4096,
            num_hidden_layers=43,
            num_attention_heads=64,
            num_key_value_heads=4,
            head_dim=512,
            compress_ratios=[1, 4],
            vocab_size=129280,
            num_nextn_predict_layers=3,
            n_routed_experts=256,
            num_experts_per_tok=8,
            max_position_embeddings=1048576,
        ),
        vision_config={"num_hidden_layers": 24} if vision else None,
    )


def convert(converters, hf, **kwargs):
    cls = converters.module.MODEL_ARCH_CONFIG_CONVERTORS.get(
        hf.model_type, converters.module.ModelArchConfigConvertorBase
    )
    return cls(hf, hf).convert(**kwargs)


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("mla_disabled", [False, True])
def test_v41_main_and_bundled_draft_keep_architecture_dimensions(
    converters, vision, mla_disabled
):
    converters.envs.VLLM_MLA_DISABLE = mla_disabled
    hf = make_hf(converters, vision)
    main = convert(converters, hf)
    assert main.architectures == ["DeepseekV41ForCausalLM"]
    assert main.total_num_hidden_layers == 43
    assert (main.hidden_size, main.head_size, main.vocab_size) == (4096, 512, 129280)
    assert (main.total_num_attention_heads, main.total_num_kv_heads) == (64, 4)
    assert (main.num_experts, main.num_experts_per_token) == (256, 8)
    assert main.is_deepseek_mla
    assert not main.is_mm_prefix_lm
    assert main.derived_max_model_len_and_key == (1048576, "max_position_embeddings")

    hf.model_type = "deepseek_mtp"
    temporary = convert(converters, hf)
    assert temporary.total_num_hidden_layers == 3
    assert temporary.architectures == main.architectures
    assert temporary.head_size == main.head_size
    assert temporary.is_deepseek_mla

    hf.model_type = "deepseek_v41"
    hf.architectures = ["DSparkV41DraftModel"]
    draft = convert(converters, hf)
    assert draft.total_num_hidden_layers == main.total_num_hidden_layers
    assert draft.architectures == ["DSparkV41DraftModel"]
    assert draft.head_size == main.head_size


def test_explicit_multimodal_prefix_setting_obeys_serving_configuration(converters):
    hf = make_hf(converters, vision=True)
    hf.is_mm_prefix_lm = True
    assert convert(converters, hf).is_mm_prefix_lm
    assert not convert(converters, hf, supports_multimodal=False).is_mm_prefix_lm


def test_modelopt_nvfp4_metadata_is_preserved_and_normalized(converters):
    hf = make_hf(converters)
    hf.quantization_config = {
        "producer": {"name": "modelopt"},
        "quantization": {"quant_algo": "NVFP4"},
        "expert_dtype": "fp4",
    }
    quant = convert(converters, hf).quantization_config
    assert quant["quant_method"] == "modelopt_fp4"
    assert quant["quantization"] == {"quant_algo": "NVFP4"}
    assert quant["expert_dtype"] == "fp4"


@pytest.mark.parametrize("source", ["root", "text", "vision", "weights", "default"])
def test_dtype_resolution_retains_multimodal_and_weight_fallbacks(converters, source):
    hf = converters.hf_class(dtype=None)
    if source == "root":
        hf.dtype = "bfloat16"
    elif source in ("text", "vision"):
        setattr(hf, f"{source}_config", converters.hf_class(dtype="bfloat16"))
    elif source == "weights":
        converters.metadata["weight"] = {"dtype": "BF16"}
    actual = converters.module.ModelArchConfigConvertorBase.get_torch_dtype(
        hf, "unused-checkpoint", revision=None, config_format="hf"
    )
    assert actual == ("float32" if source == "default" else "bfloat16")
