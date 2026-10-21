# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test real speculative config control flow with CPU dependency stubs.

The full production module is loaded. HF/model loading and Pydantic decorators
are stubbed, so these tests do not validate field schemas, real model metadata,
checkpoint loading, GPU execution or numerical correctness. Run with
--confcutdir=tests/standalone_tests to exclude the GPU test fixtures.
"""

import copy
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SPECULATIVE_PATH = Path(__file__).resolve().parents[2] / "vllm/config/speculative.py"


class HfConfig(SimpleNamespace):
    def update(self, values):
        self.__dict__.update(values)


@pytest.fixture
def speculative(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class SkipValidation:
        def __class_getitem__(cls, item):
            return item

    def config(cls):
        return dataclasses.dataclass(cls)

    target_hf = HfConfig(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        num_nextn_predict_layers=3,
        dspark_block_size=5,
        vocab_size=128,
    )
    loaded = []

    def inspect(architectures, model):
        assert architectures == ["DSparkV41DraftModel"]
        return None, architectures[0]

    class ModelConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hf_config = kwargs["hf_overrides"](copy.deepcopy(target_hf))
            loaded.append(copy.deepcopy(self.hf_config))
            self.max_model_len = kwargs.get("max_model_len") or 256
            self.registry = SimpleNamespace(inspect_model_cls=inspect)
            self.is_moe = False

        @property
        def architectures(self):
            return self.hf_config.architectures

        def get_model_arch_config(self):
            return self.hf_config

    parallel = SimpleNamespace(
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        distributed_executor_backend=None,
        max_parallel_loading_workers=None,
        disable_custom_all_reduce=False,
        ray_workers_use_nsight=False,
        placement_group=None,
    )
    target = SimpleNamespace(
        model="target-checkpoint",
        model_weights=None,
        hf_config=target_hf,
        hf_overrides={},
        quantization="fp8",
        tokenizer="target-tokenizer",
        tokenizer_mode="deepseek_v41",
        trust_remote_code=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="bfloat16",
        seed=0,
        tokenizer_revision=None,
        max_model_len=128,
        enforce_eager=False,
        max_logprobs=20,
        config_format="auto",
    )
    stub(
        "pydantic",
        Field=lambda *, default, **kwargs: default,
        SkipValidation=SkipValidation,
        field_validator=lambda *args, **kwargs: lambda fn: fn,
        model_validator=lambda *args, **kwargs: lambda fn: fn,
    )
    stub("vllm", __path__=[])
    stub("vllm.config", __path__=[], LoadConfig=SimpleNamespace)
    stub("vllm.config.cache", CacheDType=str)
    stub("vllm.config.kernel", MoEBackend=str)
    stub("vllm.config.model", HfOverrides=object, ModelConfig=ModelConfig)
    stub("vllm.config.parallel", ParallelConfig=SimpleNamespace)
    stub("vllm.config.utils", config=config, replace=dataclasses.replace)
    logger = SimpleNamespace(
        info=lambda *a: None, warning=lambda *a: None, warning_once=lambda *a: None
    )
    stub("vllm.logger", init_logger=lambda name: logger)
    stub("vllm.transformers_utils.config", get_hf_text_config=lambda hf: hf)
    stub("vllm.utils.hashing", safe_hash=None)
    stub(
        "vllm.utils.import_utils",
        LazyLoader=lambda *args: SimpleNamespace(QuantizationMethods=str),
        has_arctic_inference=lambda: True,
    )
    stub("vllm.v1.attention.backends.registry", AttentionBackendEnum=str)
    name = "vllm.config.speculative"
    spec = importlib.util.spec_from_file_location(name, SPECULATIVE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    def make(**kwargs):
        return module.SpeculativeConfig(
            target_model_config=target, target_parallel_config=parallel, **kwargs
        )

    return SimpleNamespace(
        module=module, make=make, target=target, parallel=parallel, loaded=loaded
    )


@pytest.mark.parametrize("width", [None, 5, 7])
def test_bundled_dspark_preserves_two_stage_architecture_transition(speculative, width):
    config = speculative.make(
        method="dspark", model="target-checkpoint", num_speculative_tokens=width
    )
    initial = speculative.loaded[0]
    assert initial.model_type == "deepseek_mtp"
    assert initial.architectures == ["DeepseekV41ForCausalLM"]
    assert initial.n_predict == 3
    draft = config.draft_model_config
    assert draft.hf_config.model_type == "deepseek_v41"
    assert draft.architectures == ["DSparkV41DraftModel"]
    assert draft._architecture == "DSparkV41DraftModel"
    assert draft.quantization == "fp8"
    assert draft.max_model_len == 128
    assert config.num_speculative_tokens == (width or 5)
    assert config.parallel_drafting
    assert config.max_num_new_slots_for_drafting == (width or 5) - 1
    assert config.draft_tensor_parallel_size == 4
    assert not config.draft_parallel_config.enable_expert_parallel


def test_dspark_uses_target_checkpoint_and_quantization_when_model_is_omitted(
    speculative,
):
    config = speculative.make(method="dspark", num_speculative_tokens=5)
    assert config.model == speculative.target.model
    assert config.quantization == speculative.target.quantization


def test_dspark_rejects_an_unsupported_target_before_architecture_rewrite(speculative):
    speculative.target.hf_config.model_type = "deepseek_v4"
    with pytest.raises(ValueError, match="requires a DeepSeek V4.1 target"):
        speculative.make(method="dspark", num_speculative_tokens=5)


def test_callable_override_sees_v41_stage_metadata_before_dspark_selection(speculative):
    seen = []

    def shrink(hf):
        seen.append((hf.model_type, hf.n_predict, list(hf.architectures)))
        hf.num_hidden_layers = 1
        return hf

    speculative.target.hf_overrides = shrink
    config = speculative.make(method="dspark", num_speculative_tokens=5)
    assert seen == [("deepseek_mtp", 3, ["DeepseekV41ForCausalLM"])]
    assert config.draft_model_config.hf_config.num_hidden_layers == 1


@pytest.mark.parametrize("method", ["mtp", "deepseek_mtp"])
def test_classic_mtp_rejects_v41_checkpoint_with_dspark_guidance(speculative, method):
    with pytest.raises(
        ValueError, match="speculative method 'dspark' instead of 'mtp'"
    ):
        speculative.make(method=method, num_speculative_tokens=3)


@pytest.mark.parametrize("source", ["argument", "checkpoint"])
def test_qwen_only_topk_cannot_be_silently_enabled_for_v41(speculative, source):
    kwargs = {"dspark_draft_topk": 16} if source == "argument" else {}
    if source == "checkpoint":
        speculative.target.hf_config.dspark_draft_topk = 16
    with pytest.raises(ValueError, match="not supported by DeepSeek V4.1 DSpark"):
        speculative.make(method="dspark", num_speculative_tokens=5, **kwargs)


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu", "suffix", "custom_class"])
def test_model_free_methods_do_not_load_a_draft_checkpoint(speculative, method):
    config = speculative.make(
        method=method,
        num_speculative_tokens=5,
        **({"model": "test.CustomProposer"} if method == "custom_class" else {}),
    )
    assert config.method == method
    if method == "suffix":
        assert config.draft_model_config is None
        assert config.draft_parallel_config is None
    else:
        assert config.draft_model_config is speculative.target
        assert config.draft_parallel_config is speculative.parallel
    assert not speculative.loaded


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
def test_ngram_lookup_window_validation_is_preserved(speculative, method):
    with pytest.raises(ValueError, match="must be <= prompt_lookup_max"):
        speculative.make(
            method=method,
            num_speculative_tokens=5,
            prompt_lookup_min=6,
            prompt_lookup_max=5,
        )


def test_v41_draft_tensor_parallel_override_is_validated(speculative):
    config = speculative.make(
        method="dspark", num_speculative_tokens=5, draft_tensor_parallel_size=1
    )
    assert config.draft_parallel_config.tensor_parallel_size == 1
    with pytest.raises(ValueError, match="target model tensor_parallel_size"):
        speculative.make(
            method="dspark", num_speculative_tokens=5, draft_tensor_parallel_size=2
        )
