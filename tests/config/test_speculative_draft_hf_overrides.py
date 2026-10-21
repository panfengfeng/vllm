# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for V4.1 bundled draft overrides and callable override composition."""

import functools
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


def _make_hf_config(**kwargs) -> PreTrainedConfig:
    defaults = dict(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        num_hidden_layers=64,
    )
    defaults.update(kwargs)
    return PreTrainedConfig(**defaults)


@pytest.mark.cpu_test
def test_dict_overrides_are_not_forwarded_to_draft():
    """Dict overrides are target-specific key patches; the draft must get
    only the architecture-mapping override."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"max_position_embeddings": 1234}
    )
    assert composed is SpeculativeConfig.hf_config_override


@pytest.mark.cpu_test
def test_none_overrides_fall_back_to_arch_mapping():
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)
    assert composed is SpeculativeConfig.hf_config_override


def _make_speculative_config(
    hf_kwargs: dict, architecture: str, **speculative_kwargs
) -> SpeculativeConfig:
    hf_config = SpeculativeConfig.hf_config_override(_make_hf_config(**hf_kwargs))
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, architecture)
    target = MagicMock(max_model_len=128, quantization=None, hf_overrides={})
    target.hf_config.model_type = hf_kwargs["model_type"]

    with patch("vllm.config.speculative.ModelConfig", return_value=draft):
        return SpeculativeConfig(
            model="draft",
            **speculative_kwargs,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("hf_kwargs", "n_predict", "architecture"),
    [
        (
            dict(
                model_type="deepseek_v41",
                num_nextn_predict_layers=3,
                dspark_block_size=5,
            ),
            3,
            "DSparkV41DraftModel",
        ),
    ],
)
@pytest.mark.parametrize("speculative_kwargs", [{"num_speculative_tokens": 5}, {}])
def test_dspark_width_is_independent_of_mtp_stages(
    hf_kwargs, n_predict, architecture, speculative_kwargs
):
    config = _make_speculative_config(
        hf_kwargs, architecture, method="dspark", **speculative_kwargs
    )

    assert config.num_speculative_tokens == 5
    assert getattr(config.draft_model_config.hf_config, "n_predict", None) == n_predict
    assert config.draft_model_config.hf_config.architectures == [architecture]


@pytest.mark.cpu_test
def test_classic_mtp_is_rejected_for_v41():
    with pytest.raises(ValueError, match="has no classic-MTP draft"):
        _make_speculative_config(
            dict(model_type="deepseek_v41", num_nextn_predict_layers=3),
            "DeepseekV41ForCausalLM",
            method="mtp",
            num_speculative_tokens=3,
        )


@pytest.mark.cpu_test
def test_callable_overrides_reach_the_draft_config():
    """A callable override (config-to-config transform) composes with the
    architecture-mapping override and is applied to the draft config."""

    def shrink(hf_config: PreTrainedConfig) -> PreTrainedConfig:
        hf_config.num_hidden_layers = 1
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(shrink)
    assert composed is not SpeculativeConfig.hf_config_override

    out = composed(_make_hf_config())
    # The shrink transform must have been applied to the draft config.
    assert out.num_hidden_layers == 1


@pytest.mark.cpu_test
def test_v41_stage_metadata_applies_before_callable_override():
    seen = []

    def record(hf_config: PreTrainedConfig) -> PreTrainedConfig:
        seen.append(
            (hf_config.model_type, hf_config.n_predict, hf_config.architectures)
        )
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(record)
    composed(_make_hf_config(num_nextn_predict_layers=3))
    assert seen == [("deepseek_mtp", 3, ["DeepseekV41ForCausalLM"])]


def _module_level_shrink(hf_config: PreTrainedConfig) -> PreTrainedConfig:
    hf_config.num_hidden_layers = 1
    return hf_config


@pytest.mark.cpu_test
def test_composed_override_is_picklable():
    """The draft ``ModelConfig`` is sent to spawned engine-core processes, so
    the composed override must be picklable. A nested local closure is not
    (it raised ``Can't get local object`` on DFlashDraftModel); a
    ``functools.partial`` over a module-referenceable static method is.
    Guard against regressing to a closure."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(_module_level_shrink)

    assert isinstance(composed, functools.partial)
    assert composed.func is SpeculativeConfig._apply_composed_hf_override

    out = composed(_make_hf_config())
    assert out.num_hidden_layers == 1
