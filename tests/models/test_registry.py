# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch.cuda

from vllm.model_executor.models import is_text_generation_model, supports_multimodal
from vllm.model_executor.models.registry import ModelRegistry, _LazyRegisteredModel
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_imports(model_arch):
    if not current_platform.is_cuda_alike():
        pytest.skip("DeepSeek V4.1 model imports require a CUDA-like platform")
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )
    model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
        model_arch, SimpleNamespace(model_impl="vllm")
    )
    assert resolved_arch == model_arch
    if model_arch == "DeepseekV41ForCausalLM":
        assert is_text_generation_model(model_cls)
        assert supports_multimodal(model_cls)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires a CUDA-like platform"
)
def test_registry_model_properties():
    model_info, arch = ModelRegistry.inspect_model_cls(
        "DeepseekV41ForCausalLM", SimpleNamespace(model_impl="vllm")
    )
    assert arch == "DeepseekV41ForCausalLM"
    assert model_info.is_text_generation_model
    assert model_info.supports_multimodal
    assert model_info.supports_pp
    assert model_info.score_type == "bi-encoder"


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_arch", ModelRegistry.get_supported_archs())
def test_registry_inspect_does_not_init_cuda(model_arch):
    """Model metadata inspection must leave CUDA initialization to the worker."""
    if not current_platform.is_cuda_alike():
        pytest.skip("Requires a CUDA-like platform")
    model_info, arch = ModelRegistry.inspect_model_cls(
        model_arch, SimpleNamespace(model_impl="vllm")
    )
    assert model_info is not None
    assert arch == model_arch
    assert not torch.cuda.is_initialized()


def test_lazy_modelinfo_package_hash_includes_submodules(tmp_path):
    package_dir = tmp_path / "model_package"
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    init_file.write_text("from .model import Model\n", encoding="utf-8")
    model_file = package_dir / "model.py"
    model_file.write_text("class Model: pass\n", encoding="utf-8")

    first_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    model_file.write_text("class Model:\n    supports_pp = True\n", encoding="utf-8")
    second_hash = _LazyRegisteredModel._get_modelinfo_module_hash(init_file)

    assert first_hash != second_hash


def test_lazy_modelinfo_package_attempts_cache_load(monkeypatch):
    cached_model_info = object()
    loaded_hashes = []

    def fake_load_cache(self, module_hash):
        loaded_hashes.append(module_hash)
        return cached_model_info

    monkeypatch.setattr(
        _LazyRegisteredModel,
        "_load_modelinfo_from_cache",
        fake_load_cache,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry._run_in_subprocess",
        lambda _: pytest.fail("Package-backed model should use the cache path"),
    )

    registered_model = _LazyRegisteredModel(
        module_name="vllm.models.deepseek_v41",
        class_name="DeepseekV41ForCausalLM",
    )

    result = registered_model.inspect_model_cls()

    assert result is cached_model_info
    assert len(loaded_hashes) == 1
    assert loaded_hashes[0]


def test_hf_registry_coverage():
    untested_archs = (
        ModelRegistry.get_supported_archs() - HF_EXAMPLE_MODELS.get_supported_archs()
    )

    assert not untested_archs, (
        "Please add the following architectures to "
        f"`tests/models/registry.py`: {untested_archs}"
    )
