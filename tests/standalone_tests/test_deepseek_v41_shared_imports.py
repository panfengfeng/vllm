# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check shared imports and V4.1 execution dispatch without GPU dependencies."""

import __future__

import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_shared_v4_package_only_initializes_quantization(monkeypatch):
    package = "vllm.models.deepseek_v4"
    for name in ("vllm", "vllm.models"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    quantization = ModuleType(f"{package}.quant_config")
    config_class = type("DeepseekV4FP8Config", (), {})
    quantization.__dict__["DeepseekV4FP8Config"] = config_class
    monkeypatch.setitem(sys.modules, quantization.__name__, quantization)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.platforms" or (
            level and name.split(".")[0] in {"amd", "cpu", "nvidia", "xpu", "common"}
        ):
            raise AssertionError("Importing shared helpers must not load a V4 model")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    source = Path(__file__).resolve().parents[2] / "vllm/models/deepseek_v4/__init__.py"
    spec = importlib.util.spec_from_file_location(package, source)
    assert spec is not None and spec.loader is not None
    shared = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, shared)
    spec.loader.exec_module(shared)

    assert shared.DeepseekV4FP8Config is config_class
    assert not hasattr(shared, "DeepseekV4ForCausalLM")
    assert not hasattr(shared, "DSparkDeepseekV4ForCausalLM")


def _load_function(path, name, namespace):
    tree = ast.parse((REPO_ROOT / path).read_text())
    function = next(n for n in tree.body if getattr(n, "name", None) == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(
        compile(module, path, "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "missing_hook",
        "other_state",
        "is_encoder_decoder",
        "is_hybrid",
        "is_attention_free",
    ],
)
def test_model_state_keeps_v41_wrappers_and_rejects_unsupported_states(
    monkeypatch, unsupported
):
    state = type("DeepseekV41ModelState", (), {})
    module_name = "vllm.models.deepseek_v41.nvidia.model_state"
    module = ModuleType(module_name)
    module.__dict__["DeepseekV41ModelState"] = state
    monkeypatch.setitem(sys.modules, module_name, module)
    resolve = _load_function(
        "vllm/v1/worker/gpu/model_states/__init__.py", "resolve_model_state_cls", {}
    )

    class Model:
        @staticmethod
        def get_model_state_cls():
            return object if unsupported == "other_state" else state

    class ConvertedModel(Model):
        pass

    model = object() if unsupported == "missing_hook" else ConvertedModel()
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_encoder_decoder=unsupported == "is_encoder_decoder",
            is_hybrid=unsupported == "is_hybrid",
            is_attention_free=unsupported == "is_attention_free",
        )
    )
    if unsupported:
        with pytest.raises(ValueError, match="This build"):
            resolve(config, model)
    else:
        assert resolve(config, model) is state


@pytest.mark.parametrize("capture", [False, True])
def test_attention_metadata_preserves_per_group_buffers_and_replay_kwargs(capture):
    build_metadata = _load_function(
        "vllm/v1/worker/gpu/attn_utils.py",
        "build_attn_metadata",
        {
            "torch": SimpleNamespace(Tensor=tuple),
            "CommonAttentionMetadata": SimpleNamespace,
        },
    )
    builder = SimpleNamespace(
        build=lambda **kwargs: kwargs,
        build_for_cudagraph_capture=lambda common: {"common_attn_metadata": common},
    )
    groups = [
        [SimpleNamespace(layer_names=[name], get_metadata_builder=lambda _: builder)]
        for name in ("swa", "compressed")
    ]
    replay_start, is_prefilling = object(), object()
    replay = SimpleNamespace(
        get_extra_common_attn_kwargs=lambda *_: {"is_prefilling": is_prefilling},
        get_extra_attn_kwargs=lambda *_: {"replay_start": replay_start},
    )
    tables, slots = [object(), object()], [object(), object()]
    result = build_metadata(
        attn_groups=groups,
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=[0, 1],
        query_start_loc_cpu=[0, 1],
        max_query_len=1,
        seq_lens=[3],
        max_seq_len=3,
        block_tables=tables,
        slot_mappings=slots,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object(), object()]),
        model_specific_attn_metadata=replay,
        for_cudagraph_capture=capture,
    )
    for i, name in enumerate(("swa", "compressed")):
        metadata = result[name]
        common = metadata["common_attn_metadata"]
        assert common.block_table_tensor is tables[i]
        assert common.slot_mapping is slots[i]
        assert common.is_prefilling is is_prefilling
        if not capture:
            assert metadata["replay_start"] is replay_start
