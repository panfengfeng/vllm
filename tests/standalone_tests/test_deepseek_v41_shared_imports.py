# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared V4 package imports must not initialize an unsupported model."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
