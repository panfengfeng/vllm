# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the retained startup warmup pipeline without Torch or a GPU.

Load the complete production module with hardware operations stubbed. This
checks the execution order and process-local boundary, not kernel compilation
or numerical correctness. Run with --confcutdir=tests/standalone_tests.
"""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

WARMUP_PATH = (
    Path(__file__).resolve().parents[2] / "vllm/model_executor/warmup/kernel_warmup.py"
)


@pytest.fixture
def warmup(monkeypatch):
    calls = []

    def record(name):
        def invoke(*args, **kwargs):
            calls.append(name)

        return invoke

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for package in (
        "vllm",
        "vllm.utils",
        "vllm.distributed",
        "vllm.distributed.device_communicators",
        "vllm.model_executor",
        "vllm.model_executor.warmup",
        "flashinfer",
    ):
        stub(package, __path__=[])
    stub("torch", nn=SimpleNamespace(Module=object), inference_mode=nullcontext)
    stub("vllm.logger", init_logger=lambda name: Mock())
    stub(
        "vllm.envs",
        VLLM_DEEP_GEMM_WARMUP="full",
        VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=True,
    )
    stub(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_device_capability_family=lambda family: True,
            has_device_capability=lambda minimum: True,
            is_rocm=lambda: False,
        ),
    )
    stub("vllm.utils.deep_gemm", is_deep_gemm_supported=lambda: True)
    stub("vllm.utils.flashinfer", has_flashinfer=lambda: True)
    prefix = "vllm.model_executor.warmup."
    for name in (
        "b12x_warmup",
        "cutedsl_warmup",
        "deep_gemm_warmup",
        "watermark_sample_warmup",
    ):
        stub(prefix + name, **{name: record(name)})
    stub(
        prefix + "flashinfer_autotune_cache",
        resolve_flashinfer_autotune_file=lambda runner: Path("unused.json"),
    )
    stub(
        prefix + "flashinfer_sparse_mla_warmup",
        **{
            name: record(name)
            for name in (
                "autotune_hisparse_flashinfer_attention",
                "deepseek_v4_sparse_mla_attention_warmup",
                "flashinfer_sparse_mla_decode_autotune_warmup",
            )
        },
    )
    stub(
        "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar",
        warmup_gemm_rs_ar=record("gemm_rs_ar"),
    )
    stub(
        "vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce",
        warmup_flashinfer_pcie_ipc_allreduce=record("pcie_ipc"),
    )
    name = prefix + "kernel_warmup"
    spec = importlib.util.spec_from_file_location(name, WARMUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_warmup_ll_bf16_router_gemm", record("ll_bf16"))
    monkeypatch.setattr(module, "_warmup_bf16x3_router_gemm", record("bf16x3"))
    return SimpleNamespace(module=module, calls=calls, record=record, stub=stub)


@pytest.mark.parametrize("process_local_only", [False, True])
@pytest.mark.parametrize("use_v2_model_runner", [False, True])
@pytest.mark.parametrize("enable_jit_warmup", [False, True])
def test_target_warmups_preserve_order_and_collective_boundary(
    warmup, monkeypatch, process_local_only, use_v2_model_runner, enable_jit_warmup
):
    monkeypatch.setattr(
        warmup.module, "flashinfer_autotune", warmup.record("flashinfer_autotune")
    )
    runner = SimpleNamespace(
        _kv_block_zeroer=SimpleNamespace(warmup=warmup.record("zero_blocks")),
        kv_cache_config=SimpleNamespace(num_blocks=10),
        jit_warmup_registry=SimpleNamespace(warmup=warmup.record("jit_registry")),
        is_pooling_model=False,
        attn_groups=[
            [SimpleNamespace(backend=SimpleNamespace(get_name=lambda: "FLASHINFER"))]
        ],
        _dummy_run=warmup.record("attention_dummy_run"),
    )
    worker = SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        model_runner=runner,
        get_model=lambda: object(),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(
                enable_jit_warmup=enable_jit_warmup,
                enable_cutedsl_warmup=True,
                enable_flashinfer_autotune=True,
            ),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2]),
        ),
    )
    warmup.module.kernel_warmup(worker, process_local_only=process_local_only)
    expected = [] if use_v2_model_runner else ["zero_blocks"]
    if enable_jit_warmup:
        expected.extend(["jit_registry", "watermark_sample_warmup", "bf16x3"])
    expected.extend(["ll_bf16", "gemm_rs_ar", "cutedsl_warmup"])
    if not process_local_only:
        expected.extend(
            [
                "flashinfer_sparse_mla_decode_autotune_warmup",
                "deepseek_v4_sparse_mla_attention_warmup",
                "deep_gemm_warmup",
                "b12x_warmup",
                "pcie_ipc",
                "flashinfer_autotune",
                "attention_dummy_run",
            ]
        )
    assert warmup.calls == expected


def test_flashinfer_autotune_keeps_model_runs_without_foreign_model_hooks(
    warmup, monkeypatch, tmp_path
):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    warmup.stub(
        "vllm.distributed.parallel_state",
        **dict.fromkeys(
            ("get_world_group", "get_tp_group", "get_pp_group"), lambda: group
        ),
    )
    warmup.stub(
        "flashinfer.autotuner",
        AutoTuner=SimpleNamespace(get=lambda: SimpleNamespace(_dirty=False)),
        set_autotune_process_group=warmup.record("set_tuning_group"),
    )
    warmup.stub("vllm.utils.flashinfer", autotune=lambda **kwargs: nullcontext())
    monkeypatch.setattr(warmup.module, "_flashinfer_autotune_skip_ops", lambda r: None)
    monkeypatch.setattr(
        warmup.module,
        "resolve_flashinfer_autotune_file",
        lambda r: tmp_path / "cache.json",
    )
    monkeypatch.setattr(
        warmup.module, "_run_flashinfer_autotune_dummy_runs", warmup.record("model_run")
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=0),
            attention_config=SimpleNamespace(hisparse_config=None),
        ),
    )
    warmup.module.flashinfer_autotune(runner)
    assert warmup.calls == ["set_tuning_group", "model_run", "set_tuning_group"]
