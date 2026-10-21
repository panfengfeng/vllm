# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BF16 skinny GEMMs and model-specific selectors."""

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.models.deepseek_v32.nvidia import glm52_low_latency_gemm as glm52_gemm
from vllm.models.qwen4_exp.nvidia import low_latency_gemm as qwen4_exp_gemm

GLM_CUTE_CASES = [
    (spec, config)
    for spec in glm52_gemm.GLM52_PROJECTIONS.values()
    for _, config in spec.cute_configs
]

QWEN4_EXP_SM90_CASES = [
    (n, k, num_tokens, config)
    for (n, k), plans in qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS.items()
    for num_tokens, config in plans.items()
]

QWEN4_EXP_SM100_CASES = [
    (n, k, num_tokens, config)
    for (n, k), plans in qwen4_exp_gemm.QWEN4_EXP_SM100_GEMM_PLANS.items()
    for num_tokens, config in plans.items()
]

QWEN4_EXP_SM121_CASES = [
    (n, k, num_tokens, config)
    for (n, k), plans in qwen4_exp_gemm.QWEN4_EXP_SM121_GEMM_PLANS.items()
    for num_tokens, config in plans.items()
]


def test_glm52_projection_plans_are_separate() -> None:
    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION

    qkv_a_plan = qkv_a.build_plan()
    q_b_plan = q_b.build_plan()

    assert (qkv_a.n, qkv_a.k, set(qkv_a_plan)) == (
        2624,
        6144,
        set(range(1, 17)),
    )
    assert (q_b.n, q_b.k, set(q_b_plan)) == (
        2048,
        2048,
        set(range(1, 17)),
    )
    assert {
        num_tokens
        for num_tokens, (backend, _) in qkv_a_plan.items()
        if backend == "cute"
    } == {1, 2}
    assert {
        num_tokens
        for num_tokens, (backend, _) in qkv_a_plan.items()
        if backend == "dsv3_fused_a"
    } == set(range(3, 17))
    assert {
        num_tokens for num_tokens, (backend, _) in q_b_plan.items() if backend == "cute"
    } == {1, 2}
    assert {
        num_tokens
        for num_tokens, (backend, _) in q_b_plan.items()
        if backend == "dsv3_fused_a"
    } == set(range(3, 17))

    eh = glm52_gemm.GLM52_EH_PROJECTION
    eh_plan = eh.build_plan()
    assert (eh.n, eh.k) == (6144, 12288)
    # The MTP eh_proj has no dsv3 winners; M >= 4 falls back to cuBLAS.
    assert set(eh_plan) == {1, 2, 3}
    assert all(backend == "cute" for backend, _ in eh_plan.values())


def test_glm52_layout_rejects_nonpacked_single_row_view() -> None:
    single_row = torch.empty(1, 144)[:, :128]
    multiple_rows = torch.empty(2, 144)[:, :128]

    assert single_row.stride() == multiple_rows.stride() == (144, 1)
    assert not glm52_gemm._is_supported_row_major(single_row)
    assert not glm52_gemm._is_supported_row_major(multiple_rows)


def test_glm52_installer_maps_only_selected_unquantized_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearBase(nn.Module):
        def __init__(
            self,
            n: int,
            k: int,
            quant_method: object,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(
                    n,
                    k,
                    dtype=torch.bfloat16,
                    device="meta",
                )
            )
            self.quant_method = quant_method

    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION
    root = nn.Module()
    root.attn = nn.Module()
    root.attn.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.attn.q_b_proj = FakeLinearBase(
        q_b.n, q_b.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.same_shape_other_name = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.quantized = nn.Module()
    quantized_method = object()
    root.quantized.q_b_proj = FakeLinearBase(q_b.n, q_b.k, quantized_method)
    root.wrong_shape = nn.Module()
    root.wrong_shape.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n + 1, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    monkeypatch.setattr(glm52_gemm, "LinearBase", FakeLinearBase)
    monkeypatch.setattr(glm52_gemm, "_is_sm103", lambda: True)
    monkeypatch.setattr(
        glm52_gemm.shape_dynamic_skinny_gemm,
        "is_available",
        lambda: False,
    )

    glm52_gemm.enable_glm52_low_latency_gemm(
        root,
        torch.bfloat16,
    )

    assert isinstance(
        root.attn.fused_qkv_a_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert isinstance(
        root.attn.q_b_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.attn.fused_qkv_a_proj.quant_method._plan == qkv_a.build_plan()
    assert root.attn.q_b_proj.quant_method._plan == q_b.build_plan()
    assert isinstance(
        root.same_shape_other_name.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.quantized.q_b_proj.quant_method is quantized_method
    assert (
        type(root.wrong_shape.fused_qkv_a_proj.quant_method)
        is glm52_gemm.UnquantizedLinearMethod
    )


def test_qwen4_exp_hopper_plans_are_valid() -> None:
    plans = qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS

    assert len(plans) == 7
    assert sum(map(len, plans.values())) == 24
    assert (320, 10240) in plans
    assert (10240, 320) not in plans
    for (n, k), shape_plans in plans.items():
        for num_tokens, config in shape_plans.items():
            assert config.num_rows == num_tokens
            assert n % config.outputs_per_block == 0
            assert k % (config.block_size * config.vector_width) == 0
            assert config.static_k in (None, k)


@pytest.mark.parametrize(
    "capability,expected_plans",
    [
        ((10, 3), qwen4_exp_gemm.QWEN4_EXP_SM103_GEMM_PLANS),
        ((10, 0), qwen4_exp_gemm.QWEN4_EXP_SM100_GEMM_PLANS),
        ((9, 0), qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS),
        ((12, 1), qwen4_exp_gemm.QWEN4_EXP_SM121_GEMM_PLANS),
        ((12, 0), {}),
        ((8, 0), {}),
    ],
)
def test_qwen4_exp_gemm_capability_routing(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected_plans: dict[tuple[int, int], dict[int, SkinnyGemmConfig]],
) -> None:
    monkeypatch.setattr(
        qwen4_exp_gemm.current_platform,
        "get_device_capability",
        lambda: capability,
    )

    assert (
        qwen4_exp_gemm.QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY.get(
            qwen4_exp_gemm.current_platform.get_device_capability(), {}
        )
        == expected_plans
    )


def _require_capability_and_dsv3(capability: tuple[int, int]) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != capability
    ):
        pytest.skip(f"DSV3 GEMM selection requires SM{capability[0]}{capability[1]}")
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        pytest.skip("dsv3_fused_a_gemm was not built")


def _require_capability_and_cute(capability: tuple[int, int]) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != capability
    ):
        pytest.skip(f"CuTe DSL selection requires SM{capability[0]}{capability[1]}")
    if not shape_dynamic_skinny_gemm.is_available():
        pytest.skip("CuTe DSL is not available")


def _require_sm103_and_dsv3() -> None:
    _require_capability_and_dsv3((10, 3))


def _require_sm103_and_cute() -> None:
    _require_capability_and_cute((10, 3))


@pytest.mark.parametrize("spec,config", GLM_CUTE_CASES)
def test_glm_cute_selected_shapes(
    spec: glm52_gemm.GLM52ProjectionSpec,
    config: SkinnyGemmConfig,
) -> None:
    _require_sm103_and_cute()
    torch.manual_seed(42)
    x = torch.randn(
        config.num_rows,
        spec.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    plan = spec.build_plan()

    output = glm52_gemm.run_glm52_plan(plan, x, weight)

    assert output is not None
    reference = x.float() @ weight.float().t()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("n,k,num_tokens,config", QWEN4_EXP_SM90_CASES)
def test_qwen4_exp_sm90_selected_shapes(
    n: int,
    k: int,
    num_tokens: int,
    config: SkinnyGemmConfig,
) -> None:
    _require_capability_and_cute((9, 0))
    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    selected = qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS[(n, k)][num_tokens]
    assert selected == config
    output = qwen4_exp_gemm._qwen4_exp_low_latency_gemm(x, weight)

    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("n,k,num_tokens,config", QWEN4_EXP_SM100_CASES)
def test_qwen4_exp_sm100_selected_shapes(
    n: int,
    k: int,
    num_tokens: int,
    config: SkinnyGemmConfig,
) -> None:
    _require_capability_and_cute((10, 0))
    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    selected = qwen4_exp_gemm.QWEN4_EXP_SM100_GEMM_PLANS[(n, k)][num_tokens]
    assert selected == config
    output = qwen4_exp_gemm._qwen4_exp_low_latency_gemm(x, weight)

    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("n,k,num_tokens,config", QWEN4_EXP_SM121_CASES)
def test_qwen4_exp_sm121_selected_shapes(
    n: int,
    k: int,
    num_tokens: int,
    config: SkinnyGemmConfig,
) -> None:
    _require_capability_and_cute((12, 1))
    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    selected = qwen4_exp_gemm.QWEN4_EXP_SM121_GEMM_PLANS[(n, k)][num_tokens]
    assert selected == config
    output = qwen4_exp_gemm._qwen4_exp_low_latency_gemm(x, weight)

    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_glm52_q_b_nonpacked_single_row_falls_back() -> None:
    _require_sm103_and_cute()
    spec = glm52_gemm.GLM52_Q_B_PROJECTION
    storage = torch.randn(
        1,
        glm52_gemm.GLM52_QKV_A_PROJECTION.n,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = storage[:, : spec.k]
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    assert x.stride() == (glm52_gemm.GLM52_QKV_A_PROJECTION.n, 1)
    assert not glm52_gemm._runtime_ok(x, weight)
    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is None


@pytest.mark.parametrize("spec,config", GLM_CUTE_CASES)
def test_glm_cute_selected_shapes_cuda_graph_capture(
    spec: glm52_gemm.GLM52ProjectionSpec,
    config: SkinnyGemmConfig,
) -> None:
    _require_sm103_and_cute()
    x = torch.randn(
        config.num_rows,
        spec.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    plan = spec.build_plan()
    glm52_gemm.run_glm52_plan(plan, x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = glm52_gemm.run_glm52_plan(plan, x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = x.float() @ weight.float().t()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


def _dsv3_probe_tokens(tokens: frozenset[int]) -> set[int]:
    """Extremes, plus both sides of the kernel's num_tokens<=8 tile_n branch."""
    if not tokens:
        return set()
    return {min(tokens), max(tokens)} | ({8, 9} & set(tokens))


# Derived from the table rather than hand-listed, so a shape routed to dsv3
# cannot be added without being exercised here.

GLM_DSV3_CASES = [
    (num_tokens, spec)
    for spec in (
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    )
    for num_tokens in sorted(_dsv3_probe_tokens(spec.dsv3_tokens))
]


@pytest.mark.parametrize("num_tokens,spec", GLM_DSV3_CASES)
def test_glm_dsv3_selected_shapes(
    num_tokens: int,
    spec: glm52_gemm.GLM52ProjectionSpec,
) -> None:
    _require_sm103_and_dsv3()
    torch.manual_seed(42)
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_cute_residual_epilogue_all_supported_token_counts(num_tokens: int) -> None:
    _require_sm103_and_cute()
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        ShapeDynamicSkinnyGemm,
    )

    n, k = 64, 512
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
    config = ShapeDynamicSkinnyGemm._config(num_tokens, n, k)

    output = shape_dynamic_skinny_gemm(x, weight, config, residual)

    reference = x.float() @ weight.float().t() + residual.float()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)
