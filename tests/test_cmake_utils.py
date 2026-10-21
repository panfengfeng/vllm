# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validate_target():
    """Run the setup device policy without invoking packaging or importing Torch."""
    setup_path = Path(__file__).parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text())
    policy = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_target_device"
    )
    namespace = {
        "sys": SimpleNamespace(platform="linux"),
        "torch": SimpleNamespace(
            version=SimpleNamespace(cuda="13.0", hip=None, xpu=None)
        ),
    }
    exec(
        compile(ast.Module(body=[policy], type_ignores=[]), setup_path, "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("target", ["cpu", "rocm", "xpu", "tpu", "hpu", "unknown"])
def test_setup_rejects_removed_backend_before_build(validate_target, target):
    with pytest.raises(ValueError, match="only CUDA builds"):
        validate_target["_validate_target_device"](target)


def test_setup_accepts_linux_cuda_build(validate_target):
    validate_target["_validate_target_device"]("cuda")


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_setup_rejects_cuda_build_on_unsupported_host(validate_target, platform):
    validate_target["sys"].platform = platform
    with pytest.raises(RuntimeError, match="require Linux"):
        validate_target["_validate_target_device"]("cuda")


@pytest.mark.parametrize(
    "cuda,hip,xpu", [(None, None, None), (None, "7.2", None), (None, None, "2026")]
)
def test_setup_rejects_non_cuda_torch_build(validate_target, cuda, hip, xpu):
    validate_target["torch"].version = SimpleNamespace(cuda=cuda, hip=hip, xpu=xpu)
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        validate_target["_validate_target_device"]("cuda")


def test_empty_installation_does_not_require_cuda(validate_target):
    validate_target["sys"].platform = "darwin"
    validate_target["torch"].version = SimpleNamespace(cuda=None, hip=None, xpu=None)
    validate_target["_validate_target_device"]("empty")


def _get_cmake_bin() -> str:
    cmake = shutil.which("cmake")
    if cmake:
        return cmake
    venv_cmake = Path(sys.executable).parent / "cmake"
    if venv_cmake.is_file():
        return str(venv_cmake)
    return "cmake"


@pytest.mark.parametrize("target", ["cpu", "rocm", "xpu", "tpu"])
def test_cmake_rejects_removed_backend_before_dependency_discovery(tmp_path, target):
    result = subprocess.run(
        [
            _get_cmake_bin(),
            "-S",
            str(Path(__file__).parents[1]),
            "-B",
            str(tmp_path / "build"),
            f"-DVLLM_TARGET_DEVICE={target}",
            "-DVLLM_PYTHON_EXECUTABLE=/missing/python",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "DeepSeek-V4.1" in result.stderr
    assert "CUDA" in result.stderr
    assert "Could NOT find Python" not in result.stderr


def test_cmake_empty_target_does_not_discover_cuda_or_torch(tmp_path):
    result = subprocess.run(
        [
            _get_cmake_bin(),
            "-S",
            str(Path(__file__).parents[1]),
            "-B",
            str(tmp_path / "build"),
            "-DVLLM_TARGET_DEVICE=empty",
            "-DVLLM_PYTHON_EXECUTABLE=/missing/python",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "skipping native extensions" in result.stdout


def test_exact_family_arch_precedes_generic_family_fallback(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_cuda_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
cuda_archs_loose_intersection(
  actual "10.0f;10.7f" "10.7")
if(NOT "${{actual}}" STREQUAL "10.7f")
  message(FATAL_ERROR "Expected 10.7f, got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_extract_archs_prefers_sass_target_over_corrupted_virtual_arch(
    tmp_path: Path,
):
    """Torch's autodetection can emit a bogus arch=compute_* half (e.g.
    capability 12.1 corrupted to arch=compute_20,code=sm_121); the SASS
    target must win, while PTX-only entries keep the virtual arch."""
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_extract_archs.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
extract_unique_cuda_archs_ascending(actual
  "-gencode arch=compute_20,code=sm_121;\
-gencode arch=compute_80,code=sm_80;\
-gencode arch=compute_80,code=compute_80")
if(NOT "${{actual}}" STREQUAL "8.0;12.1")
  message(FATAL_ERROR "Expected '8.0;12.1', got '${{actual}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)


def test_clear_cuda_gencode_flags(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    script = tmp_path / "test_clear_flags.cmake"
    script.write_text(
        f"""
cmake_minimum_required(VERSION 3.26)
include("{repo_root / "cmake" / "utils.cmake"}")
set(CMAKE_CUDA_FLAGS "-Wall -gencode arch=compute_80,code=sm_80")
clear_cuda_gencode_flags(CUDA_ARCH_FLAGS)
if(NOT "${{CMAKE_CUDA_FLAGS}}" STREQUAL "-Wall ")
  message(FATAL_ERROR "Expected '-Wall ', got '${{CMAKE_CUDA_FLAGS}}'")
endif()
if(NOT "${{CUDA_ARCH_FLAGS}}" STREQUAL "-gencode arch=compute_80,code=sm_80")
  message(FATAL_ERROR "Expected '-gencode arch=compute_80,code=sm_80', "
    "got '${{CUDA_ARCH_FLAGS}}'")
endif()
"""
    )

    subprocess.run([_get_cmake_bin(), "-P", script], check=True)
