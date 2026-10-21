#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  UV_VERSION="0.11.28"
  UV_BIN_DIR="$ROOT_DIR/.ci-tools"
  curl -L --proto '=https' --tlsv1.2 -sSf \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" \
    | env UV_INSTALL_DIR="$UV_BIN_DIR" sh
  export PATH="$UV_BIN_DIR:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python pytest 'cmake>=3.26.1'

.venv/bin/python -m pytest --confcutdir=tests/standalone_tests \
  tests/standalone_tests/test_deepseek_v41_*.py \
  tests/standalone_tests/test_precompiled_wheel.py -q
PYTHONPATH=. .venv/bin/python -m pytest --confcutdir=tests/tools \
  tests/tools/test_prune_models.py -q
.venv/bin/python -m pytest --noconftest tests/test_cmake_utils.py -q
.venv/bin/python tools/prune_models.py --check
