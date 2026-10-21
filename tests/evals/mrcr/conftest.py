# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--config-list-file",
        help="Path to a file listing evaluation YAMLs, one per line",
    )


def pytest_generate_tests(metafunc):
    if "config_filename" not in metafunc.fixturenames:
        return
    configured_path = metafunc.config.getoption("--config-list-file")
    if not configured_path:
        raise pytest.UsageError(
            "Pass --config-list-file with an explicit evaluation configuration list."
        )
    list_path = Path(configured_path).expanduser()
    if not list_path.is_absolute():
        local_path = Path(__file__).parent / list_path
        list_path = local_path if local_path.is_file() else Path.cwd() / list_path
    if not list_path.is_file():
        raise pytest.UsageError(f"Evaluation config list does not exist: {list_path}")

    configs = [
        list_path.parent / entry
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if (entry := line.strip()) and not entry.startswith("#")
    ]
    if not configs:
        raise pytest.UsageError(f"Evaluation config list is empty: {list_path}")
    for path in configs:
        if not path.is_file():
            raise pytest.UsageError(f"Evaluation config does not exist: {path}")
    metafunc.parametrize(
        "config_filename", configs, ids=[path.stem for path in configs]
    )
