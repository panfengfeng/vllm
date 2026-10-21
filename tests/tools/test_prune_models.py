# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect import closure and user files when pruning tracked model sources.

Temporary Git repositories exercise the tool without importing vLLM or torch.
"""

import subprocess
from pathlib import Path

import pytest

from tools.prune_models import apply_plan, audit


def repository(tmp_path: Path, sources: dict[str, str]) -> Path:
    for path, text in sources.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Pruning Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    return tmp_path


def test_prune_preserves_imports_initializers_and_untracked_files(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/__init__.py": "",
            "vllm/engine.py": "from vllm.models.target import model\n",
            "vllm/models/__init__.py": "",
            "vllm/models/target/__init__.py": "from . import registration\n",
            "vllm/models/target/model.py": "from .. import shared\n",
            "vllm/models/target/registration.py": "register()\n",
            "vllm/models/shared.py": "",
            "vllm/models/spare.py": "",
            "vllm/models/unused.py": "",
            "vllm/models/data.json": "{}",
        },
    )
    local = root / "vllm/models/local.py"
    local.write_text("from vllm.models import spare\n")
    report = audit(root)
    assert [entry["path"] for entry in report["delete"]] == ["vllm/models/unused.py"]
    assert report["summary"]["delete_files"] == 1
    assert report["summary"]["kept_files"] == 6
    assert apply_plan(root, report) == 1
    assert not (root / "vllm/models/unused.py").exists()
    assert local.exists()
    assert (root / "vllm/models/data.json").exists()
    assert not audit(root)["missing_dependencies"]


def test_lazy_registry_and_dynamic_prefix_retain_possible_targets(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "from vllm.model_executor.models import registry\n",
            "vllm/model_executor/models/__init__.py": "",
            "vllm/model_executor/models/registry.py": (
                "MODELS = {'Target': ('target', 'Model')}\n"
                "BACKEND = 'vllm.models.target.model:Backend'\n"
                "def load(self):\n"
                "    return importlib.import_module(self.module_name)\n"
            ),
            "vllm/model_executor/models/target.py": (
                "from importlib import import_module as load\n"
                "def extension(name):\n"
                "    return load(f'vllm.models.plugins.{name}')\n"
            ),
            "vllm/models/target/model.py": "",
            "vllm/models/plugins/a.py": "",
            "vllm/models/plugins/b.py": "",
            "vllm/models/unused.py": "",
        },
    )
    report = audit(root)
    assert [entry["path"] for entry in report["delete"]] == ["vllm/models/unused.py"]
    assert not any(entry["blocks_apply"] for entry in report["dynamic_imports"])


@pytest.mark.parametrize("change", ["source", "manifest"])
def test_apply_rejects_stale_or_forged_plan_before_deleting(tmp_path, change):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "",
            "vllm/models/unused.py": "",
        },
    )
    report = audit(root)
    if change == "source":
        (root / "vllm/engine.py").write_text("import vllm.models.unused\n")
    else:
        report["delete"].append({"path": "vllm/engine.py", "bytes": 0})
    with pytest.raises(ValueError, match="stale or modified"):
        apply_plan(root, report)
    assert (root / "vllm/models/unused.py").exists()


def test_apply_refuses_modified_candidate_even_with_fresh_plan(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "",
            "vllm/models/unused.py": "",
        },
    )
    (root / "vllm/models/unused.py").write_text("user_work = True\n")
    with pytest.raises(ValueError, match="modified tracked"):
        apply_plan(root, audit(root))


def test_unresolved_model_loader_blocks_application(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "import vllm.models.target\n",
            "vllm/models/target.py": "importlib.import_module(runtime_name)\n",
            "vllm/models/unused.py": "",
        },
    )
    with pytest.raises(ValueError, match="Unresolved model imports"):
        apply_plan(root, audit(root))
    assert (root / "vllm/models/unused.py").exists()


def test_dynamic_module_name_with_multiple_assignments_is_not_guessed(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "import vllm.models.target\n",
            "vllm/models/target.py": (
                "PREFIX = 'vllm.models.a'\n"
                "if runtime_condition:\n"
                "    PREFIX = 'vllm.models.b'\n"
                "importlib.import_module(PREFIX)\n"
            ),
            "vllm/models/a.py": "",
            "vllm/models/b.py": "",
        },
    )
    with pytest.raises(ValueError, match="Unresolved model imports"):
        apply_plan(root, audit(root))


def test_relative_dynamic_import_preserves_package_subtree(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "import vllm.models.target\n",
            "vllm/models/target/__init__.py": (
                "from importlib import import_module\n"
                "def load(name):\n"
                "    return import_module(f'.{name}', __name__)\n"
            ),
            "vllm/models/target/a.py": "",
            "vllm/models/target/b.py": "",
            "vllm/models/unused.py": "",
        },
    )
    assert [entry["path"] for entry in audit(root)["delete"]] == [
        "vllm/models/unused.py"
    ]


def test_missing_model_import_fails_closure_validation(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "import vllm.models.missing\n",
            "vllm/models/unused.py": "",
        },
    )
    with pytest.raises(ValueError, match="Missing model import"):
        audit(root)


def test_relative_dynamic_keyword_package_preserves_target_subtree(tmp_path):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "import vllm.models.target\n",
            "vllm/models/target.py": (
                "from importlib import import_module\n"
                "def load(name):\n"
                "    return import_module(f'.{name}', package='vllm.models.plugins')\n"
            ),
            "vllm/models/plugins/__init__.py": "",
            "vllm/models/plugins/a.py": "",
            "vllm/models/plugins/b.py": "",
            "vllm/models/unused.py": "",
        },
    )
    assert [entry["path"] for entry in audit(root)["delete"]] == [
        "vllm/models/unused.py"
    ]


@pytest.mark.parametrize(
    "package_arg", ["package=runtime_package", "runtime_package", "**runtime_kwargs"]
)
def test_unknown_relative_import_package_blocks_pruning(tmp_path, package_arg):
    root = repository(
        tmp_path,
        {
            "vllm/engine.py": "import vllm.models.target\n",
            "vllm/models/target.py": (
                f"importlib.import_module('.worker', {package_arg})\n"
            ),
            "vllm/models/unused.py": "",
        },
    )
    with pytest.raises(ValueError, match="Unresolved model imports"):
        apply_plan(root, audit(root))


def test_symlink_sources_cannot_escape_repository(tmp_path):
    root = repository(tmp_path, {"vllm/engine.py": ""})
    (root / "outside.py").write_text("important = True\n")
    (root / "vllm/symlink.py").symlink_to(root / "outside.py")
    with pytest.raises(ValueError, match="Refusing symlink"):
        audit(root)
