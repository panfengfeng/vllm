# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit model-source import closure without importing vLLM or accelerator libraries.

Run with --plan FILE to save a dry-run manifest, then --apply FILE to apply that
exact plan. Only unchanged, Git-tracked Python files in the two model trees can
be removed. All other vLLM Python files are roots, including untracked files.

This is a source dependency audit, not a proof about arbitrary runtime plugins.
Literal lazy module references and package initializers are followed; dynamic
prefixes retain their entire subtree. Unresolved imports in reachable model
code block application (except the registry's audited literal mapping loader).
External dynamic loaders are reported and remain outside the supported closure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import deque
from pathlib import Path

MODEL_TREES = ("vllm/model_executor/models/", "vllm/models/")
REGISTRY = "vllm.model_executor.models.registry"


def git(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path for path in result.stdout.decode().split("\0") if path]


def is_model(path: str) -> bool:
    return path.startswith(MODEL_TREES)


def module_name(path: str) -> str:
    name = path.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def package_name(module: str, path: str) -> str:
    return module if path.endswith("/__init__.py") else module.rpartition(".")[0]


def absolute_name(name: str, package: str) -> str:
    level = len(name) - len(name.lstrip("."))
    if not level:
        return name
    parts = package.split(".")
    if level > len(parts):
        raise ValueError(f"Relative import escapes package: {package}: {name}")
    return ".".join(parts[: len(parts) - level + 1] + [name[level:]]).rstrip(".")


def _literal_prefix(node: ast.AST, constants: dict[str, str]) -> tuple[str, bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id], True
    if isinstance(node, ast.JoinedStr):
        result = ""
        for value in node.values:
            part = value.value if isinstance(value, ast.FormattedValue) else value
            prefix, complete = _literal_prefix(part, constants)
            result += prefix
            if not complete:
                return result, False
        return result, True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, complete = _literal_prefix(node.left, constants)
        if complete:
            right, complete = _literal_prefix(node.right, constants)
            return left + right, complete
        return left, False
    return "", False


def dependencies(
    module: str, path: str, source: str, modules: dict[str, str]
) -> tuple[dict[str, str], list[dict[str, object]]]:
    tree = ast.parse(source, filename=path)
    package = package_name(module, path)
    edges: dict[str, str] = {}
    dynamic: list[dict[str, object]] = []
    constants = {"__package__": package, "__name__": module}
    dynamic_names = {"import_module", "__import__"}
    assignments: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assignments[node.id] = assignments.get(node.id, 0) + 1
        elif isinstance(node, ast.arg):
            assignments[node.arg] = assignments.get(node.arg, 0) + 1
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            dynamic_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "import_module"
            )
    for name in tuple(constants):
        if assignments.get(name):
            del constants[name]
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                continue
            value, complete = _literal_prefix(node.value, constants)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if complete:
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.isupper()
                        and assignments.get(target.id) == 1
                    ):
                        constants[target.id] = value

    def add(name: str, reason: str, required: bool = False) -> None:
        if (
            required
            and name.startswith(("vllm.models.", "vllm.model_executor.models."))
            and name not in modules
        ):
            raise ValueError(f"Missing model import {name}: {reason}")
        parts = name.split(".")
        for size in range(1, len(parts) + 1):
            target = ".".join(parts[:size])
            if target in modules and target != module:
                edges.setdefault(target, reason)

    add(package, "package initializer")
    for node in ast.walk(tree):
        reason = f"{path}:{getattr(node, 'lineno', 1)}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, f"import at {reason}", required=True)
        elif isinstance(node, ast.ImportFrom):
            base = absolute_name("." * node.level + (node.module or ""), package)
            add(base, f"from import at {reason}", required=True)
            for alias in node.names:
                if alias.name != "*":
                    add(f"{base}.{alias.name}", f"from submodule at {reason}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.split(":", 1)[0]
            if value.startswith("vllm.") and all(
                part.isidentifier() for part in value.split(".")
            ):
                add(value, f"literal module reference at {reason}")
        elif module == REGISTRY and isinstance(node, (ast.Tuple, ast.List)):
            if node.elts and isinstance(node.elts[0], ast.Constant):
                value = node.elts[0].value
                if isinstance(value, str) and value.isidentifier():
                    add(
                        f"vllm.model_executor.models.{value}",
                        f"legacy lazy registry entry at {reason}",
                    )
        elif isinstance(node, ast.Call):
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name not in dynamic_names or not node.args:
                continue
            prefix, complete = _literal_prefix(node.args[0], constants)
            if prefix.startswith("."):
                import_package = package
                package_arg = (
                    node.args[1]
                    if len(node.args) > 1
                    else next(
                        (kw.value for kw in node.keywords if kw.arg == "package"),
                        None,
                    )
                )
                if package_arg is not None:
                    import_package, known = _literal_prefix(package_arg, constants)
                    if not known:
                        prefix, complete = "", False
                elif any(kw.arg is None for kw in node.keywords):
                    prefix, complete = "", False
                if prefix:
                    prefix = absolute_name(prefix, import_package)
            if complete:
                add(prefix, f"literal dynamic import at {reason}", required=True)
            elif prefix.startswith("vllm."):
                for target in modules:
                    if target.startswith(prefix):
                        add(target, f"dynamic prefix {prefix!r} at {reason}")
            else:
                registry_loader = (
                    module == REGISTRY
                    and isinstance(node.args[0], ast.Attribute)
                    and isinstance(node.args[0].value, ast.Name)
                    and node.args[0].value.id == "self"
                    and node.args[0].attr == "module_name"
                )
                dynamic.append(
                    {
                        "path": path,
                        "line": node.lineno,
                        "expression": ast.unparse(node.args[0]),
                        "blocks_apply": is_model(path) and not registry_loader,
                        "policy": "literal registry mappings"
                        if registry_loader
                        else "unresolved runtime loader",
                    }
                )
    return edges, dynamic


def audit(root: Path, keep: tuple[str, ...] = ()) -> dict[str, object]:
    root = root.resolve()
    tracked = set(git(root, "ls-files", "-z", "--", "vllm"))
    paths = tracked | set(
        git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", "vllm")
    )
    paths = {path for path in paths if path.endswith(".py")}
    modules = {module_name(path): path for path in sorted(paths)}
    if len(modules) != len(paths):
        raise ValueError("Ambiguous module/package names in source inventory")
    contents: dict[str, bytes] = {}
    for path in sorted(paths):
        file = root / path
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError(f"Refusing symlink or escaped source: {path}")
        if file.exists():
            contents[path] = file.read_bytes()
    if not contents:
        raise ValueError("No vLLM Python sources found")
    roots = {
        module
        for module, path in modules.items()
        if path in contents and (not is_model(path) or path not in tracked)
    }
    for name in keep:
        if name not in modules or modules[name] not in contents:
            raise ValueError(f"Explicit kept module does not exist: {name}")
        roots.add(name)
    reasons = {module: "external source root" for module in roots}
    reasons.update({module: "explicit keep" for module in keep})
    pending = deque(sorted(roots))
    visited: set[str] = set()
    missing: list[dict[str, str]] = []
    dynamic = []
    while pending:
        module = pending.popleft()
        if module in visited:
            continue
        visited.add(module)
        path = modules[module]
        if path not in contents:
            missing.append({"path": path, "reason": reasons[module]})
            continue
        edges, unresolved = dependencies(
            module, path, contents[path].decode("utf-8-sig"), modules
        )
        dynamic.extend(unresolved)
        for target, reason in sorted(edges.items()):
            reasons.setdefault(target, f"{module}: {reason}")
            if target not in visited:
                pending.append(target)
    candidates = sorted(path for path in contents if path in tracked and is_model(path))
    kept = [path for path in candidates if module_name(path) in visited]
    deleted = [path for path in candidates if module_name(path) not in visited]
    return {
        "version": 1,
        "root": str(root),
        "keep_modules": list(keep),
        "scope": list(MODEL_TREES),
        "source_sha256": {
            path: hashlib.sha256(data).hexdigest()
            for path, data in sorted(contents.items())
        },
        "kept": [
            {
                "path": path,
                "bytes": len(contents[path]),
                "reason": reasons[module_name(path)],
            }
            for path in kept
        ],
        "delete": [
            {
                "path": path,
                "bytes": len(contents[path]),
                "reason": "unreachable from external roots and literal references",
            }
            for path in deleted
        ],
        "summary": {
            "roots": len(roots),
            "kept_files": len(kept),
            "kept_bytes": sum(len(contents[path]) for path in kept),
            "delete_files": len(deleted),
            "delete_bytes": sum(len(contents[path]) for path in deleted),
        },
        "missing_dependencies": sorted(missing, key=lambda entry: entry["path"]),
        "dynamic_imports": sorted(
            dynamic, key=lambda entry: (entry["path"], entry["line"])
        ),
    }


def apply_plan(root: Path, plan: dict[str, object]) -> int:
    current = audit(root, tuple(plan.get("keep_modules", ())))
    if current != plan:
        raise ValueError("Plan is stale or modified; generate and review a new plan")
    if current["missing_dependencies"] or any(
        entry["blocks_apply"] for entry in current["dynamic_imports"]
    ):
        raise ValueError("Unresolved model imports block pruning; inspect the plan")
    deleted = [entry["path"] for entry in current["delete"]]
    dirty = set(git(root, "diff", "--name-only", "-z", "HEAD", "--", *MODEL_TREES))
    changed = dirty.intersection(deleted)
    if changed:
        raise ValueError(
            f"Refusing to delete modified tracked files: {sorted(changed)}"
        )
    for path in deleted:
        file = root / path
        if (
            file.is_symlink()
            or hashlib.sha256(file.read_bytes()).hexdigest()
            != current["source_sha256"][path]
        ):
            raise ValueError(f"Source changed after audit: {path}")
    for path in deleted:
        (root / path).unlink()
    return len(deleted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--keep", action="append", default=[], metavar="MODULE")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", type=Path, help="Write a dry-run JSON manifest")
    action.add_argument("--apply", type=Path, help="Apply this exact reviewed manifest")
    action.add_argument(
        "--check", action="store_true", help="Validate retained imports"
    )
    args = parser.parse_args()
    try:
        if args.apply:
            if args.keep:
                parser.error("--keep is recorded in the plan; do not use with --apply")
            plan = json.loads(args.apply.read_text())
            print(json.dumps({"deleted_files": apply_plan(args.root, plan)}))
            return 0
        report = audit(args.root, tuple(args.keep))
        if args.plan:
            args.plan.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
        blockers = report["missing_dependencies"] + [
            entry for entry in report["dynamic_imports"] if entry["blocks_apply"]
        ]
        if blockers:
            print(json.dumps({"blockers": blockers}, indent=2))
        print(f"Unresolved runtime loaders: {len(report['dynamic_imports'])}")
        return int(bool(blockers))
    except (ValueError, OSError, SyntaxError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
