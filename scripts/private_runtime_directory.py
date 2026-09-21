"""Privatize MSVC/OpenMP runtime dependencies in a Windows package directory.

The public Microsoft runtime DLL names are process-global on Windows.  Native
Python packages that ship different copies can therefore bind to whichever
copy was loaded first.  NeuronBridge avoids that collision by giving its
app-local runtime DLLs package-private names and rewriting every bundled PE
file to reference those names.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


RUNTIME_RENAMES = {
    "msvcp140.dll": "neuronbridge_msvcp140.dll",
    "msvcp140_1.dll": "neuronbridge_msvcp140_1.dll",
    "msvcp140_2.dll": "neuronbridge_msvcp140_2.dll",
    "msvcp140_atomic_wait.dll": "neuronbridge_msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll": "neuronbridge_msvcp140_codecvt_ids.dll",
    "vcomp140.dll": "neuronbridge_vcomp140.dll",
    "concrt140.dll": "neuronbridge_concrt140.dll",
    "vcruntime140.dll": "neuronbridge_vcruntime140.dll",
    "vcruntime140_1.dll": "neuronbridge_vcruntime140_1.dll",
}

NATIVE_SUFFIXES = {".dll", ".pyd"}


def _require_delvewheel() -> None:
    if importlib.util.find_spec("delvewheel") is None:
        raise RuntimeError(
            "delvewheel is required to privatize Windows runtime DLLs. "
            "Install it with: python -m pip install 'delvewheel>=1.9'"
        )


def _run_delvewheel(*arguments: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "delvewheel", *arguments]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"delvewheel command failed: {' '.join(command)}"
            + (f"\n{details}" if details else "")
        ) from error


def needed_libraries(path: Path) -> set[str]:
    result = _run_delvewheel("needed", str(path))
    return {
        line.strip().lower()
        for line in result.stdout.splitlines()
        if line.strip()
    }


def _native_files(package_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in NATIVE_SUFFIXES
    )


def _files_by_lower_name(package_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in _native_files(package_dir):
        name = path.name.lower()
        if name in result and path.resolve() != result[name].resolve():
            raise RuntimeError(
                f"duplicate native dependency name {path.name}: "
                f"{result[name]} and {path}"
            )
        result[name] = path
    return result


def _replace_needed(path: Path, old: str, new: str) -> None:
    _run_delvewheel(
        "replace-needed",
        "-change",
        old,
        new,
        "--strip",
        str(path),
    )


def privatize_directory(package_dir: Path) -> dict[str, object]:
    """Rewrite one assembled package directory in place and verify closure."""
    if sys.platform != "win32":
        raise RuntimeError("private runtime rewriting is Windows-only")
    _require_delvewheel()

    package_dir = package_dir.resolve()
    if not package_dir.is_dir():
        raise RuntimeError(f"package directory does not exist: {package_dir}")

    native_files = _native_files(package_dir)
    if not any(path.suffix.lower() == ".pyd" for path in native_files):
        raise RuntimeError(f"no Python extension was found in {package_dir}")

    dependency_map = {path: needed_libraries(path) for path in native_files}
    required_public_names = sorted(
        {
            dependency
            for dependencies in dependency_map.values()
            for dependency in dependencies
            if dependency in RUNTIME_RENAMES
        }
    )
    files_by_name = _files_by_lower_name(package_dir)
    for old_name in required_public_names:
        private_name = RUNTIME_RENAMES[old_name]
        if old_name not in files_by_name and private_name not in files_by_name:
            raise RuntimeError(
                f"{old_name} is required by a bundled native module, but neither "
                f"it nor {private_name} exists in {package_dir}"
            )

    patched: list[str] = []
    for path, dependencies in dependency_map.items():
        changed = False
        for old_name, private_name in RUNTIME_RENAMES.items():
            if old_name in dependencies:
                _replace_needed(path, old_name, private_name)
                changed = True
        if changed:
            patched.append(path.relative_to(package_dir).as_posix())

    renamed: list[dict[str, str]] = []
    files_by_name = _files_by_lower_name(package_dir)
    for old_name, private_name in RUNTIME_RENAMES.items():
        old_path = files_by_name.get(old_name)
        if old_path is None:
            continue
        private_path = old_path.with_name(private_name)
        os.replace(old_path, private_path)
        renamed.append({"from": old_name, "to": private_name})

    public_files = [
        path.relative_to(package_dir).as_posix()
        for path in _native_files(package_dir)
        if path.name.lower() in RUNTIME_RENAMES
    ]
    public_dependencies: dict[str, list[str]] = {}
    for path in _native_files(package_dir):
        dependencies = sorted(needed_libraries(path).intersection(RUNTIME_RENAMES))
        if dependencies:
            public_dependencies[path.relative_to(package_dir).as_posix()] = dependencies

    if public_files or public_dependencies:
        raise RuntimeError(
            "runtime privatization left process-global dependencies: "
            + json.dumps(
                {
                    "public_runtime_files": public_files,
                    "public_runtime_dependencies": public_dependencies,
                },
                indent=2,
            )
        )

    return {
        "package_directory": str(package_dir),
        "patched_native_files": patched,
        "renamed_runtime_files": renamed,
        "public_runtime_files": public_files,
        "public_runtime_dependencies": public_dependencies,
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = privatize_directory(args.package_dir)
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
