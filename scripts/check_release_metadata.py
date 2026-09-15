"""Validate release metadata shared by CMake and Python packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tomllib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-license", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = tomllib.loads((root / "python" / "pyproject.toml").read_text(encoding="utf-8"))
    python_version = pyproject["project"]["version"]
    init_text = (root / "python" / "src" / "neuronbridge" / "__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    options_text = (root / "cmake" / "NeuronBridgeOptions.cmake").read_text(encoding="utf-8")
    cmake_match = re.search(r'set\(NR_VERSION_FULL\s+"([^"]+)"', options_text)
    versions = {
        "VERSION": version,
        "pyproject": python_version,
        "python_package": init_match.group(1) if init_match else None,
        "cmake": cmake_match.group(1) if cmake_match else None,
    }
    license_present = (root / "LICENSE").is_file()
    failures = []
    if len(set(versions.values())) != 1:
        failures.append(f"version mismatch: {versions}")
    if args.require_license and not license_present:
        failures.append("project-level LICENSE is required for public release")
    report = {
        "result": "PASS" if not failures else "FAIL",
        "versions": versions,
        "project_license_present": license_present,
        "public_release_ready": license_present and not failures,
        "failures": failures,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
