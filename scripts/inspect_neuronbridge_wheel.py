"""Inspect a NeuronBridge wheel without importing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


WINDOWS_REQUIRED_PATTERNS = {
    "extension": re.compile(r"^neuronbridge/_core.+\.pyd$", re.IGNORECASE),
    "cuda_runtime": re.compile(r"^neuronbridge(?:/|\.libs/)cudart64_.+\.dll$", re.IGNORECASE),
    "pinocchio": re.compile(r"^neuronbridge(?:/|\.libs/)pinocchio_default(?:-.+)?\.dll$", re.IGNORECASE),
    "zeromq": re.compile(r"^neuronbridge(?:/|\.libs/)libzmq.+\.dll$", re.IGNORECASE),
}

LINUX_REQUIRED_PATTERNS = {
    "extension": re.compile(
        r"^neuronbridge/_core(?:\.[^/]+)?\.so$", re.IGNORECASE
    ),
}


def _wheel_platform(path: Path, names: list[str]) -> str:
    filename = path.name.lower()
    if "win_amd64" in filename or any(name.lower().endswith(".pyd") for name in names):
        return "windows"
    if "manylinux" in filename or "linux_" in filename or any(
        name.lower().endswith(".so") for name in names
    ):
        return "linux"
    return "unknown"


def inspect_wheel(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [PurePosixPath(item.filename).as_posix() for item in entries]
        platform = _wheel_platform(path, names)
        unsafe = [name for name in names if name.startswith("/") or ".." in PurePosixPath(name).parts]
        forbidden = [
            name
            for name in names
            if name.lower().endswith((".lib", ".pdb", ".exp", ".obj"))
            or "/release/" in f"/{name.lower()}/"
            or "__pycache__" in name.lower()
        ]
        if platform == "linux":
            forbidden.extend(
                name
                for name in names
                if name.lower().endswith((".a", ".o"))
                or PurePosixPath(name).name.lower() in {"libcuda.so", "libcuda.so.1"}
            )
        required_patterns = (
            WINDOWS_REQUIRED_PATTERNS
            if platform == "windows"
            else LINUX_REQUIRED_PATTERNS
        )
        matches = {
            key: [name for name in names if pattern.match(name)]
            for key, pattern in required_patterns.items()
        }
        missing = [key for key, values in matches.items() if not values]
        dlls = sorted(name for name in names if name.lower().endswith(".dll"))
        shared_objects = sorted(
            name for name in names
            if re.search(r"\.so(?:\.|$)", PurePosixPath(name).name, re.IGNORECASE)
        )
        report = {
            "result": "PASS" if not (unsafe or forbidden or missing) else "FAIL",
            "wheel": str(path.resolve()),
            "platform": platform,
            "entry_count": len(entries),
            "size_bytes": path.stat().st_size,
            "native_dll_count": len(dlls),
            "native_shared_object_count": len(shared_objects),
            "required": matches,
            "unsafe_entries": unsafe,
            "forbidden_entries": forbidden,
            "missing_requirements": missing,
            "dlls": dlls,
            "shared_objects": shared_objects,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_wheel(args.wheel)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
