"""Inspect a NeuronBridge wheel without importing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


REQUIRED_PATTERNS = {
    "extension": re.compile(r"^neuronbridge/_core.+\.pyd$", re.IGNORECASE),
    "cuda_runtime": re.compile(r"^neuronbridge(?:/|\.libs/)cudart64_.+\.dll$", re.IGNORECASE),
    "pinocchio": re.compile(r"^neuronbridge(?:/|\.libs/)pinocchio_default(?:-.+)?\.dll$", re.IGNORECASE),
    "zeromq": re.compile(r"^neuronbridge(?:/|\.libs/)libzmq.+\.dll$", re.IGNORECASE),
}


def inspect_wheel(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [PurePosixPath(item.filename).as_posix() for item in entries]
        unsafe = [name for name in names if name.startswith("/") or ".." in PurePosixPath(name).parts]
        forbidden = [
            name
            for name in names
            if name.lower().endswith((".lib", ".pdb", ".exp", ".obj"))
            or "/release/" in f"/{name.lower()}/"
            or "__pycache__" in name.lower()
        ]
        matches = {
            key: [name for name in names if pattern.match(name)]
            for key, pattern in REQUIRED_PATTERNS.items()
        }
        missing = [key for key, values in matches.items() if not values]
        dlls = sorted(name for name in names if name.lower().endswith(".dll"))
        report = {
            "result": "PASS" if not (unsafe or forbidden or missing) else "FAIL",
            "wheel": str(path.resolve()),
            "entry_count": len(entries),
            "size_bytes": path.stat().st_size,
            "native_dll_count": len(dlls),
            "required": matches,
            "unsafe_entries": unsafe,
            "forbidden_entries": forbidden,
            "missing_requirements": missing,
            "dlls": dlls,
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
