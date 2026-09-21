"""Check that a wheel does not carry process-global MSVC/OpenMP DLLs."""

from __future__ import annotations

import argparse
import json
import zipfile


FORBIDDEN = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcomp140.dll",
    "concrt140.dll",
)


def inspect(path: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
    forbidden = [
        entry for entry in entries
        if entry.lower().rsplit("/", 1)[-1] in FORBIDDEN
    ]
    private_runtime = [
        entry for entry in entries
        if entry.lower().rsplit("/", 1)[-1].startswith("neuronbridge_")
        and entry.lower().endswith(".dll")
    ]
    return {
        "wheel": path,
        "forbidden_process_global_runtime_dlls": forbidden,
        "private_runtime_dlls": sorted(private_runtime),
        "pass": not forbidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect(args.wheel)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
