"""Validate ELF dependencies and RPATHs in a repaired Linux wheel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


SO_PATTERN = re.compile(r"\.so(?:\.|$)", re.IGNORECASE)
RPATH_PATTERN = re.compile(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, env=env)


def inspect_wheel(path: Path, allow_missing_driver: bool) -> dict:
    failures: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        bundled_driver = [
            name for name in names
            if Path(name).name.lower() in {"libcuda.so", "libcuda.so.1"}
        ]
        if bundled_driver:
            failures.append(f"NVIDIA driver libraries must not be bundled: {bundled_driver}")

        with tempfile.TemporaryDirectory(prefix="neuronbridge_linux_wheel_") as temp:
            root = Path(temp)
            archive.extractall(root)
            native_files = [
                file for file in root.rglob("*")
                if file.is_file() and SO_PATTERN.search(file.name)
            ]
            extension = [file for file in native_files if file.name.startswith("_core")]
            if len(extension) != 1:
                failures.append(f"expected one _core shared object, found {len(extension)}")

            rpaths: dict[str, list[str]] = {}
            for native in native_files:
                result = _run(["readelf", "-d", str(native)])
                if result.returncode != 0:
                    failures.append(f"readelf failed for {native.name}: {result.stderr.strip()}")
                    continue
                values = RPATH_PATTERN.findall(result.stdout)
                rpaths[native.relative_to(root).as_posix()] = values
                for value in values:
                    for entry in value.split(":"):
                        if entry.startswith("/"):
                            failures.append(
                                f"absolute RPATH in {native.name}: {entry}"
                            )

            missing: list[str] = []
            if extension:
                library_dirs = sorted({str(file.parent) for file in native_files})
                env = os.environ.copy()
                existing = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = ":".join(
                    library_dirs + ([existing] if existing else [])
                )
                result = _run(["ldd", str(extension[0])], env=env)
                if result.returncode != 0:
                    failures.append(f"ldd failed: {result.stderr.strip()}")
                missing = [
                    line.strip() for line in result.stdout.splitlines()
                    if "=> not found" in line
                ]
                if allow_missing_driver:
                    missing = [line for line in missing if not line.startswith("libcuda.so")]
                if missing:
                    failures.append(f"unresolved shared libraries: {missing}")

    auditwheel = _run([sys.executable, "-m", "auditwheel", "show", str(path)])
    if auditwheel.returncode != 0:
        failures.append(f"auditwheel show failed: {auditwheel.stderr.strip()}")
    return {
        "result": "PASS" if not failures else "FAIL",
        "wheel": str(path.resolve()),
        "allow_missing_driver": allow_missing_driver,
        "bundled_nvidia_driver": bundled_driver,
        "rpaths": rpaths if "rpaths" in locals() else {},
        "unresolved_libraries": missing if "missing" in locals() else [],
        "auditwheel_output": auditwheel.stdout,
        "failures": failures,
    }


def main() -> int:
    if sys.platform != "linux":
        raise RuntimeError("check_linux_wheel_runtime.py must run on Linux")
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--allow-missing-driver", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_wheel(args.wheel, args.allow_missing_driver)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
