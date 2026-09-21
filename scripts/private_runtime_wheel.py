"""Give bundled MSVC/OpenMP DLLs package-private names in a Windows wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import zipfile

from private_runtime_directory import privatize_directory


def _record_line(path: str, data: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [path, f"sha256={digest}", str(len(data))]


def rewrite_wheel(wheel: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="neuronbridge_private_runtime_") as temp:
        root = Path(temp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        package_dir = root / "neuronbridge"
        if not package_dir.is_dir():
            raise RuntimeError("wheel has no neuronbridge package directory")

        privatize_directory(package_dir)

        record = next(root.glob("*.dist-info/RECORD"), None)
        if record is None:
            raise RuntimeError("wheel has no dist-info/RECORD")
        record_rel = record.relative_to(root).as_posix()
        rows = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            if rel != record_rel:
                rows.append(_record_line(rel, path.read_bytes()))
        rows.append([record_rel, "", ""])
        with record.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                archive.write(path, path.relative_to(root).as_posix())
        os.replace(temporary, output)


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("private_runtime_wheel.py is Windows-only")
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    rewrite_wheel(wheel, (args.output or wheel).resolve())
    print(f"Private runtime wheel written: {args.output or wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
