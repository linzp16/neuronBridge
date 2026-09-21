"""Check repository-local quoted includes for Linux case sensitivity issues."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh"}
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
INCLUDE_ROOTS = (
    PurePosixPath("."),
    PurePosixPath("include"),
    PurePosixPath("src/native/legacy"),
    PurePosixPath("src/native/legacy/source_file_realtime_v1_async"),
    PurePosixPath("src/native/shared/include"),
    PurePosixPath("src/native/gpu/include"),
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def _normalise(path: PurePosixPath) -> str:
    return posixpath.normpath(path.as_posix()).lstrip("./")


def main() -> int:
    tracked = _tracked_files()
    by_lower = {name.lower(): name for name in tracked}
    failures: list[str] = []
    for source_name in tracked:
        source = ROOT / source_name
        if source.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            failures.append(f"{source_name}: cannot read: {exc}")
            continue
        source_parent = PurePosixPath(source_name).parent
        for line_number, line in enumerate(lines, 1):
            match = INCLUDE_PATTERN.match(line)
            if not match:
                continue
            include = PurePosixPath(match.group(1).replace("\\", "/"))
            candidates = [source_parent / include]
            candidates.extend(root / include for root in INCLUDE_ROOTS)
            for candidate in candidates:
                requested = _normalise(candidate)
                actual = by_lower.get(requested.lower())
                if actual is None:
                    continue
                if requested != actual:
                    failures.append(
                        f"{source_name}:{line_number}: {match.group(1)} -> {actual}"
                    )
                break
    if failures:
        print("include case check FAIL")
        print("\n".join(failures))
        return 1
    print("include case check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
