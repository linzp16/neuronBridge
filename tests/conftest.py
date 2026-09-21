"""Early environment diagnostics for the scientific/release test suite."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def pytest_sessionstart(session) -> None:
    root = Path(__file__).resolve().parents[1]
    checker = root / "scripts" / "check_scientific_runtime.py"
    completed = subprocess.run(
        [sys.executable, str(checker)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        raise RuntimeError(
            "Scientific test environment is incompatible. "
            "Run: python scripts/check_scientific_runtime.py"
        )
