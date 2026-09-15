"""Shared repository paths for the source-tree examples."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def artifact_dir(name: str) -> Path:
    return REPOSITORY_ROOT / "artifacts" / "examples" / name


def data_root() -> Path:
    configured = os.environ.get("NEURONBRIDGE_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else REPOSITORY_ROOT / "examples" / "data"


def data_dir(name: str) -> Path:
    return data_root() / name / "input_data"
