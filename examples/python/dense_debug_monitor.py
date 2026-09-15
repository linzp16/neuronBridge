"""Python-side visualization for the dense_debug_monitor example output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import neuronbridge as nb
from _example_paths import artifact_dir


def main() -> None:
    result_dir = artifact_dir("dense_debug_monitor")
    result = nb.open_debug_monitor(result_dir)
    print(result.meta())
    print(result.file_sizes())

    result.plot_spike_raster(limit=200_000)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
