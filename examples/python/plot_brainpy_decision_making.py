"""Plot raster and population rates for brainpy_decision_making_streaming.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DT_MS = 0.1
BIN_MS = 10.0
N_A = 240
N_B = 240
TOTAL_MS = 1600.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input_dir / "decision_making_raster_rates.png"

    raster: dict[str, tuple[list[float], list[int]]] = {
        "A": ([], []),
        "B": ([], []),
    }
    bin_count = int(TOTAL_MS / BIN_MS)
    binned = {"A": [0] * bin_count, "B": [0] * bin_count}
    with (args.input_dir / "spikes.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            population = row["population"]
            if population not in raster:
                continue
            time_ms = float(row["time_ms"])
            local_id = int(row["population_index"])
            raster[population][0].append(time_ms)
            raster[population][1].append(local_id)
            bin_index = min(bin_count - 1, int(time_ms / BIN_MS))
            binned[population][bin_index] += 1

    stimulus_times = [0.0]
    stimulus_a = [0.0]
    stimulus_b = [0.0]
    with (args.input_dir / "stimulus_rates.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stimulus_times.append(float(row["time_ms"]))
            stimulus_a.append(float(row["rate_a_hz"]))
            stimulus_b.append(float(row["rate_b_hz"]))
    stimulus_times.append(TOTAL_MS)
    stimulus_a.append(0.0)
    stimulus_b.append(0.0)

    bin_centers = [(index + 0.5) * BIN_MS for index in range(bin_count)]
    rate_a = [count / (N_A * BIN_MS / 1000.0) for count in binned["A"]]
    rate_b = [count / (N_B * BIN_MS / 1000.0) for count in binned["B"]]

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    colors = {"A": "#2878B5", "B": "#D95319"}
    for axis, population in zip(axes[:2], ("A", "B")):
        axis.scatter(
            raster[population][0],
            raster[population][1],
            s=1.5,
            color=colors[population],
            linewidths=0,
            rasterized=True,
        )
        axis.set_ylabel(f"{population} neuron")
        axis.set_ylim(-1, 241)
        axis.set_title(f"Selective population {population} raster")

    axes[2].plot(bin_centers, rate_a, color=colors["A"], label="A", linewidth=1.4)
    axes[2].plot(bin_centers, rate_b, color=colors["B"], label="B", linewidth=1.4)
    axes[2].set_ylabel("Population rate (Hz)")
    axes[2].set_title(f"Population firing rates ({BIN_MS:g} ms bins)")
    axes[2].legend()

    axes[3].step(stimulus_times, stimulus_a, where="post", color=colors["A"], label="IA")
    axes[3].step(stimulus_times, stimulus_b, where="post", color=colors["B"], label="IB")
    axes[3].set_ylabel("Input rate (Hz)")
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_title("Decision stimulus")
    axes[3].legend()

    for axis in axes:
        axis.axvline(100.0, color="0.25", linestyle="--", linewidth=0.8)
        axis.axvline(1100.0, color="0.25", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.set_xlim(0.0, TOTAL_MS)
    fig.suptitle("NeuronBridge decision-making network (BrainPy-style)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
