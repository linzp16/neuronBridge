"""Plot the attractor-ring spike raster and report native DebugMonitor coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    monitor_spikes = run_dir / "debug_monitor_ring" / "spikes.csv"
    monitor_size = monitor_spikes.stat().st_size if monitor_spikes.exists() else 0
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    rows = []
    source = "DebugMonitor"
    if monitor_spikes.exists() and monitor_size > 0:
        with monitor_spikes.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append((int(row["time_step"]), int(row["global_neuron_id"])))
    else:
        source = "ZMQ fallback"
        with (run_dir / "ring_spikes_from_zmq.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append((int(row["time_step"]), int(row["neuron_id"])))

    times = [item[0] for item in rows]
    neurons = [item[1] for item in rows]
    if neurons:
        min_neuron = min(neurons)
        max_neuron = max(neurons)
    else:
        ids = result.get("debug_monitor_neuron_ids", [64, 127])
        min_neuron, max_neuron = int(ids[0]), int(ids[1])
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.scatter(times, neurons, s=2.0, linewidths=0, rasterized=True, color="#2155a6")
    ax.set_xlim(left=0)
    ax.set_ylim(min_neuron - 0.5, max_neuron + 0.5)
    ax.set_xlabel("simulation time step")
    ax.set_ylabel("global neuron ID")
    ax.set_title(f"Attractor-ring spike raster (neurons {min_neuron}–{max_neuron})\n"
                 f"source: {source}")
    ax.grid(True, alpha=0.25)
    note = (f"ring spikes: {len(rows):,}   |   "
            f"native DebugMonitor spikes.csv: {monitor_size} bytes   |   "
            f"closed loop: {'PASS' if result.get('pass') else 'FAIL'}")
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)
    fig.savefig(run_dir / "ring_spike_raster.png", dpi=220)
    fig.savefig(run_dir / "ring_spike_raster.svg")
    print(json.dumps({
        "rows": len(rows),
        "monitor_spikes_csv_bytes": monitor_size,
        "png": str(run_dir / "ring_spike_raster.png"),
        "svg": str(run_dir / "ring_spike_raster.svg"),
    }, indent=2))


if __name__ == "__main__":
    main()
