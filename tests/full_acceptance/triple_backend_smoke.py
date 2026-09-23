"""Cover the remaining exponential-triple neuron across all three backends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import neuronbridge as nb


def run_legacy(model: str) -> dict:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_current(1))
    network.add_layer(nb.NeuronLayer(model, 1, output=True, monitored=True))
    network.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=80, timestep=1.0)).init()
    sim.add_external_currents([0], [0], [12.0]).run()
    state = sim.neuron_state(1)
    spikes = sim.output_spikes()
    return {
        "model": model,
        "state": state["state_variables"],
        "finite": all(math.isfinite(float(value)) for value in state["state_variables"]),
        "spikes": len(spikes),
        "spike_times": [int(spike["time"]) for spike in spikes],
    }


def run_dense() -> dict:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_current(1))
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_triple",
            1,
            parameters={"dense_subnetwork_name": "triple_dense"},
        )
    )
    network.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=80, timestep=1.0)).init()
    sim.add_external_currents([0], [0], [12.0]).run()
    snapshot = sim.dense_subnetwork_snapshot("triple_dense")
    return {
        "model": "TimeDrivenLIF_Exponential_triple:dense_gpu",
        "state": snapshot["membrane_v"],
        "finite": all(math.isfinite(float(value)) for value in snapshot["membrane_v"]),
        "gpu_backend_ready": snapshot["gpu_backend_ready"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "legacy_cpu": run_legacy("TimeDrivenLIF_Exponential_triple"),
        "legacy_gpu": run_legacy("TimeDrivenLIF_Exponential_triple_GPU"),
        "dense_gpu": run_dense(),
    }
    report["cpu_gpu_spike_times_match"] = (
        report["legacy_cpu"]["spike_times"] == report["legacy_gpu"]["spike_times"]
    )
    report["pass"] = (
        all(value["finite"] for value in report.values() if isinstance(value, dict))
        and report["cpu_gpu_spike_times_match"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
