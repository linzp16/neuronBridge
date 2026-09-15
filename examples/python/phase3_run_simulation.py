"""Run a minimal native Simulation through neuronbridge.

This phase-3 example uses the public Python API to build a small network,
construct the C++ Simulation, inject an external spike, run a few steps, and
flush DebugMonitor output.
"""

from __future__ import annotations

from pathlib import Path

import neuronbridge as nb
from _example_paths import artifact_dir


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1))
    network.connect(nb.Connection(source=0, target=1, weight=1.25, max_weight=10.0, delay=1))
    return network


def main() -> None:
    output_dir = artifact_dir("phase3_minimal_run")
    network = build_network()
    sim = nb.Simulation(network, nb.SimulationConfig(steps=8, timestep=0.1))

    sim.init()
    print("initial_weight", sim.get_connection_weight(0))
    sim.set_connection_weight(0, 2.5)
    print("modified_weight", sim.get_connection_weight(0))

    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=output_dir,
            sample_interval_steps=1,
            flush_interval_steps=1,
            neuron_ids=[0, 1],
        )
    )
    sim.add_external_spikes([0], [0])
    sim.run(4)
    sim.flush()

    result = sim.result()
    print("debug_monitor_path", result.path)
    print("debug_monitor_files", sorted(result.file_sizes()))


if __name__ == "__main__":
    main()
