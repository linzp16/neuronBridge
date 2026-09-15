"""Python migration of the dense_mixed baseline."""

from __future__ import annotations

from pathlib import Path

import neuronbridge as nb

from _dense_baseline_builders import build_dense_mixed_network
from _example_paths import artifact_dir


def main() -> None:
    output_dir = artifact_dir("dense_mixed") / "with_relay"
    sim = nb.Simulation(build_dense_mixed_network(connect_relay_to_dense=True), nb.SimulationConfig(steps=300, timestep=0.1))
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=output_dir,
            sample_interval_steps=1,
            flush_interval_steps=10,
            all_neurons=True,
            record_state=True,
            record_spikes=True,
            record_pending_channels=True,
            record_weights=False,
            record_outer_dynamic_state=False,
            record_inputconv_outputs=False,
        )
    )
    sim.add_external_spikes([0, 10, 40], [0, 0, 0])
    for _ in range(300):
        sim.run(1)
    sim.flush()

    snapshot = sim.dense_subnetwork_snapshot("mixed_dense_subnetwork")
    result = sim.result()
    print("dense_name", snapshot["name"])
    print("snapshot_time_step", snapshot["time_step"])
    print("snapshot_neurons", len(snapshot["membrane_v"]))
    print("snapshot_synapses", len(snapshot["synaptic_weights"]))
    print("debug_output_dir", result.path)
    print("debug_file_sizes", result.file_sizes())


if __name__ == "__main__":
    main()
