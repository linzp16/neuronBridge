"""Python migration of the dense_mixed_current baseline."""

from __future__ import annotations

import neuronbridge as nb

from _dense_baseline_builders import build_dense_mixed_current_network


def main() -> None:
    sim = nb.Simulation(build_dense_mixed_current_network(), nb.SimulationConfig(steps=64, timestep=1.0))
    sim.init()
    sim.add_external_spikes([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0])
    sim.add_external_currents([0], [1], [22.0])
    sim.run(40)

    output_spikes = sim.output_spikes()
    snapshot = sim.dense_subnetwork_snapshot("mixed_current_dense_demo")
    print("buffered_spike_count", len(output_spikes))
    for index, spike in enumerate(output_spikes):
        print(f"output_spike[{index}] cell={spike['neuron_id']} t={spike['time']}")
    print("dense_name", snapshot["name"])
    print("snapshot_time_step", snapshot["time_step"])
    print("snapshot_neurons", len(snapshot["membrane_v"]))
    print("snapshot_synapses", len(snapshot["synaptic_weights"]))
    print("visible_output_firing_count", len(snapshot["visible_output_firing_ids"]))
    print("gexc_nonzero", sum(1 for value in snapshot["gexc"] if value > 0.0))


if __name__ == "__main__":
    main()
