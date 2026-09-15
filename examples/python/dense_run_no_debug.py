"""Python migration of the dense_run_no_debug baseline."""

from __future__ import annotations

import neuronbridge as nb

from _dense_baseline_builders import build_dense_run_no_debug_network


def main() -> None:
    sim = nb.Simulation(build_dense_run_no_debug_network(), nb.SimulationConfig(steps=1200, timestep=1.0))
    sim.init()
    sim.add_external_spikes([0, 20, 40, 60, 80, 100], [0, 0, 0, 0, 0, 0])
    sim.run(1000)

    output_spikes = sim.output_spikes()
    readout0 = sim.neuron_state(8)
    readout1 = sim.neuron_state(9)
    print("mode=normal_run_no_debug")
    print("total_steps=1000")
    print("sample_interval_steps=none")
    print(f"buffered_output_spike_count={len(output_spikes)}")
    print(f"readout_g neuron8={readout0['state_variables'][1]} neuron9={readout1['state_variables'][1]}")
    print("stage=done")


if __name__ == "__main__":
    main()
