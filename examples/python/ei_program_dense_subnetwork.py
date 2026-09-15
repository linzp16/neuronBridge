"""Python migration of the dense EI benchmark baseline."""

from __future__ import annotations

import argparse

import neuronbridge as nb

from _ei_baseline_builders import (
    BASE_TIMESTEP_MS,
    DELAY_STEPS,
    EXCITATORY_COUNT,
    EXTERNAL_CURRENT,
    INHIBITORY_COUNT,
    NEURON_COUNT,
    SIMULATION_STEPS,
    build_dense_ei_program,
    default_connectivity_dir,
    load_connectivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectivity-dir", default=str(default_connectivity_dir()))
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connectivity = load_connectivity(args.connectivity_dir)
    sim = nb.Simulation(
        build_dense_ei_program(connectivity),
        nb.SimulationConfig(steps=args.steps, timestep=BASE_TIMESTEP_MS),
    )
    sim.init()
    sim.set_dense_subnetwork_full_firing_export_enabled("ei_program_dense_subnetwork")
    sim.add_external_currents([0], [NEURON_COUNT], [EXTERNAL_CURRENT])

    total_e_spikes = 0
    total_i_spikes = 0
    for _step in range(args.steps):
        sim.run(1)
        snapshot = sim.dense_subnetwork_snapshot("ei_program_dense_subnetwork")
        for neuron_id in snapshot["full_firing_ids"]:
            if neuron_id < EXCITATORY_COUNT:
                total_e_spikes += 1
            elif neuron_id < EXCITATORY_COUNT + INHIBITORY_COUNT:
                total_i_spikes += 1

    print("mode=ei_dense_subnetwork")
    print("monitor_enabled=0")
    print(f"steps={args.steps}")
    print(f"delay={DELAY_STEPS}")
    print(f"dense_subnetwork_count={sim.dense_subnetwork_count}")
    print(f"total_e_spikes={total_e_spikes}")
    print(f"total_i_spikes={total_i_spikes}")


if __name__ == "__main__":
    main()
