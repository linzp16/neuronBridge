"""Run the six-axis Pinocchio-backed ROKAE xMate OuterDynamic."""

from __future__ import annotations

import argparse

import neuronbridge as nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", help="Path to xMateSR3C.urdf")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_outer_dynamic(
        nb.OuterDynamic.rokae_arm(
            urdf_path=args.urdf,
            name="rokae",
            end_effector_frame="xMateSR3C_link6",
            trajectory_mode="smooth_step",
            motion_duration_s=1.0,
            update_timestep=1,
            communication_interval=1,
        )
    )

    simulation = nb.Simulation(
        network,
        nb.SimulationConfig(steps=args.steps, timestep=1.0),
    )
    simulation.init().run()
    state = simulation.outer_dynamic_state()
    print("q:     ", state["q"])
    print("q_des: ", state["q_des"])
    print("tau:   ", state["tau_total"])


if __name__ == "__main__":
    main()
