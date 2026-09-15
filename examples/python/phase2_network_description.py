"""Build phase-2 neuronbridge network descriptions without running Simulation.

This example mirrors the shape of current C++ smoke/example construction:

- ordinary neuron layers and connections
- an OuterDynamicSpikeCounter sink with torque-style input connections
- an InputConvV1 visual stimulus routed to a dense subnetwork
- a PlanarArm2DOFPinocchio description with double-precision parameters

Phase 3 will wire the same Network and SimulationConfig objects to the native
Simulation lifecycle.
"""

from __future__ import annotations

import pprint

import neuronbridge as nb


def build_description() -> nb.Network:
    network = nb.Network()

    network.add_layer(nb.NeuronLayer.input_spike(2))
    network.add_layer(nb.NeuronLayer.lif_double(4, dense_name="dense_demo", dense_steps_to_keep=64))
    network.connect(nb.Connection(source=0, target=[2, 3], weight=[1.0, 1.5], max_weight=2.0, delay=1))

    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(name="counter_sink", slot_count=3, type_count=2))
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=[0, 1, 0],
            target_outer_dynamic=0,
            target_joint=[1, 1, 2],
            synapse_type=[0, 1, 0],
            weight=[2.0, 3.5, 4.0],
            delay=1,
        )
    )

    network.add_outer_dynamic(
        nb.OuterDynamic.planar_arm_2dof(
            name="arm",
            link_lengths=[0.40, 0.30],
            link_masses=[1.20, 0.80],
            joint_damping=[0.05, 0.04],
            pd_kp=[14.0, 11.0],
            pd_kd=[2.5, 2.0],
            trajectory_frequency_hz=0.35,
            dcn_torque_gain=0.0015,
            spike_retention_steps=512,
            state_feedback_single=nb.FeedbackSingleEncoding(
                bins=[7, 7, 7, 7],
                neuron_indices_by_joint_variable=[[[10], [11], [12], [13]], [[14], [15], [16], [17]]],
            ),
        )
    )

    network.add_input_conv(
        nb.InputConv.v1_bar(
            width=8,
            height=8,
            channels=1,
            speed=1.5,
            bar_width=2,
            motion_period_steps=16,
            output_target="dense_subnetwork",
            target_dense_subnetwork_name="dense_demo",
            output_source_indices=[0, 1, 2, 3],
            output_target_neuron_ids=[0, 1, 2, 3],
            output_scales=[1.0, 1.0, 1.0, 1.0],
        )
    )

    return network


def main() -> None:
    network = build_description()
    config = nb.SimulationConfig(steps=8, timestep=1.0, queues=1)

    print("backend_info:")
    pprint.pp(nb.backend_info())
    print("\npython description:")
    pprint.pp(network.to_dict())
    print("\nsimulation config:")
    pprint.pp(config.to_dict())

    if nb.backend_info()["native_extension_loaded"]:
        native = network.to_native()
        print("\nnative bridge summary:")
        pprint.pp(
            {
                "layers": native.layer_count,
                "connections": native.connection_block_count,
                "outer_dynamics": native.outer_dynamic_count,
                "outer_dynamic_connections": native.outer_dynamic_connection_block_count,
                "input_convs": native.input_conv_count,
                "neurons": native.neuron_count,
                "config": config.to_native().to_dict(),
            }
        )


if __name__ == "__main__":
    main()
