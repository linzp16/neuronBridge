"""Python migration of the InputConvV1 -> dense PoissonRate baseline."""

from __future__ import annotations

from pathlib import Path

import neuronbridge as nb
from _example_paths import artifact_dir


def build_network(output_count: int = 512) -> nb.Network:
    dense_name = "inputconv_poisson_dense_demo"
    network = nb.Network()
    network.add_layer(
        nb.NeuronLayer.poisson_rate(
            output_count,
            dense_name=dense_name,
            rate_bias_hz=0.0,
            rate_gain_hz_per_current=1.0,
            dense_steps_to_keep=128,
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
            target_dense_subnetwork_name=dense_name,
            output_source_indices=list(range(output_count)),
            output_target_neuron_ids=list(range(output_count)),
            output_pending_channel=2,
        )
    )
    return network


def main() -> None:
    output_dir = artifact_dir("inputconv_poisson_dense")
    sim = nb.Simulation(build_network(), nb.SimulationConfig(steps=40, timestep=1.0))
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=output_dir,
            all_neurons=True,
            record_state=True,
            record_spikes=True,
            record_pending_channels=True,
            monitor_all_inputconv=True,
            record_inputconv_outputs=True,
            record_inputconv_inputs=True,
        )
    )
    sim.run(32)
    sim.flush()
    snapshot = sim.dense_subnetwork_snapshot("inputconv_poisson_dense_demo")
    result = sim.result()
    print("inputconv_count", sim.input_conv_count)
    print("inputconv_output_count", sim.input_conv_output_count(0))
    print("snapshot_neurons", len(snapshot["membrane_v"]))
    print("debug_output_dir", result.path)
    print("debug_file_sizes", result.file_sizes())


if __name__ == "__main__":
    main()
