"""Dependency-free installed-wheel smoke for all direct streaming route classes."""

from pathlib import Path
import tempfile

import neuronbridge as nb


def main() -> None:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_double(1, dense_name="dense_a"))
    network.add_layer(nb.NeuronLayer.lif_double(1, dense_name="dense_b"))
    network.add_layer(nb.NeuronLayer.lif_double(1, output=True))
    network.connect(nb.Connection(0, 1, weight=2.0, max_weight=5.0, delay=1))
    network.connect(nb.Connection(1, 2, weight=3.0, max_weight=5.0, delay=1))
    network.connect(nb.Connection(2, 3, weight=4.0, max_weight=5.0, delay=1))
    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(name="counter", slot_count=1, type_count=1))
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=0,
            target_outer_dynamic=0,
            target_joint=0,
            synapse_type=0,
            weight=1.5,
            delay=1,
        )
    )
    network.add_input_conv(nb.InputConv.v1_bar(width=2, height=2, channels=1))

    with tempfile.TemporaryDirectory(prefix="neuronbridge-full-direct-") as directory:
        path = Path(directory) / "mixed.nbnet"
        nb.nbnet.from_network(network, path)
        sim = nb.Simulation(path, nb.SimulationConfig(steps=6, timestep=1.0)).init()
        assert sim.build_stats["runtime_build_path"] == "streaming_direct"
        assert sim.dense_subnetwork_count == 2
        assert sim.input_conv_count == 1
        assert [sim.get_connection_weight(i) for i in range(3)] == [2.0, 3.0, 4.0]
        sim.add_external_spikes([0], [0]).run(3)
        snapshot = sim.outer_dynamic_spike_counter_snapshot("counter")
        assert snapshot["spike_counts"][0] == 1
        assert abs(snapshot["weighted_sums"][0] - 1.5) <= 1e-7

    print(f"installed wheel full direct streaming smoke PASS: {Path(nb.__file__).resolve()}")


if __name__ == "__main__":
    main()
