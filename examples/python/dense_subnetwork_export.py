"""Python migration of the dense subnetwork export baseline."""

from __future__ import annotations

import neuronbridge as nb


def all_to_all(source_begin, source_count, target_begin, target_count, *, weight, delay=1, synapse_type=0):
    return nb.Connection(
        source=[source_begin + source for source in range(source_count) for _ in range(target_count)],
        target=[target_begin + target for _ in range(source_count) for target in range(target_count)],
        synapse_type=[int(synapse_type)] * (source_count * target_count),
        weight=[float(weight)] * (source_count * target_count),
        max_weight=[10.0] * (source_count * target_count),
        delay=[int(delay)] * (source_count * target_count),
    )


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_double(2, random_sigma=nb.float32_array4([1.0, 0.0, 0.0, 0.0])))
    network.add_layer(
        nb.NeuronLayer.lif_double(
            4,
            dense_name="export_dense_subnetwork",
            random_sigma=nb.float32_array4([2.0, 0.0, 0.0, 0.0]),
        )
    )
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_double",
            3,
            update_timestep=2,
            parameters={
                "dense_subnetwork_name": "export_dense_subnetwork",
                "random_sigma": nb.float32_array4([3.0, 0.0, 0.0, 0.0]),
            },
        )
    )
    network.add_layer(nb.NeuronLayer.lif_decay(2, output=True, random_sigma=nb.float32_array3([0.0, 0.0, 0.0])))
    network.connect(all_to_all(0, 1, 1, 2, weight=20.0, delay=1))
    network.connect(all_to_all(1, 2, 3, 4, weight=12.0, delay=1))
    network.connect(all_to_all(3, 4, 7, 3, weight=8.0, delay=2))
    network.connect(all_to_all(7, 3, 10, 2, weight=0.60, delay=3))
    return network


def main() -> None:
    sim = nb.Simulation(build_network(), nb.SimulationConfig(steps=48, timestep=0.1))
    sim.init()
    sim.set_dense_subnetwork_full_firing_export_enabled("export_dense_subnetwork")
    sim.add_external_spikes([0, 1, 2], [0, 0, 0])
    sim.run(30)
    snapshot = sim.dense_subnetwork_snapshot("export_dense_subnetwork")
    print("subnetwork_count", sim.dense_subnetwork_count)
    print("snapshot_name", snapshot["name"])
    print("snapshot_time_step", snapshot["time_step"])
    print("snapshot_neurons", len(snapshot["membrane_v"]))
    print("snapshot_synapses", len(snapshot["synaptic_weights"]))
    print("snapshot_gexc_nonzero", sum(1 for value in snapshot["gexc"] if value > 0.0))
    sim.reset_dense_subnetwork("export_dense_subnetwork")
    reset_snapshot = sim.dense_subnetwork_snapshot(0)
    print("reset_fired_count", sum(reset_snapshot["fired"]))


if __name__ == "__main__":
    main()
