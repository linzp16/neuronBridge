"""Python migration of public-API dense subnetwork smoke checks."""

from __future__ import annotations

import neuronbridge as nb


def _dense_layer(name: str, model: str = "TimeDrivenLIF_Exponential_double") -> nb.NeuronLayer:
    return nb.NeuronLayer(model, 1, parameters={"dense_subnetwork_name": name})


def expect_error(label: str, build_network, text: str) -> None:
    try:
        nb.Simulation(build_network(), nb.SimulationConfig(steps=4, timestep=1.0))
    except RuntimeError as exc:
        if text not in str(exc):
            raise
        print(f"{label}=ok")
        return
    raise RuntimeError(f"{label} did not fail as expected")


def run_dense_layer_ownership_smoke() -> None:
    def dense_input_spike() -> nb.Network:
        network = nb.Network()
        network.add_layer(nb.NeuronLayer.input_spike(1, dense_subnetwork_name="bad_dense"))
        return network

    def dense_input_current() -> nb.Network:
        network = nb.Network()
        network.add_layer(nb.NeuronLayer.input_current(1, dense_subnetwork_name="bad_dense"))
        return network

    def dense_output_layer() -> nb.Network:
        network = nb.Network()
        network.add_layer(
            nb.NeuronLayer(
                "TimeDrivenLIF_Exponential_double",
                1,
                output=True,
                parameters={"dense_subnetwork_name": "bad_output_dense"},
            )
        )
        return network

    expect_error("input_spike_dense_reject_smoke", dense_input_spike, "cannot be placed inside dense subnetwork")
    expect_error("input_current_dense_reject_smoke", dense_input_current, "cannot be placed inside dense subnetwork")
    expect_error("dense_output_reject_smoke", dense_output_layer, "cannot be marked isOutput=true")

    main_relay = nb.Network()
    main_relay.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 1))
    nb.Simulation(main_relay, nb.SimulationConfig(steps=4, timestep=1.0)).init()
    print("main_trigger_relay_build_smoke=ok")

    dense_relay = nb.Network()
    dense_relay.add_layer(_dense_layer("relay_dense", "TriggerRelayNeuronModel"))
    sim = nb.Simulation(dense_relay, nb.SimulationConfig(steps=4, timestep=1.0))
    sim.init()
    print(f"dense_trigger_relay_build_smoke=ok count={sim.dense_subnetwork_count}")


def run_dense_to_dense_build_smoke() -> None:
    network = nb.Network()
    network.add_layer(_dense_layer("dense_a"))
    network.add_layer(_dense_layer("dense_b"))
    network.connect(nb.Connection(0, 1, weight=7.0, max_weight=7.0, delay=2))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=4, timestep=1.0))
    sim.init()
    print(f"dense_to_dense_build_smoke=ok count={sim.dense_subnetwork_count}")
    print(f"dense_name[0]={sim.dense_subnetwork_name(0)}")
    print(f"dense_name[1]={sim.dense_subnetwork_name(1)}")


def main() -> None:
    run_dense_layer_ownership_smoke()
    run_dense_to_dense_build_smoke()
    print("stage=done")


if __name__ == "__main__":
    main()
