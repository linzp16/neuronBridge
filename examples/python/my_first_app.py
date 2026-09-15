"""Python migration of the C++ my_first_app example.

This is the smallest end-user example: build a four-neuron input layer,
connect it all-to-all into four output LIF neurons, inject one spike into
each input neuron, and run the native simulation through neuronbridge.
"""

from __future__ import annotations

import neuronbridge as nb


def all_to_all(
    source_begin: int,
    source_count: int,
    target_begin: int,
    target_count: int,
    *,
    weight: float,
    delay: int = 1,
) -> nb.Connection:
    return nb.Connection(
        source=[source_begin + source for target in range(target_count) for source in range(source_count)],
        target=[target_begin + target for target in range(target_count) for source in range(source_count)],
        weight=[float(weight)] * (source_count * target_count),
        max_weight=[float(weight)] * (source_count * target_count),
        delay=[int(delay)] * (source_count * target_count),
        synapse_type=[0] * (source_count * target_count),
        synapse_rule=[-1] * (source_count * target_count),
        trigger_rule=[-1] * (source_count * target_count),
    )


def build_network() -> nb.Network:
    input_count = 4
    lif_count = 4

    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(input_count))
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_double",
            lif_count,
            output=True,
            parameters={
                "V_rest": nb.float32(-60.0),
                "V_reset": nb.float32(-60.0),
                "V_th": nb.float32(-50.0),
                "tau": nb.float32(20.0),
                "R": nb.float32(1.0),
                "t_ref": nb.int32(1),
                "gexc_tau": nb.float32(5.0),
                "ginh_tau": nb.float32(10.0),
                "Eexc": nb.float32(0.0),
                "Einh": nb.float32(-80.0),
            },
        )
    )
    network.connect(all_to_all(0, input_count, input_count, lif_count, weight=3.0))
    return network


def run_example(total_steps: int = 100, dt_ms: float = 0.1) -> nb.Simulation:
    sim = nb.Simulation(build_network(), nb.SimulationConfig(steps=total_steps, timestep=dt_ms))
    sim.init()
    sim.add_external_spikes([1, 1, 1, 1], [0, 1, 2, 3])
    sim.run(total_steps)
    return sim


def main() -> None:
    input_count = 4
    lif_count = 4
    total_steps = 100
    dt_ms = 0.1

    sim = run_example(total_steps=total_steps, dt_ms=dt_ms)

    print("mode=my_first_app")
    print(f"steps={total_steps}")
    print(f"dt_ms={dt_ms}")
    print(f"input_count={input_count}")
    print(f"lif_count={lif_count}")
    print(f"buffered_output_spike_count={len(sim.output_spikes())}")
    print("simulation finished")


if __name__ == "__main__":
    main()
