"""Numerical alignment checks for legacy CPU/GPU triple LIF models."""

from __future__ import annotations

import neuronbridge as nb
import pytest


CPU_MODEL = "TimeDrivenLIF_Exponential_triple"
GPU_MODEL = "TimeDrivenLIF_Exponential_triple_GPU"


def _run(model: str, synapse_type: int | None) -> tuple[list[int], list[float]]:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.input_current(1))
    network.add_layer(
        nb.NeuronLayer(
            model,
            1,
            output=True,
            monitored=True,
            parameters={
                "V_rest": nb.float32(-60.0),
                "V_th": nb.float32(-50.0),
                "V_reset": nb.float32(-60.0),
                "tau": nb.float32(10.0),
                "R": nb.float32(1.0),
                "t_ref": nb.int32(5),
                "E_ampa": nb.float32(0.0),
                "ampa_tau": nb.float32(5.0),
                "E_gaba": nb.float32(-80.0),
                "gaba_tau": nb.float32(8.0),
                "nmda_tau": nb.float32(30.0),
                "random_mu": nb.float32_array5([0.0] * 5),
                "random_sigma": nb.float32_array5([0.0] * 5),
            },
        )
    )
    network.connect(nb.Connection(1, 2, synapse_type=3, weight=1.0, delay=1))
    if synapse_type is not None:
        weights = {0: 0.25, 1: 0.35, 2: 0.45}
        network.connect(
            nb.Connection(
                0,
                2,
                synapse_type=synapse_type,
                weight=weights[synapse_type],
                delay=1,
            )
        )

    sim = nb.Simulation(network, nb.SimulationConfig(steps=120, timestep=1.0)).init()
    sim.add_external_currents([0], [1], [10.5])
    if synapse_type is not None:
        sim.add_external_spikes([5, 15, 25, 35, 45, 55], [0] * 6)
    sim.run()
    spikes = [int(item["time"]) for item in sim.output_spikes()]
    state = [float(value) for value in sim.neuron_state(2)["state_variables"]]
    return spikes, state


def test_triple_cpu_gpu_default_parameters_align() -> None:
    def constant_current(model: str) -> tuple[list[int], list[float]]:
        network = nb.Network()
        network.add_layer(nb.NeuronLayer.input_current(1))
        network.add_layer(nb.NeuronLayer(model, 1, output=True, monitored=True))
        network.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
        sim = nb.Simulation(network, nb.SimulationConfig(steps=80, timestep=1.0)).init()
        sim.add_external_currents([0], [0], [12.0]).run()
        return (
            [int(item["time"]) for item in sim.output_spikes()],
            [float(value) for value in sim.neuron_state(1)["state_variables"]],
        )

    cpu_spikes, cpu_state = constant_current(CPU_MODEL)
    gpu_spikes, gpu_state = constant_current(GPU_MODEL)
    assert cpu_spikes == gpu_spikes == [18]
    assert cpu_state == gpu_state


def test_triple_cpu_gpu_all_synapse_channels_align() -> None:
    results: dict[int | None, tuple[list[int], list[float]]] = {}
    for synapse_type in (None, 0, 1, 2):
        cpu_spikes, cpu_state = _run(CPU_MODEL, synapse_type)
        gpu_spikes, gpu_state = _run(GPU_MODEL, synapse_type)
        assert cpu_spikes == gpu_spikes
        assert cpu_state == pytest.approx(gpu_state, abs=1.0e-5)
        results[synapse_type] = (cpu_spikes, cpu_state)

    baseline_spikes = results[None][0]
    assert results[0][0] != baseline_spikes, "AMPA input did not affect output spikes"
    assert results[1][0] != baseline_spikes, "GABA input did not affect output spikes"
    assert results[2][0] != baseline_spikes, "NMDA input did not affect output spikes"
