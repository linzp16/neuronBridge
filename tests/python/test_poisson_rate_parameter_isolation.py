"""Regression tests for distinct PoissonRate layer parameters."""

from __future__ import annotations

from pathlib import Path

import neuronbridge as nb
import pytest


def _assert_only_high_rate_layer_fires(sim: nb.Simulation, high_rate_id: int) -> None:
    sim.run()
    spikes = sim.output_spikes()
    assert spikes
    assert {int(spike["neuron_id"]) for spike in spikes} == {high_rate_id}


@pytest.mark.parametrize("high_rate_first", [False, True])
def test_fast_build_keeps_distinct_poisson_parameters(high_rate_first: bool) -> None:
    rates = (10_000.0, 0.0) if high_rate_first else (0.0, 10_000.0)
    network = nb.Network()
    for rate in rates:
        network.add_layer(nb.NeuronLayer.poisson_rate(1, rate_bias_hz=rate, output=True))

    sim = nb.Simulation(network, nb.SimulationConfig(steps=20, timestep=0.1)).init()
    _assert_only_high_rate_layer_fires(sim, 0 if high_rate_first else 1)


@pytest.mark.parametrize("high_rate_first", [False, True])
def test_streaming_build_keeps_distinct_poisson_parameters(
    tmp_path: Path, high_rate_first: bool
) -> None:
    rates = (10_000.0, 0.0) if high_rate_first else (0.0, 10_000.0)
    path = tmp_path / f"poisson_{int(high_rate_first)}.nbnet"
    with nb.NbnetDescriptionBuilder(
        path, options=nb.NbnetWriteOptions(overwrite=True)
    ) as builder:
        for rate in rates:
            builder.add_layer(nb.NeuronLayer.poisson_rate(1, rate_bias_hz=rate, output=True))
        builder.finalize()

    sim = nb.Simulation(
        path,
        nb.SimulationConfig(steps=20, timestep=0.1),
        build_options=nb.StreamingBuildOptions(
            memory_budget_mb=8, mmap=True, verify_checksum=True
        ),
    ).init()
    _assert_only_high_rate_layer_fires(sim, 0 if high_rate_first else 1)
