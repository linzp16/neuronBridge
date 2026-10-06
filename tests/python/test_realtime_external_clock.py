from __future__ import annotations

import math

import neuronbridge as nb


def _minimal_simulation(*, steps: int = 12) -> nb.Simulation:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1, output=True))
    network.connect(nb.Connection(source=0, target=1, weight=20.0, delay=1))
    return nb.Simulation(
        network,
        nb.SimulationConfig(steps=steps, timestep=1.0, queues=1),
    )


def test_realtime_internal_clock_smoke() -> None:
    sim = _minimal_simulation()
    sim.enable_realtime(slot_steps=2, max_advance_seconds=0.004)
    sim.init()
    sim.add_external_spikes([0, 4, 8], [0, 0, 0])
    sim.run_realtime(12)

    counters = sim.realtime_restriction_counts()
    assert all(isinstance(value, int) and value >= 0 for value in counters.values())
    sim.disable_realtime()


def test_realtime_external_clock_smoke() -> None:
    sim = _minimal_simulation()
    sim.enable_realtime(
        slot_steps=2,
        max_advance_seconds=0.004,
        use_external_clock=True,
    )
    sim.set_realtime_external_clock(0.0)
    sim.init()
    sim.add_external_spikes([0, 4, 8], [0, 0, 0])
    sim.run_realtime(12)

    counters = sim.realtime_restriction_counts()
    assert all(isinstance(value, int) and value >= 0 for value in counters.values())
    sim.set_realtime_external_clock(0.012)
    sim.disable_realtime()


def test_external_clock_rejects_nonfinite_values() -> None:
    sim = _minimal_simulation(steps=1)
    for value in (math.nan, math.inf, -math.inf):
        try:
            sim.set_realtime_external_clock(value)
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"non-finite external clock accepted: {value!r}")
