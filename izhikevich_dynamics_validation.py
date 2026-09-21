from __future__ import annotations

import json
from pathlib import Path

import neuronbridge as nb

OUT = Path(r"D:\testfile\izhikevich_dynamics")
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.25
STEPS = 4000
GAP_STEPS = 50

CASES = {
    "tonic_spiking": ({"a": 0.02, "b": 0.20, "c": -65.0, "d": 6.0}, 14.0),
    "phasic_spiking": ({"a": 0.02, "b": 0.25, "c": -65.0, "d": 6.0}, 0.5),
    "tonic_bursting": ({"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0}, 15.0),
    # This current level gives a clear transient burst train in this wheel.
    "phasic_bursting": ({"a": 0.02, "b": 0.25, "c": -55.0, "d": 0.05}, 0.8),
    "mixed_mode": ({"a": 0.02, "b": 0.20, "c": -55.0, "d": 4.0}, 10.0),
}


def run_case(parameters: dict[str, float], current: float) -> dict:
    n = nb.Network()
    n.add_layer(nb.NeuronLayer.input_current(1))
    n.add_layer(
        nb.NeuronLayer(
            "TimeDrivenIzhikevic_Exponential_Decay",
            1,
            parameters={k: nb.float32(v) for k, v in parameters.items()},
            output=True,
            monitored=True,
        )
    )
    n.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(n, nb.SimulationConfig(steps=STEPS, timestep=DT)).init()
    sim.add_external_currents([0], [0], [current]).run()
    spikes = [int(item["time"]) for item in sim.output_spikes()]
    isi = [b - a for a, b in zip(spikes, spikes[1:])]
    groups: list[list[int]] = []
    for spike in spikes:
        if not groups or spike - groups[-1][-1] > GAP_STEPS:
            groups.append([spike])
        else:
            groups[-1].append(spike)
    steady_isi = isi[len(isi) // 2 :] if len(isi) >= 4 else isi
    return {
        "parameters": parameters,
        "current": current,
        "timestep": DT,
        "steps": STEPS,
        "spike_count": len(spikes),
        "spike_times": spikes,
        "isi_first_20": isi[:20],
        "isi_min": min(isi) if isi else None,
        "isi_max": max(isi) if isi else None,
        "steady_isi_mean": sum(steady_isi) / len(steady_isi) if steady_isi else None,
        "burst_group_sizes": [len(group) for group in groups],
        "burst_group_starts": [group[0] for group in groups],
    }


results = {name: run_case(params, current) for name, (params, current) in CASES.items()}
report = {
    "model": "TimeDrivenIzhikevic_Exponential_Decay",
    "description": "Canonical Izhikevich parameter-regime validation",
    "burst_gap_threshold_steps": GAP_STEPS,
    "cases": results,
}
(OUT / "izhikevich_dynamics_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
