from __future__ import annotations

import json
from pathlib import Path

import neuronbridge as nb


OUT = Path(r"D:\testfile\other_neuron_backend_comparison\legacy_gpu_lif_bugfix_validation.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def run(model: str, backend: str, params: dict, current: float) -> dict:
    net = nb.Network()
    net.add_layer(nb.NeuronLayer.input_current(1))
    native_model = model + ("_GPU" if backend == "legacy_gpu" else "")
    net.add_layer(nb.NeuronLayer(native_model, 1, parameters={k: nb.float32(v) for k, v in params.items()}, output=True, monitored=True))
    net.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(net, nb.SimulationConfig(steps=120, timestep=0.25)).init()
    initial = sim.neuron_state(1)
    sim.add_external_currents([0], [0], [current])
    trace = []
    for _ in range(120):
        sim.run(1)
        trace.append(sim.neuron_state(1))
    voltage = [float(s["state_variables"][0]) for s in trace]
    return {
        "model": model,
        "backend": backend,
        "initial": initial,
        "final": trace[-1],
        "initial_voltage": float(initial["state_variables"][0]),
        "final_voltage": voltage[-1],
        "voltage_min": min(voltage),
        "voltage_max": max(voltage),
        "voltage_changed": max(voltage) - min(voltage) > 1e-7,
        "spike_count": len(sim.output_spikes()),
    }


cases = [
    ("TimeDrivenLIF_Exponential_Decay", {"V_rest": -65.0, "V_reset": -65.0, "V_th": -20.0, "tau": 20.0, "R": 1.0}, 5.0),
    ("TimeDrivenLIF_Voltage_jump", {"V_rest": 0.0, "V_reset": -5.0, "V_th": 20.0, "tau": 10.0, "R": 1.0}, 5.0),
]
results = [run(model, backend, params, current) for model, params, current in cases for backend in ("legacy_cpu", "legacy_gpu")]
payload = {"results": results, "checks": {
    "legacy_gpu_state_changes_for_current": all(r["voltage_changed"] for r in results if r["backend"] == "legacy_gpu"),
    "voltage_jump_gpu_initial_equals_cpu": next(r["initial_voltage"] for r in results if r["model"] == "TimeDrivenLIF_Voltage_jump" and r["backend"] == "legacy_gpu") == next(r["initial_voltage"] for r in results if r["model"] == "TimeDrivenLIF_Voltage_jump" and r["backend"] == "legacy_cpu"),
}}
OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
