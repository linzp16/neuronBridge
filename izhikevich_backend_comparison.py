from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import neuronbridge as nb

OUT = Path(r"D:\testfile\izhikevich_backend_comparison")
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.25
STEPS = 4000

CASES = {
    "Tonic spiking": ({"a": 0.02, "b": 0.20, "c": -65.0, "d": 6.0, "V_th": 30.0}, 14.0),
    "Phasic spiking": ({"a": 0.02, "b": 0.25, "c": -65.0, "d": 6.0, "V_th": 30.0}, 0.5),
    "Tonic bursting": ({"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0, "V_th": 30.0}, 15.0),
    "Phasic bursting": ({"a": 0.02, "b": 0.25, "c": -55.0, "d": 0.05, "V_th": 30.0}, 0.8),
    "Mixed mode": ({"a": 0.02, "b": 0.20, "c": -55.0, "d": 4.0, "V_th": 30.0}, 10.0),
}


def build_network(backend: str, parameters: dict[str, float]) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_current(1))
    model = "TimeDrivenIzhikevic_Exponential_Decay_GPU" if backend == "legacy_gpu" else "TimeDrivenIzhikevic_Exponential_Decay"
    params = {key: nb.float32(value) for key, value in parameters.items()}
    if backend == "dense_gpu":
        params["dense_subnetwork_name"] = "izh_backend_compare"
    network.add_layer(nb.NeuronLayer(model, 1, parameters=params, output=backend != "dense_gpu", monitored=backend != "dense_gpu"))
    network.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    return network


def run_case(backend: str, parameters: dict[str, float], current: float) -> dict:
    sim = nb.Simulation(build_network(backend, parameters), nb.SimulationConfig(steps=STEPS, timestep=DT)).init()
    sim.add_external_currents([0], [0], [current])
    voltage: list[float] = []
    recovery: list[float] = []
    fired: list[int] = []
    for _ in range(STEPS):
        sim.run(1)
        if backend == "dense_gpu":
            snapshot = sim.dense_subnetwork_snapshot("izh_backend_compare")
            voltage.append(float(snapshot["membrane_v"][0]))
            recovery.append(float(snapshot["model_debug_states"][0]["float_state_vectors"]["u"][0]))
            fired.append(int(snapshot["fired"][0]))
        else:
            state = sim.neuron_state(1)
            voltage.append(float(state["state_variables"][0]))
            recovery.append(float(state["state_variables"][1]))
    if backend == "dense_gpu":
        spike_times = [index for index, value in enumerate(fired) if value]
    else:
        spike_times = [int(item["time"]) for item in sim.output_spikes()]
    return {
        "backend": backend,
        "parameters": parameters,
        "current": current,
        "voltage": voltage,
        "recovery": recovery,
        "spike_times": spike_times,
        "spike_count": len(spike_times),
        "backend_info": nb.backend_info(),
    }


traces: dict[str, dict[str, dict]] = {}
for case_name, (parameters, current) in CASES.items():
    traces[case_name] = {
        backend: run_case(backend, parameters, current)
        for backend in ("legacy_cpu", "dense_gpu", "legacy_gpu")
    }

summary = {}
for case_name, backends in traces.items():
    cpu = backends["legacy_cpu"]
    summary[case_name] = {}
    for backend, data in backends.items():
        summary[case_name][backend] = {
            "spike_count": data["spike_count"],
            "spike_times_first_20": data["spike_times"][:20],
            "max_abs_voltage_error_vs_cpu": max(abs(a - b) for a, b in zip(data["voltage"], cpu["voltage"])),
            "max_abs_recovery_error_vs_cpu": max(abs(a - b) for a, b in zip(data["recovery"], cpu["recovery"])),
        }

result = {"timestep": DT, "steps": STEPS, "cases": summary, "traces": traces}
(OUT / "izhikevich_backend_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

time = [index * DT for index in range(STEPS)]
colors = {"legacy_cpu": "#1769aa", "dense_gpu": "#d55e00", "legacy_gpu": "#009e73"}
labels = {"legacy_cpu": "ordinary CPU", "dense_gpu": "Dense GPU", "legacy_gpu": "legacy GPU"}
fig, axes = plt.subplots(len(CASES), 2, figsize=(14, 13), sharex=True)
for row, (case_name, backends) in enumerate(traces.items()):
    ax_v, ax_u = axes[row]
    for backend, data in backends.items():
        ax_v.plot(time, data["voltage"], linewidth=0.75, color=colors[backend], label=labels[backend])
        ax_u.plot(time, data["recovery"], linewidth=0.75, color=colors[backend], label=labels[backend])
    ax_v.set_ylabel(f"{case_name}\nV (mV)")
    ax_u.set_ylabel("u")
    ax_v.set_ylim(-90, 35)
    ax_v.grid(alpha=0.25)
    ax_u.grid(alpha=0.25)
axes[0, 0].legend(loc="upper right", fontsize=8)
axes[-1, 0].set_xlabel("time (ms)")
axes[-1, 1].set_xlabel("time (ms)")
fig.suptitle("Izhikevich backend comparison: ordinary CPU vs Dense GPU vs legacy GPU", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT / "izhikevich_backend_curves.png", dpi=220)
fig.savefig(OUT / "izhikevich_backend_curves.svg")
plt.close(fig)

print(json.dumps(summary, indent=2))
