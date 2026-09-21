from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import neuronbridge as nb

OUT = Path(r"D:\testfile\other_neuron_backend_comparison")
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.25
STEPS = 4000
BACKENDS = ("legacy_cpu", "dense_gpu", "legacy_gpu")
COLORS = {"legacy_cpu": "#1769aa", "dense_gpu": "#d55e00", "legacy_gpu": "#009e73"}
LABELS = {"legacy_cpu": "ordinary CPU", "dense_gpu": "Dense GPU", "legacy_gpu": "legacy GPU"}

CASES = {
    "LIF exponential decay": {
        "model": "TimeDrivenLIF_Exponential_Decay",
        "parameters": {"V_rest": -65.0, "V_reset": -65.0, "V_th": -20.0, "tau": 20.0, "R": 1.0},
        "current": 5.0,
    },
    "LIF exponential double": {
        "model": "TimeDrivenLIF_Exponential_double",
        "parameters": {"V_rest": -65.0, "V_reset": -65.0, "V_th": -20.0, "tau": 20.0, "R": 1.0},
        "current": 5.0,
    },
    "LIF voltage jump": {
        "model": "TimeDrivenLIF_Voltage_jump",
        "parameters": {"V_rest": 0.0, "V_reset": -5.0, "V_th": 20.0, "tau": 10.0, "R": 1.0},
        "current": 5.0,
    },
    "Poisson rate": {
        "model": "PoissonRate",
        "parameters": {"rate_bias_hz": 40.0, "rate_gain_hz_per_current": 0.0},
        "current": 0.0,
    },
}


def make_network(case: dict, backend: str) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_current(1))
    model = case["model"]
    if backend == "legacy_gpu":
        model += "_GPU"
    parameters = {key: nb.float32(value) for key, value in case["parameters"].items()}
    if backend == "dense_gpu" and case["model"] == "PoissonRate":
        parameters["poisson_rate_bias_hz"] = parameters.pop("rate_bias_hz")
        parameters["poisson_rate_gain_hz_per_current"] = parameters.pop("rate_gain_hz_per_current")
    if backend == "dense_gpu":
        parameters["dense_subnetwork_name"] = "other_model_backend_compare"
    network.add_layer(nb.NeuronLayer(model, 1, parameters=parameters, output=backend != "dense_gpu", monitored=backend != "dense_gpu"))
    network.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    return network


def run_case(case: dict, backend: str) -> dict:
    sim = nb.Simulation(make_network(case, backend), nb.SimulationConfig(steps=STEPS, timestep=DT)).init()
    sim.add_external_currents([0], [0], [case["current"]])
    voltage: list[float] = []
    fired: list[int] = []
    for _ in range(STEPS):
        sim.run(1)
        if backend == "dense_gpu":
            snapshot = sim.dense_subnetwork_snapshot("other_model_backend_compare")
            voltage.append(float(snapshot["membrane_v"][0]))
            fired.append(int(snapshot["fired"][0]))
        else:
            state = sim.neuron_state(1)
            voltage.append(float(state["state_variables"][0]))
    if backend == "dense_gpu":
        spike_times = [i for i, value in enumerate(fired) if value]
    else:
        spike_times = [int(item["time"]) for item in sim.output_spikes()]
    return {"backend": backend, "voltage": voltage, "spike_times": spike_times, "spike_count": len(spike_times)}


results = {}
for name, case in CASES.items():
    results[name] = {backend: run_case(case, backend) for backend in BACKENDS}

summary = {}
for name, backends in results.items():
    cpu = backends["legacy_cpu"]
    summary[name] = {}
    for backend, data in backends.items():
        summary[name][backend] = {
            "spike_count": data["spike_count"],
            "spike_times_first_20": data["spike_times"][:20],
            "max_abs_voltage_error_vs_cpu": max(abs(a - b) for a, b in zip(data["voltage"], cpu["voltage"])),
        }

payload = {"timestep": DT, "steps": STEPS, "cases": CASES, "summary": summary, "traces": results}
(OUT / "other_neuron_backend_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

time = [i * DT for i in range(STEPS)]
analog_names = [name for name in CASES if name != "Poisson rate"]
fig, axes = plt.subplots(len(analog_names), 1, figsize=(14, 9), sharex=True)
for ax, name in zip(axes, analog_names):
    for backend, data in results[name].items():
        ax.plot(time, data["voltage"], color=COLORS[backend], linewidth=0.8, label=LABELS[backend])
    ax.set_ylabel(f"{name}\nV")
    ax.set_ylim(-90, 35)
    ax.grid(alpha=0.25)
axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time (ms)")
fig.suptitle("Other neuron models: ordinary CPU vs Dense GPU vs legacy GPU", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT / "other_neuron_voltage_curves.png", dpi=220)
fig.savefig(OUT / "other_neuron_voltage_curves.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 4))
for row, backend in enumerate(BACKENDS):
    spikes = results["Poisson rate"][backend]["spike_times"]
    ax.scatter([t * DT for t in spikes], [row] * len(spikes), s=12, color=COLORS[backend], label=LABELS[backend])
ax.set_yticks(range(len(BACKENDS)), [LABELS[b] for b in BACKENDS])
ax.set_xlabel("time (ms)")
ax.set_title("PoissonRate spike raster")
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "poisson_backend_raster.png", dpi=220)
fig.savefig(OUT / "poisson_backend_raster.svg")
plt.close(fig)

print(json.dumps(summary, indent=2))
