from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import neuronbridge as nb

OUT = Path(r"D:\testfile\firing_activity_curves")
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.25
STEPS = 800
BACKENDS = ("legacy_cpu", "dense_gpu", "legacy_gpu")
COLORS = {"legacy_cpu": "#0072B2", "dense_gpu": "#D55E00", "legacy_gpu": "#009E73"}
LABELS = {"legacy_cpu": "ordinary CPU", "dense_gpu": "Dense GPU", "legacy_gpu": "legacy GPU"}

CASES = {
    "LIF exponential decay": ("TimeDrivenLIF_Exponential_Decay", {"V_rest": -65.0, "V_reset": -65.0, "V_th": -20.0, "tau": 20.0, "R": 1.0}, 50.0),
    "LIF exponential double": ("TimeDrivenLIF_Exponential_double", {"V_rest": -65.0, "V_reset": -65.0, "V_th": -20.0, "tau": 20.0, "R": 1.0}, 50.0),
    "LIF voltage jump": ("TimeDrivenLIF_Voltage_jump", {"V_rest": 0.0, "V_reset": -5.0, "V_th": 20.0, "tau": 10.0, "R": 1.0}, 30.0),
    "Izhikevich tonic spiking": ("TimeDrivenIzhikevic_Exponential_Decay", {"a": 0.02, "b": 0.20, "c": -65.0, "d": 6.0, "V_th": 30.0}, 14.0),
}


def run_case(name: str, spec: tuple[str, dict[str, float], float], backend: str) -> dict:
    model, raw_params, current = spec
    net = nb.Network()
    net.add_layer(nb.NeuronLayer.input_current(1))
    model_name = model + ("_GPU" if backend == "legacy_gpu" else "")
    params = {k: nb.float32(v) for k, v in raw_params.items()}
    dense_name = "firing_activity_" + name.replace(" ", "_")
    if backend == "dense_gpu":
        params["dense_subnetwork_name"] = dense_name
    net.add_layer(nb.NeuronLayer(model_name, 1, parameters=params, output=backend != "dense_gpu", monitored=backend != "dense_gpu"))
    net.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(net, nb.SimulationConfig(steps=STEPS, timestep=DT)).init()
    sim.add_external_currents([0], [0], [current])
    voltage = []
    fired = []
    for _ in range(STEPS):
        sim.run(1)
        if backend == "dense_gpu":
            snap = sim.dense_subnetwork_snapshot(dense_name)
            voltage.append(float(snap["membrane_v"][0]))
            fired.append(int(snap["fired"][0]))
        else:
            voltage.append(float(sim.neuron_state(1)["state_variables"][0]))
    if backend == "dense_gpu":
        spike_steps = [i for i, x in enumerate(fired) if x]
    else:
        spike_steps = [int(x["time"]) for x in sim.output_spikes()]
    return {"backend": backend, "current": current, "voltage": voltage, "spike_steps": spike_steps, "spike_count": len(spike_steps)}


results = {name: {b: run_case(name, spec, b) for b in BACKENDS} for name, spec in CASES.items()}
(OUT / "firing_activity_curves.json").write_text(json.dumps({"timestep_ms": DT, "steps": STEPS, "cases": results}, indent=2), encoding="utf-8")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
time = [i * DT for i in range(STEPS)]
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
for ax, (name, _) in zip(axes, CASES.items()):
    for backend in BACKENDS:
        data = results[name][backend]
        ax.plot(time, data["voltage"], color=COLORS[backend], linewidth=0.9, label=LABELS[backend])
        spikes = [s for s in data["spike_steps"] if s < STEPS]
        if spikes:
            ax.scatter([time[s] for s in spikes], [data["voltage"][s] for s in spikes], color=COLORS[backend], s=10, zorder=3)
    ax.set_title(f"{name}  |  firing activity")
    ax.set_ylabel("V (mV)")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].legend(ncol=3, frameon=False, loc="upper right")
axes[-1].set_xlabel("Time (ms)")
fig.savefig(OUT / "all_neuron_firing_activity_curves.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "all_neuron_firing_activity_curves.svg", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
for row, backend in enumerate(BACKENDS):
    data = results["Izhikevich tonic spiking"][backend]
    spikes = [s for s in data["spike_steps"] if s < STEPS]
    ax.scatter([time[s] for s in spikes], [row] * len(spikes), s=14, color=COLORS[backend], label=LABELS[backend])
ax.set_yticks(range(3), [LABELS[b] for b in BACKENDS])
ax.set_xlabel("Time (ms)")
ax.set_title("Izhikevich tonic-spiking raster")
ax.grid(axis="x", alpha=0.25)
ax.legend(frameon=False, ncol=3, loc="upper right")
fig.savefig(OUT / "izhikevich_tonic_spiking_raster.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "izhikevich_tonic_spiking_raster.svg", bbox_inches="tight")
plt.close(fig)
print(json.dumps({name: {b: results[name][b]["spike_count"] for b in BACKENDS} for name in CASES}, indent=2))
