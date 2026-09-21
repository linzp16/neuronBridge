from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import neuronbridge as nb

OUT = Path(r"D:\testfile\izhikevich_dynamics")
OUT.mkdir(parents=True, exist_ok=True)
DT = 0.25
STEPS = 4000

CASES = {
    "Tonic spiking": ({"a": 0.02, "b": 0.20, "c": -65.0, "d": 6.0}, 14.0),
    "Phasic spiking": ({"a": 0.02, "b": 0.25, "c": -65.0, "d": 6.0}, 0.5),
    "Tonic bursting": ({"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0}, 15.0),
    "Phasic bursting": ({"a": 0.02, "b": 0.25, "c": -55.0, "d": 0.05}, 0.8),
    "Mixed mode": ({"a": 0.02, "b": 0.20, "c": -55.0, "d": 4.0}, 10.0),
}


def simulate(parameters: dict[str, float], current: float):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_current(1))
    network.add_layer(
        nb.NeuronLayer(
            "TimeDrivenIzhikevic_Exponential_Decay",
            1,
            parameters={key: nb.float32(value) for key, value in parameters.items()},
            output=True,
            monitored=True,
        )
    )
    network.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=STEPS, timestep=DT)).init()
    sim.add_external_currents([0], [0], [current])
    voltage = []
    recovery = []
    for _ in range(STEPS):
        sim.run(1)
        state = sim.neuron_state(1)["state_variables"]
        voltage.append(float(state[0]))
        recovery.append(float(state[1]))
    spikes = [int(item["time"]) for item in sim.output_spikes()]
    return voltage, recovery, spikes


traces = {}
for name, (parameters, current) in CASES.items():
    voltage, recovery, spikes = simulate(parameters, current)
    traces[name] = {
        "parameters": parameters,
        "current": current,
        "voltage": voltage,
        "recovery": recovery,
        "spikes": spikes,
    }

(OUT / "izhikevich_voltage_traces.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")

time = [index * DT for index in range(STEPS)]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True)
for row, (name, data) in enumerate(traces.items()):
    ax_v, ax_u = axes[row]
    ax_v.plot(time, data["voltage"], color="#1769aa", linewidth=0.8)
    ax_u.plot(time, data["recovery"], color="#c43d3d", linewidth=0.8)
    for spike in data["spikes"]:
        if 0 <= spike < STEPS:
            ax_v.axvline(spike * DT, color="#e69f00", alpha=0.18, linewidth=0.5)
    ax_v.set_ylabel(f"{name}\nV (mV)")
    ax_u.set_ylabel("u")
    ax_v.set_title(f"I={data['current']} | (a,b,c,d)=({data['parameters']['a']}, {data['parameters']['b']}, {data['parameters']['c']}, {data['parameters']['d']})", fontsize=8)
    ax_v.set_ylim(-90, 35)
    ax_u.set_ylim(min(data["recovery"]) - 2, max(data["recovery"]) + 2)
axes[-1, 0].set_xlabel("time (ms)")
axes[-1, 1].set_xlabel("time (ms)")
fig.suptitle("Izhikevich neuron dynamics in neuronbridge wheel", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT / "izhikevich_dynamics_curves.png", dpi=220)
fig.savefig(OUT / "izhikevich_dynamics_curves.svg")
plt.close(fig)

# A zoomed voltage-only figure makes the burst structure easier to inspect.
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
for ax, (name, data) in zip(axes, traces.items()):
    ax.plot(time, data["voltage"], color="#1769aa", linewidth=0.8)
    ax.set_ylabel(f"{name}\nV (mV)")
    ax.set_ylim(-90, 35)
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("time (ms)")
fig.suptitle("Voltage traces of distinct Izhikevich firing regimes", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT / "izhikevich_voltage_curves.png", dpi=220)
fig.savefig(OUT / "izhikevich_voltage_curves.svg")
plt.close(fig)

print(f"wrote plots and traces to {OUT}")
