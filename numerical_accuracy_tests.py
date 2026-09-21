from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import neuronbridge as nb

ROOT = Path(r"D:\testfile\numerical_accuracy")
ROOT.mkdir(parents=True, exist_ok=True)
DT = 1.0
STEPS = 24
CURRENT = 5.0
TOL = 2e-5
INTERNAL_SUBSTEPS_PER_RUN = 2


def one_current_network(model: str, parameters: dict | None = None):
    n = nb.Network()
    n.add_layer(nb.NeuronLayer.input_current(1))
    n.add_layer(nb.NeuronLayer(model, 1, parameters=parameters or {}, output=True, monitored=True))
    n.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    return n


def trace_model(model: str, parameters: dict | None = None, steps: int = STEPS):
    sim = nb.Simulation(one_current_network(model, parameters), nb.SimulationConfig(steps=steps, timestep=DT)).init()
    sim.add_external_currents([0], [0], [CURRENT])
    trace = []
    for _ in range(steps):
        sim.run(1)
        trace.append(sim.neuron_state(1))
    return trace


def trace_model_with_initial(model: str, parameters: dict | None = None, steps: int = STEPS):
    sim = nb.Simulation(one_current_network(model, parameters), nb.SimulationConfig(steps=steps, timestep=DT)).init()
    initial = sim.neuron_state(1)
    sim.add_external_currents([0], [0], [CURRENT])
    trace = []
    for _ in range(steps):
        sim.run(1)
        trace.append(sim.neuron_state(1))
    return initial, trace


def compare_scalar(name, actual, expected):
    errors = [abs(float(a) - float(e)) for a, e in zip(actual, expected)]
    max_error = max(errors, default=0.0)
    return {"name": name, "pass": max_error <= TOL, "max_abs_error": max_error, "tolerance": TOL, "actual": actual, "expected": expected}


def reference_lif(v_rest, tau, r, steps, initial_voltage=None):
    v = v_rest if initial_voltage is None else initial_voltage
    result = []
    for outer_step in range(steps):
        substeps = INTERNAL_SUBSTEPS_PER_RUN if outer_step == 0 else 1
        for substep in range(substeps):
            # The first substep is processed before the delay=1 current event.
            current = 0.0 if outer_step == 0 and substep == 0 else CURRENT
            v += ((r * current / tau) + (v_rest - v) / tau) * DT
        result.append(v)
    return result


def test_lif_decay():
    trace = trace_model("TimeDrivenLIF_Exponential_Decay")
    return compare_scalar("TimeDrivenLIF_Exponential_Decay voltage", [s["state_variables"][0] for s in trace], reference_lif(-65.0, 20.0, 1.0, STEPS))


def test_lif_double():
    trace = trace_model("TimeDrivenLIF_Exponential_double")
    return compare_scalar("TimeDrivenLIF_Exponential_double voltage", [s["state_variables"][0] for s in trace], reference_lif(-60.0, 20.0, 1.0, STEPS))


def test_lif_voltage_jump():
    params = {"V_rest": nb.float32(0.0), "V_reset": nb.float32(-5.0), "V_th": nb.float32(20.0), "tau": nb.float32(10.0), "R": nb.float32(1.0)}
    initial, trace = trace_model_with_initial("TimeDrivenLIF_Voltage_jump", params)
    v0 = float(initial["state_variables"][0])
    return compare_scalar("TimeDrivenLIF_Voltage_jump voltage", [s["state_variables"][0] for s in trace], reference_lif(0.0, 10.0, 1.0, STEPS, initial_voltage=v0)) | {"initial_voltage": v0}


def test_izhikevich():
    trace = trace_model("TimeDrivenIzhikevic_Exponential_Decay", steps=STEPS)
    # Match the native float32 arithmetic and operation order.
    f = np.float32
    v = f(-65.0)
    u = f(f(0.2) * f(-65.0))
    expected_v = []
    expected_u = []
    # Neuron_State_Vector::InitNeuronState initializes LastSpike to 10000.
    last_spike = 10000
    for outer_step in range(STEPS):
        substeps = INTERNAL_SUBSTEPS_PER_RUN if outer_step == 0 else 1
        for substep in range(substeps):
            if last_spike > 0:
                # The first internal substep precedes the delay=1 current event.
                effective_current = f(0.0 if outer_step == 0 and substep == 0 else CURRENT)
                dv = f(f(f(0.04) * v * v) + f(f(5.0) * v) + f(140.0) - u + effective_current)
                du = f(f(0.02) * (f(0.2) * v - u))
                v = f(v + f(dv * f(DT)))
                u = f(u + f(du * f(DT)))
            last_spike += 1
            if v > 30.0:
                v = f(-65.0)
                u = f(u + f(8.0))
                last_spike = 0
        expected_v.append(float(v))
        expected_u.append(float(u))
    actual_v = [s["state_variables"][0] for s in trace]
    actual_u = [s["state_variables"][1] for s in trace]
    v_result = compare_scalar("TimeDrivenIzhikevic voltage", actual_v, expected_v)
    u_result = compare_scalar("TimeDrivenIzhikevic recovery", actual_u, expected_u)
    return {"name": "TimeDrivenIzhikevic_Exponential_Decay", "pass": v_result["pass"] and u_result["pass"], "voltage": v_result, "recovery": u_result}


def poisson_hash(neuron_index, time_step):
    x = (neuron_index * 747796405 + (time_step + 1) * 2891336453 + 0x9E3779B9) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 2246822519) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 3266489917) & 0xFFFFFFFF
    x ^= x >> 16
    return (x & 0x00FFFFFF) / float(0x01000000)


def test_poisson():
    steps = 200
    params = {"rate_bias_hz": nb.float32(40.0), "rate_gain_hz_per_current": nb.float32(0.0)}
    n = nb.Network()
    n.add_layer(nb.NeuronLayer.input_current(1))
    n.add_layer(nb.NeuronLayer("PoissonRate", 1, parameters=params, output=True))
    n.connect(nb.Connection(0, 1, synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(n, nb.SimulationConfig(steps=steps, timestep=1.0)).init()
    sim.add_external_currents([0], [0], [0.0]).run()
    actual = [(int(item["neuron_id"]), int(item["time"])) for item in sim.output_spikes()]
    # The source layer is neuron 0; the output layer is assigned global id 1.
    expected = [(1, t) for t in range(steps) if poisson_hash(0, t) < 40.0 * 0.001]
    return {"name": "PoissonRate deterministic sequence", "pass": actual == expected, "actual": actual, "expected": expected, "count_actual": len(actual), "count_expected": len(expected)}


results = [test_lif_decay(), test_lif_double(), test_lif_voltage_jump(), test_izhikevich(), test_poisson()]
report = {"timestep": DT, "steps": STEPS, "internal_substeps_per_run": INTERNAL_SUBSTEPS_PER_RUN, "current": CURRENT, "tolerance": TOL, "results": results, "pass": all(item["pass"] for item in results)}
(ROOT / "numerical_accuracy_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
raise SystemExit(0 if report["pass"] else 1)
