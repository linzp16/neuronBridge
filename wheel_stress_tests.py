from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path

import neuronbridge as nb

ROOT = Path(r"D:\testfile\stress")
ROOT.mkdir(parents=True, exist_ok=True)
rows = []


def run_case(name, fn):
    started = time.perf_counter()
    try:
        detail = fn()
        rows.append({"name": name, "status": "PASS", "elapsed_s": round(time.perf_counter() - started, 3), "detail": detail})
    except Exception as exc:
        rows.append({"name": name, "status": "FAIL", "elapsed_s": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def repeated_small_runs():
    checks = []
    for i in range(20):
        network = nb.Network()
        network.add_layer(nb.NeuronLayer.input_spike(4))
        network.add_layer(nb.NeuronLayer.lif_decay(8, output=True))
        network.connect(nb.Connection(list(range(4)) * 2, [4 + j for j in range(8)], weight=0.2, delay=1))
        sim = nb.Simulation(network, nb.SimulationConfig(steps=100, timestep=1.0, queues=1)).init()
        sim.add_external_spikes([0, 10, 20, 30], [0, 1, 2, 3]).run()
        checks.append(len(sim.neuron_states(list(range(4, 12)))))
        del sim
        gc.collect()
    return {"iterations": len(checks), "state_counts": sorted(set(checks))}


def medium_dense_run():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 256, parameters={"dense_subnetwork_name": "stress_a"}))
    network.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 256, parameters={"dense_subnetwork_name": "stress_b"}))
    network.connect(nb.Connection(0, 1, weight=0.1, max_weight=1.0, delay=1))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=500, timestep=1.0, queues=1)).init()
    sim.run()
    a = sim.dense_subnetwork_snapshot("stress_a")
    b = sim.dense_subnetwork_snapshot("stress_b")
    return {"dense_count": sim.dense_subnetwork_count, "a_neurons": len(a["membrane_v"]), "b_neurons": len(b["membrane_v"]), "steps": 500}


def large_dense_run():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 1024, parameters={"dense_subnetwork_name": "stress_large"}))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=1000, timestep=1.0, queues=1)).init()
    sim.run()
    snap = sim.dense_subnetwork_snapshot("stress_large")
    return {"dense_count": sim.dense_subnetwork_count, "neurons": len(snap["membrane_v"]), "steps": 1000}


def run_public_long_example():
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(2, output=True))
    network.connect(nb.Connection(0, [1, 2], weight=[1.0, 0.5], delay=[1, 2]))
    sim = nb.Simulation(network, nb.SimulationConfig(steps=1200, timestep=1.0, event_queue="timing_wheel", timing_wheel_size=2048)).init()
    sim.add_external_spikes([0, 100, 200, 300, 400], [0, 0, 0, 0, 0]).run(1200)
    return {"steps": 1200, "outputs": len(sim.output_spikes()), "state_ids": [s["original_neuron_id"] for s in sim.neuron_states([1, 2])]}


run_case("repeated_small_runs", repeated_small_runs)
run_case("medium_dense_run", medium_dense_run)
run_case("large_dense_run", large_dense_run)
run_case("long_timing_wheel_run", run_public_long_example)
result = {"ok": all(row["status"] == "PASS" for row in rows), "cases": rows}
(ROOT / "stress_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ok"] else 1)
