from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import neuronbridge as nb

ROOT = Path(r"D:\testfile\complex_networks\large")
ROOT.mkdir(parents=True, exist_ok=True)
rows = []


def batched_connections(sources, targets, *, synapse_type=0, weight=0.1, delay=1, fanout=4):
    source_values, target_values, types, weights, delays = [], [], [], [], []
    for source in sources:
        for offset in range(fanout):
            source_values.append(source)
            target_values.append(targets[(source + offset * 17) % len(targets)])
            types.append(synapse_type)
            weights.append(weight)
            delays.append(delay + offset % 3)
    return nb.Connection(source_values, target_values, synapse_type=types, weight=weights, delay=delays)


def run_case(name, build, steps):
    started = time.perf_counter()
    try:
        network = build()
        definition = network.to_dict()
        init_started = time.perf_counter()
        sim = nb.Simulation(network, nb.SimulationConfig(steps=steps, timestep=1.0, queues=2, event_queue="timing_wheel", timing_wheel_size=512)).init()
        init_s = time.perf_counter() - init_started
        run_started = time.perf_counter()
        sim.run()
        run_s = time.perf_counter() - run_started
        result = {
            "layers": len(network.layers),
            "connections": len(network.connections),
            "neurons": network.neuron_count,
            "steps": steps,
            "init_s": round(init_s, 3),
            "run_s": round(run_s, 3),
            "output_spikes": len(sim.output_spikes()),
            "dense_count": sim.dense_subnetwork_count,
        }
        if name == "large_legacy_mixed":
            result["states_read"] = len(sim.neuron_states([256]))
        if sim.dense_subnetwork_count:
            result["dense_snapshots"] = {
                sim.dense_subnetwork_name(i): len(sim.dense_subnetwork_snapshot(i)["membrane_v"])
                for i in range(sim.dense_subnetwork_count)
            }
        rows.append({"name": name, "status": "PASS", "elapsed_s": round(time.perf_counter() - started, 3), "definition_counts": {k: len(v) for k, v in definition.items() if isinstance(v, list)}, "result": result})
    except Exception as exc:
        rows.append({"name": name, "status": "FAIL", "elapsed_s": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def large_legacy_mixed():
    n = nb.Network()
    n.add_layer(nb.NeuronLayer.input_spike(128))
    n.add_layer(nb.NeuronLayer.input_current(128))
    n.add_layer(nb.NeuronLayer.lif_decay(1024, monitored=True))
    n.add_layer(nb.NeuronLayer("TimeDrivenIzhikevic_Exponential_Decay", 512))
    n.add_layer(nb.NeuronLayer.poisson_rate(512, rate_bias_hz=15.0, rate_gain_hz_per_current=1.0))
    n.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 128, output=True))
    n.connect(batched_connections(range(0, 128), range(256, 1280), weight=0.4, fanout=8))
    n.connect(batched_connections(range(128, 256), range(256, 1280), synapse_type=3, weight=0.2, fanout=8))
    n.connect(batched_connections(range(256, 1280), range(1280, 1792), weight=0.08, fanout=2))
    n.connect(batched_connections(range(1280, 1792), range(1792, 2304), weight=0.05, fanout=2))
    n.connect(batched_connections(range(1792, 2304), range(2304, 2432), weight=0.2, fanout=2))
    return n


def large_dense_cascade():
    n = nb.Network()
    n.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 1024, parameters={"dense_subnetwork_name": "large_dense_a"}))
    n.add_layer(nb.NeuronLayer("CustomLifConductanceV1", 1024, parameters={"dense_subnetwork_name": "large_dense_b", "V_rest": -60.0, "V_reset": -60.0, "V_th": -59.8, "t_ref": 0.0}))
    n.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 512, parameters={"dense_subnetwork_name": "large_dense_c"}))
    n.connect(nb.Connection(0, 1, weight=0.1, max_weight=1.0, delay=1))
    n.connect(nb.Connection(1, 2, weight=0.1, max_weight=1.0, delay=2))
    return n


run_case("large_legacy_mixed", large_legacy_mixed, 200)
run_case("large_dense_cascade", large_dense_cascade, 300)
result = {"ok": all(row["status"] == "PASS" for row in rows), "cases": rows}
(ROOT / "large_mixed_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if result["ok"] else 1)
