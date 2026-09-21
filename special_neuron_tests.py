from __future__ import annotations

import json
import traceback
from pathlib import Path

import neuronbridge as nb

OUT = Path(r"D:\testfile\special_neuron_tests")
OUT.mkdir(parents=True, exist_ok=True)


def describe_matrix() -> dict:
    models = ["TriggerRelayNeuronModel", "EDLUTLikeLIF", "CustomLifConductanceV1", "InputSpikeNeuronModel", "InputCurrentNeuronModel"]
    matrix = {}
    for model in models:
        matrix[model] = {}
        for backend in ("legacy_cpu", "legacy_gpu", "dense_gpu"):
            try:
                value = nb.catalog.describe_model(model, backend=backend)
                matrix[model][backend] = {"supported": True, "implementation": value.get("implementation"), "parameters": value.get("parameters")}
            except Exception as exc:
                matrix[model][backend] = {"supported": False, "error": f"{type(exc).__name__}: {exc}"}
    return matrix


def relay_case(model_name: str) -> dict:
    net = nb.Network()
    net.add_layer(nb.NeuronLayer.input_spike(1))
    net.add_layer(nb.NeuronLayer(model_name, 1, output=True))
    net.connect(nb.Connection([0], [1], weight=1.0, delay=1))
    sim = nb.Simulation(net, nb.SimulationConfig(steps=12, timestep=1.0)).init()
    sim.add_external_spikes([0, 3, 6, 9], [0, 0, 0, 0]).run()
    return {"model": model_name, "output_spikes": len(sim.output_spikes()), "spike_times": [int(x["time"]) for x in sim.output_spikes()]}


def edlut_case() -> dict:
    params = {"V_rest": -65.0, "V_reset": -65.0, "V_th": -50.0, "tau": 10.0, "R": 1.0, "t_ref": 2,
              "gexc_tau": 5.0, "Eexc": 0.0, "ginh_tau": 5.0, "Einh": -80.0}
    net = nb.Network()
    net.add_layer(nb.NeuronLayer.input_current(1))
    net.add_layer(nb.NeuronLayer("EDLUTLikeLIF_GPU", 1, parameters=params, output=True, monitored=True))
    net.connect(nb.Connection([0], [1], synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(net, nb.SimulationConfig(steps=80, timestep=1.0)).init()
    sim.add_external_currents([0], [0], [30.0]).run()
    state = sim.neuron_state(1)
    return {"model": "EDLUTLikeLIF_GPU", "spike_count": len(sim.output_spikes()), "final_state": state}


def custom_case(backend: str) -> dict:
    params = {"dense_subnetwork_name": "special_custom_dense"} if backend == "dense_gpu" else {}
    params.update({"V_rest": -60.0, "V_reset": -60.0, "V_th": -50.0, "t_ref": 50})
    net = nb.Network()
    net.add_layer(nb.NeuronLayer.input_current(1))
    model_name = "CustomLifConductanceV1_GPU" if backend == "legacy_gpu" else "CustomLifConductanceV1"
    net.add_layer(nb.NeuronLayer(model_name, 1, parameters=params, output=backend != "dense_gpu", monitored=backend != "dense_gpu"))
    net.connect(nb.Connection([0], [1], synapse_type=3, weight=1.0, delay=1))
    sim = nb.Simulation(net, nb.SimulationConfig(steps=60, timestep=1.0)).init()
    sim.add_external_currents([0], [0], [2.0]).run()
    if backend == "dense_gpu":
        snap = sim.dense_subnetwork_snapshot("special_custom_dense")
        return {"backend": backend, "spike_count": int(sum(snap["fired"])), "snapshot_keys": sorted(snap), "final_v": float(snap["membrane_v"][0])}
    state = sim.neuron_state(1)
    return {"backend": backend, "spike_count": len(sim.output_spikes()), "final_state": state}


def main() -> int:
    result = {"support_matrix": describe_matrix(), "tests": []}
    tests = [("trigger_relay_cpu", lambda: relay_case("TriggerRelayNeuronModel")),
             ("trigger_relay_gpu", lambda: relay_case("TriggerRelayNeuronModel_GPU")),
             ("edlut_like_lif_gpu", edlut_case),
             ("custom_lif_cpu", lambda: custom_case("legacy_cpu")),
             ("custom_lif_gpu", lambda: custom_case("legacy_gpu")),
             ("custom_lif_dense", lambda: custom_case("dense_gpu"))]
    for name, fn in tests:
        try:
            result["tests"].append({"name": name, "status": "PASS", "detail": fn()})
        except Exception as exc:
            result["tests"].append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    result["ok"] = all(item["status"] == "PASS" for item in result["tests"])
    (OUT / "special_neuron_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 1


raise SystemExit(main())
