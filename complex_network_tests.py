from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import neuronbridge as nb

ROOT = Path(r"D:\testfile\complex_networks")
ROOT.mkdir(parents=True, exist_ok=True)
rows = []


def run_case(name, builder, runner):
    print(f"START {name}", flush=True)
    try:
        network = builder()
        print(f"BUILT {name}", flush=True)
        definition = network.to_dict()
        print(f"DICT {name}", flush=True)
        native = network.to_native()
        print(f"NATIVE {name}", flush=True)
        result = runner(network, native)
        print(f"RAN {name}", flush=True)
        rows.append({"name": name, "status": "PASS", "definition_counts": {k: len(v) for k, v in definition.items() if isinstance(v, list)}, "result": result})
        print(f"PASS {name}", flush=True)
    except Exception as exc:
        rows.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)


def mixed_learning_network():
    n = nb.Network()
    n.add_layer(nb.NeuronLayer.input_spike(2))
    n.add_layer(nb.NeuronLayer.input_current(2))
    n.add_layer(nb.NeuronLayer.lif_decay(2, monitored=True))
    n.add_layer(nb.NeuronLayer("TimeDrivenIzhikevic_Exponential_Decay", 2, monitored=True))
    n.add_layer(nb.NeuronLayer.poisson_rate(2, rate_bias_hz=20.0, rate_gain_hz_per_current=2.0))
    n.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 2, output=True))
    n.connect(nb.Connection([0, 1], [4, 5], weight=[0.5, 0.7], delay=[1, 2]))
    # InputCurrent connections must use the dedicated current synapse type.
    n.connect(nb.Connection([2, 3], [6, 7], synapse_type=3, weight=0.4, delay=1))
    n.connect(nb.Connection([4, 5], [8, 9], weight=0.2, delay=1))
    n.connect(nb.Connection([6, 7], [10, 11], weight=1.0, delay=1))
    n.add_learning_rule(nb.LearningRule("CustomPairStdpV1"))
    n.add_learning_rule(nb.LearningRule("CustomRStdpV1"))
    return n


def run_mixed(n, _native):
    sim = nb.Simulation(n, nb.SimulationConfig(steps=40, timestep=1.0, queues=2, event_queue="heap")).init()
    sim.add_external_spikes([0, 1, 5, 10], [0, 1, 0, 1])
    sim.add_external_currents([0, 2, 6], [2, 3, 2], [5.0, 8.0, 5.0]).run()
    # Query only stateful monitored LIF/Izhikevich neurons; relay/Poisson models
    # do not expose a legacy Neuron_State_Vector through neuron_states().
    return {"layers": len(n.layers), "connections": len(n.connections), "rules": len(n.learning_rules), "states": len(sim.neuron_states(list(range(4, 8)))), "outputs": len(sim.output_spikes())}


def dense_cascade_network():
    n = nb.Network()
    n.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 64, parameters={"dense_subnetwork_name": "dense_a"}))
    n.add_layer(nb.NeuronLayer("CustomLifConductanceV1", 64, parameters={"dense_subnetwork_name": "dense_b", "V_rest": -60.0, "V_reset": -60.0, "V_th": -59.8, "t_ref": 0.0}))
    n.add_layer(nb.NeuronLayer("TriggerRelayNeuronModel", 16, parameters={"dense_subnetwork_name": "dense_c", "threshold": 0.5}))
    n.connect(nb.Connection(0, 1, weight=0.15, max_weight=1.0, delay=1))
    n.connect(nb.Connection(1, 2, weight=0.2, max_weight=1.0, delay=2))
    n.add_learning_rule(nb.LearningRule("CustomRStdpPersistentV1"))
    return n


def run_dense_cascade(n, _native):
    sim = nb.Simulation(n, nb.SimulationConfig(steps=80, timestep=1.0, queues=2, event_queue="timing_wheel", timing_wheel_size=128)).init()
    for name in ["dense_a", "dense_b", "dense_c"]:
        sim.set_dense_subnetwork_full_firing_export_enabled(name)
    sim.run()
    snapshots = {name: len(sim.dense_subnetwork_snapshot(name)["membrane_v"]) for name in ["dense_a", "dense_b", "dense_c"]}
    return {"dense_count": sim.dense_subnetwork_count, "snapshots": snapshots}


def inputconv_hybrid_network():
    n = nb.Network()
    n.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_double", 8, parameters={"dense_subnetwork_name": "inputconv_dense"}))
    n.add_layer(nb.NeuronLayer.lif_decay(4, output=True))
    n.connect(nb.Connection(0, [8, 9, 10, 11], weight=0.5, delay=1))
    n.add_input_conv(nb.InputConv.v1_grating(width=2, height=2, channels=1, output_target="dense_subnetwork", target_dense_subnetwork_name="inputconv_dense", output_source_indices=[0, 1, 2, 3], output_target_neuron_ids=[0, 1, 2, 3], output_scales=[1.0, 0.8, 0.6, 0.4]))
    return n


def run_inputconv_hybrid(n, _native):
    sim = nb.Simulation(n, nb.SimulationConfig(steps=30, timestep=1.0, queues=1, event_queue="heap")).init()
    sim.add_input_conv_frames("hybrid_source", [nb.InputConvFrame(0, 0, 2, 2, 1, bytes([1, 2, 3, 4])), nb.InputConvFrame(1, 0, 2, 2, 1, bytes([4, 3, 2, 1]))])
    sim.bind_input_conv_frame_source(0, "hybrid_source")
    sim.run()
    return {"input_conv_count": sim.input_conv_count, "input_count": len(sim.input_conv_input(0)), "output_count": sim.input_conv_output_count(0), "dense_count": sim.dense_subnetwork_count}


def outer_closed_loop_network():
    n = nb.Network()
    n.add_layer(nb.NeuronLayer.input_spike(4))
    n.add_layer(nb.NeuronLayer.lif_decay(4, output=True, monitored=True))
    n.connect(nb.Connection(0, [4, 5, 6, 7], weight=1.0, delay=1))
    counter = nb.OuterDynamic.spike_counter(name="hybrid_counter", slot_count=2, type_count=2)
    counter.update_timestep = 2
    counter.communication_interval = 2
    n.add_outer_dynamic(counter)
    n.connect_outer_dynamic(nb.OuterDynamicConnection(source=[4, 5, 6, 7], target_outer_dynamic=0, target_joint=[0, 0, 1, 1], synapse_type=[0, 1, 0, 1], weight=[1.0, 1.0, 1.0, 1.0], delay=1))
    return n


def run_outer(n, _native):
    sim = nb.Simulation(n, nb.SimulationConfig(steps=50, timestep=1.0, queues=2, event_queue="timing_wheel", timing_wheel_size=128)).init()
    sim.add_external_spikes(list(range(0, 20, 2)), [i % 4 for i in range(10)]).run()
    return {"outer_dynamic_count": len(n.outer_dynamics), "counter": sim.outer_dynamic_spike_counter_snapshot("hybrid_counter"), "outputs": len(sim.output_spikes())}


selected = sys.argv[1:] or ["mixed", "dense", "inputconv", "outer"]
if "mixed" in selected:
    run_case("mixed_neurons_and_learning", mixed_learning_network, run_mixed)
if "dense" in selected:
    run_case("multi_dense_cascade", dense_cascade_network, run_dense_cascade)
if "inputconv" in selected:
    run_case("inputconv_main_dense_hybrid", inputconv_hybrid_network, run_inputconv_hybrid)
if "outer" in selected:
    run_case("outer_dynamic_closed_loop", outer_closed_loop_network, run_outer)

result = {"ok": all(row["status"] == "PASS" for row in rows), "cases": rows}
(ROOT / "complex_test_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if result["ok"] else 1)
