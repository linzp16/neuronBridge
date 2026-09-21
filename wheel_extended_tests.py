from __future__ import annotations

import json
from pathlib import Path
import traceback

import neuronbridge as nb

ROOT = Path(r"D:\testfile\11_regression")
ROOT.mkdir(parents=True, exist_ok=True)
rows = []


def record(name, fn):
    try:
        value = fn()
        rows.append({"name": name, "status": "PASS", "detail": value})
    except Exception as exc:
        rows.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def make_network(input_count=16, neuron_count=32, dense_name=None):
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(input_count))
    network.add_layer(nb.NeuronLayer.lif_double(neuron_count, dense_name=dense_name, output=True, monitored=True))
    source = []
    target = []
    for s in range(input_count):
        for t in range(input_count, input_count + neuron_count):
            source.append(s)
            target.append(t)
    network.connect(nb.Connection(source=source, target=target, weight=0.25, delay=1))
    return network


def long_heap_run():
    sim = nb.Simulation(make_network(), nb.SimulationConfig(steps=200, timestep=1.0, queues=1, event_queue="heap")).init()
    times = list(range(0, 200, 4))
    sim.add_external_spikes(times, [i % 16 for i in range(len(times))]).run()
    return {"steps": 200, "states": len(sim.neuron_states(list(range(16, 48)))), "spikes": len(sim.output_spikes())}


def long_timing_wheel_run():
    sim = nb.Simulation(make_network(), nb.SimulationConfig(steps=200, timestep=1.0, queues=1, event_queue="timing_wheel", timing_wheel_size=256)).init()
    sim.add_external_spikes(list(range(0, 200, 5)), [i % 16 for i in range(40)]).run()
    return {"steps": 200, "states": len(sim.neuron_states(list(range(16, 48)))), "spikes": len(sim.output_spikes())}


def repeated_run_reset():
    sim = nb.Simulation(make_network(2, 2), nb.SimulationConfig(steps=20, timestep=1.0)).init()
    sim.add_external_spikes([0, 1], [0, 1]).run(5)
    first = sim.neuron_states([2, 3])
    sim.run(5)
    second = sim.neuron_states([2, 3])
    sim.reset(preserve_weights=True).run(5)
    third = sim.neuron_states([2, 3])
    return {"first": first, "second": second, "after_reset": third}


def dense_run_and_snapshot():
    sim = nb.Simulation(make_network(8, 16, "extended_dense"), nb.SimulationConfig(steps=30, timestep=1.0)).init()
    sim.set_dense_subnetwork_full_firing_export_enabled("extended_dense")
    sim.add_external_spikes([0, 1, 2], [0, 1, 2]).run()
    snap = sim.dense_subnetwork_snapshot("extended_dense")
    return {"dense_count": sim.dense_subnetwork_count, "snapshot_keys": sorted(snap), "neuron_count": len(snap["membrane_v"])}


def weight_roundtrip():
    path = ROOT / "extended_weights.bin"
    sim = nb.Simulation(make_network(2, 2), nb.SimulationConfig(steps=10, timestep=1.0)).init()
    before = sim.get_connection_weight(0)
    sim.set_connection_weight(0, 0.75).save_weights(path)
    sim.set_connection_weight(0, 0.15).load_weights(path)
    after = sim.get_connection_weight(0)
    return {"before": before, "after": after, "file_exists": path.exists(), "file_size": path.stat().st_size}


def invalid_definition_cases():
    checks = []
    for fn in [
        lambda: nb.SimulationConfig(steps=0, timestep=1.0),
        lambda: nb.SimulationConfig(steps=1, timestep=1.0, event_queue="bad"),
        lambda: nb.NeuronLayer.input_spike(0),
        lambda: nb.InputConvFrame(0, 0, 2, 2, 1, b"bad"),
    ]:
        try:
            fn()
        except (ValueError, TypeError):
            checks.append(True)
        else:
            checks.append(False)
    assert all(checks), checks
    return {"expected_exceptions": len(checks), "all_rejected": all(checks)}


def malformed_protocol_cases():
    request = nb.InputConvFrameRequest(1, 0, 0, 2, 2, 1)
    payload = nb.pack_input_conv_frame_request(request)
    checks = []
    for bad in [payload[:-1], b"\x00" * len(payload)]:
        try:
            nb.unpack_input_conv_frame_request(bad)
        except (ValueError, TypeError, Exception):
            checks.append(True)
    assert len(checks) == 2
    return {"malformed_requests_rejected": len(checks)}


record("long_heap_run", long_heap_run)
record("long_timing_wheel_run", long_timing_wheel_run)
record("repeated_run_reset", repeated_run_reset)
record("dense_run_and_snapshot", dense_run_and_snapshot)
record("weight_roundtrip", weight_roundtrip)
record("invalid_definition_cases", invalid_definition_cases)
record("malformed_protocol_cases", malformed_protocol_cases)

result = {"ok": all(row["status"] == "PASS" for row in rows), "cases": rows}
(ROOT / "extended_test_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
raise SystemExit(0 if result["ok"] else 1)
