"""CUDA runtime acceptance using only the public API of an installed wheel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import neuronbridge as nb


def dense_layer(model: str, count: int, name: str, **parameters) -> nb.NeuronLayer:
    return nb.NeuronLayer(
        model,
        count,
        parameters={"dense_subnetwork_name": name, **parameters},
    )


def run_rule(rule_name: str, *, uses_trigger: bool) -> dict:
    dense_name = f"wheel_cuda_{rule_name.lower()}"
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2 if uses_trigger else 1))
    pre_id = 2 if uses_trigger else 1
    post_id = pre_id + 1
    network.add_layer(
        dense_layer(
            "CustomLifConductanceV1",
            2,
            dense_name,
            V_rest=-60.0,
            V_reset=-60.0,
            V_th=-59.8,
            t_ref=0.0,
        )
    )
    trigger_id = post_id + 1
    if uses_trigger:
        network.add_layer(dense_layer("TriggerRelayNeuronModel", 1, dense_name, threshold=0.5))
    network.add_learning_rule(nb.LearningRule(rule_name))
    network.connect(nb.Connection(0, pre_id, weight=1.0, max_weight=1.0, delay=0))
    network.connect(
        nb.Connection(pre_id, post_id, weight=8.0, max_weight=20.0, delay=1, synapse_rule=0)
    )
    if uses_trigger:
        network.connect(nb.Connection(1, trigger_id, weight=1.0, max_weight=1.0, delay=0))
        network.connect(
            nb.Connection(trigger_id, post_id, weight=1.0, max_weight=1.0, delay=3, trigger_rule=0)
        )

    simulation = nb.Simulation(network, nb.SimulationConfig(steps=16, timestep=1.0))
    simulation.init()
    simulation.set_dense_subnetwork_full_firing_export_enabled(dense_name)
    simulation.add_external_spikes(
        [0, 1, 5] if uses_trigger else [0, 7],
        [0, 1, 1] if uses_trigger else [0, 0],
    )
    initial_weight = simulation.get_connection_weight(1)
    trace = []
    for _ in range(10):
        simulation.run(1)
        snapshot = simulation.dense_subnetwork_snapshot(dense_name)
        trace.append(
            {
                "time_step": snapshot["time_step"],
                "firing_ids": list(snapshot["full_firing_ids"]),
                "plastic_weight": simulation.get_connection_weight(1),
            }
        )
    final_weight = simulation.get_connection_weight(1)
    snapshot = simulation.dense_subnetwork_snapshot(dense_name)
    if not snapshot["cuda_build_enabled"]:
        raise RuntimeError("installed wheel was not built with CUDA")
    if not snapshot["gpu_backend_ready"] or not snapshot["carlsim_like_gpu_active"]:
        raise RuntimeError("installed wheel did not activate the Dense CUDA runtime")
    if not any(item["firing_ids"] for item in trace):
        raise RuntimeError(f"installed wheel produced no dense firing for {rule_name}")
    return {
        "rule": rule_name,
        "initial_weight": initial_weight,
        "final_weight": final_weight,
        "weight_changed": final_weight != initial_weight,
        "runtime_trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forbid-path", action="append", default=[])
    args = parser.parse_args()
    resolved_sys_path = [str(Path(item or ".").resolve()) for item in sys.path]
    forbidden_hits = [
        item
        for item in resolved_sys_path
        if any(str(Path(root).resolve()).lower() in item.lower() for root in args.forbid_path)
    ]
    if forbidden_hits:
        raise RuntimeError(f"source/build path leaked into installed-wheel sys.path: {forbidden_hits}")
    backend = nb.backend_info()
    if not backend.get("native_extension_loaded") or not backend.get("cuda_enabled"):
        raise RuntimeError(f"installed native/CUDA backend unavailable: {backend}")
    rules = [
        run_rule("CustomRStdpV1", uses_trigger=True),
        run_rule("CustomRStdpPersistentV1", uses_trigger=True),
        run_rule("CustomPairStdpV1", uses_trigger=False),
    ]
    if not rules[-1]["weight_changed"]:
        raise RuntimeError("installed-wheel pair-STDP path did not update weight")
    report = {
        "result": "PASS",
        "backend": backend,
        "rules": rules,
        "sys_path": resolved_sys_path,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
