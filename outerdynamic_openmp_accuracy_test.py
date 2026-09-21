from __future__ import annotations

import argparse
import json
from pathlib import Path

import neuronbridge as nb


SOURCE_COUNT = 8
COUNTER_SLOTS = 8
STEPS = 120


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(SOURCE_COUNT))
    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(
        name="counter_a",
        slot_count=COUNTER_SLOTS,
        type_count=2,
    ))
    network.add_outer_dynamic(nb.OuterDynamic.spike_counter(
        name="counter_b",
        slot_count=COUNTER_SLOTS,
        type_count=2,
    ))

    # Every input neuron targets both counters.  The two connection blocks use
    # opposite synapse types so both counters' per-type paths are exercised.
    source_ids = list(range(SOURCE_COUNT))
    for target, synapse_type, weight in ((0, 0, 1.25), (1, 1, 0.75)):
        network.connect_outer_dynamic(nb.OuterDynamicConnection(
            source=source_ids,
            target_outer_dynamic=[target] * SOURCE_COUNT,
            target_joint=source_ids,
            synapse_type=[synapse_type] * SOURCE_COUNT,
            weight=[weight] * SOURCE_COUNT,
            delay=[1] * SOURCE_COUNT,
        ))
    return network


def run_case(queues: int, output_dir: Path) -> dict:
    sim = nb.Simulation(
        build_network(),
        nb.SimulationConfig(steps=STEPS, timestep=0.1, queues=queues),
    )
    sim.enable_debug_monitor(nb.DebugMonitorConfig(
        output_dir=output_dir,
        sample_interval_steps=1,
        flush_interval_steps=20,
        record_spikes=True,
        record_state=True,
        record_weights=False,
        record_pending_channels=False,
        record_outer_dynamic_state=True,
        all_neurons=True,
    ))
    sim.init()
    event_steps = [0, 10, 20, 40, 80]
    for neuron_id in range(SOURCE_COUNT):
        sim.add_external_spikes(event_steps, [neuron_id] * len(event_steps))
    for _ in range(6):
        sim.run(20)
    sim.flush()
    return {
        "queues": queues,
        "counter_a": sim.outer_dynamic_spike_counter_snapshot("counter_a"),
        "counter_b": sim.outer_dynamic_spike_counter_snapshot("counter_b"),
        "output_spikes": sim.output_spikes(),
    }


def compare(reference: dict, candidate: dict) -> dict:
    fields = ("counter_a", "counter_b", "output_spikes")
    equal = {field: reference[field] == candidate[field] for field in fields}
    return {**equal, "pass": all(equal.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {q: run_case(q, args.output_dir / f"queues_{q}") for q in (1, 2, 4)}
    report = {
        "network": {
            "input_neurons": SOURCE_COUNT,
            "outer_dynamic_models": 2,
            "counter_slots_per_model": COUNTER_SLOTS,
            "steps": STEPS,
            "input_events_per_neuron": 5,
            "expected_events_per_counter_slot": 5,
        },
        "cases": {
            str(q): {
                "counter_a_counts": case["counter_a"].get("spike_counts", []),
                "counter_b_counts": case["counter_b"].get("spike_counts", []),
                "counter_a_by_type": case["counter_a"].get("spike_counts_by_type", []),
                "counter_b_by_type": case["counter_b"].get("spike_counts_by_type", []),
            }
            for q, case in cases.items()
        },
        "comparisons": {
            f"queues_1_vs_{q}": compare(cases[1], cases[q]) for q in (2, 4)
        },
    }
    report["pass"] = all(item["pass"] for item in report["comparisons"].values())
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
