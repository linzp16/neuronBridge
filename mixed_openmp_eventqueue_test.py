from __future__ import annotations

import argparse
import json
from pathlib import Path

import neuronbridge as nb


INPUT_SIZE = 4
RELAY_SIZE = 4
DENSE_SIZE = 8
OUTPUT_SIZE = 2
DENSE_NAMES = ["dense_a", "dense_b"]
STEPS = 160


def all_to_all(source_begin: int, source_count: int, target_begin: int,
               target_count: int, weight: float, delay: int = 1) -> nb.Connection:
    return nb.Connection(
        source=[source_begin + i for i in range(source_count) for _ in range(target_count)],
        target=[target_begin + j for _ in range(source_count) for j in range(target_count)],
        weight=[weight] * (source_count * target_count),
        max_weight=[100.0] * (source_count * target_count),
        delay=[delay] * (source_count * target_count),
    )


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(INPUT_SIZE))
    network.add_layer(nb.NeuronLayer.lif_double(RELAY_SIZE, monitored=True))
    relay_begin = INPUT_SIZE
    network.connect(all_to_all(0, INPUT_SIZE, relay_begin, RELAY_SIZE, 12.0, 1))

    dense_begin = relay_begin + RELAY_SIZE
    for index, name in enumerate(DENSE_NAMES):
        network.add_layer(nb.NeuronLayer.lif_double(
            DENSE_SIZE,
            dense_name=name,
            monitored=True,
            t_ref=nb.int32(2),
            dense_queue_index=nb.int32(index),
        ))
        begin = dense_begin + index * DENSE_SIZE
        network.connect(all_to_all(relay_begin, RELAY_SIZE, begin, DENSE_SIZE, 8.0, 1))

    output_begin = dense_begin + len(DENSE_NAMES) * DENSE_SIZE
    network.add_layer(nb.NeuronLayer.lif_decay(OUTPUT_SIZE, monitored=True, output=True))
    for index in range(len(DENSE_NAMES)):
        begin = dense_begin + index * DENSE_SIZE
        network.connect(all_to_all(begin, DENSE_SIZE, output_begin, OUTPUT_SIZE, 3.0, 2))

    # Exercise two OuterDynamic models and both connection types from the
    # relay layer, whose source neurons are distributed across queues.
    for counter_name in ("counter_a", "counter_b"):
        network.add_outer_dynamic(nb.OuterDynamic.spike_counter(
            name=counter_name, slot_count=RELAY_SIZE, type_count=2))
    relay_ids = list(range(relay_begin, relay_begin + RELAY_SIZE))
    for target, synapse_type, weight in ((0, 0, 1.25), (1, 1, 0.75)):
        network.connect_outer_dynamic(nb.OuterDynamicConnection(
            source=relay_ids,
            target_outer_dynamic=[target] * RELAY_SIZE,
            target_joint=list(range(RELAY_SIZE)),
            synapse_type=[synapse_type] * RELAY_SIZE,
            weight=[weight] * RELAY_SIZE,
            delay=[1],
        ))
    return network


def run_case(queue_type: str, queues: int, root: Path) -> dict:
    sim = nb.Simulation(build_network(), nb.SimulationConfig(
        steps=STEPS,
        timestep=0.1,
        queues=queues,
        event_queue=queue_type,
        timing_wheel_size=256 if queue_type == "timing_wheel" else 0,
    ))
    sim.enable_debug_monitor(nb.DebugMonitorConfig(
        output_dir=root,
        sample_interval_steps=1,
        flush_interval_steps=20,
        record_spikes=True,
        record_state=True,
        record_weights=False,
        record_pending_channels=False,
        record_outer_dynamic_state=False,
        all_neurons=True,
        dense_local_neuron_ids={name: list(range(DENSE_SIZE)) for name in DENSE_NAMES},
    ))
    sim.init()
    event_steps = [0, 10, 20, 40, 80, 120]
    for neuron_id in range(INPUT_SIZE):
        sim.add_external_spikes(event_steps, [neuron_id] * len(event_steps))
    for _ in range(8):
        sim.run(20)
    sim.flush()
    return {
        "queue_type": queue_type,
        "queues": queues,
        "dense": {name: sim.dense_subnetwork_snapshot(name) for name in DENSE_NAMES},
        "counter_a": sim.outer_dynamic_spike_counter_snapshot("counter_a"),
        "counter_b": sim.outer_dynamic_spike_counter_snapshot("counter_b"),
        "output_spikes": sim.output_spikes(),
    }


def compare(reference: dict, candidate: dict) -> dict:
    fields = ("dense", "counter_a", "counter_b", "output_spikes")
    values = {field: reference[field] == candidate[field] for field in fields}
    return {**values, "pass": all(values.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        event_queue: {
            queues: run_case(event_queue, queues, args.output_dir / event_queue / f"queues_{queues}")
            for queues in (1, 2, 4)
        }
        for event_queue in ("heap", "timing_wheel")
    }
    report = {
        "network": {
            "input_neurons": INPUT_SIZE,
            "relay_neurons": RELAY_SIZE,
            "dense_subnetworks": len(DENSE_NAMES),
            "dense_neurons_per_subnetwork": DENSE_SIZE,
            "outer_dynamic_models": 2,
            "output_neurons": OUTPUT_SIZE,
            "steps": STEPS,
            "timing_wheel_size": 256,
        },
        "counts": {
            event_queue: {
                str(queues): {
                    "counter_a": case["counter_a"].get("spike_counts", []),
                    "counter_b": case["counter_b"].get("spike_counts", []),
                    "output_spike_count": len(case["output_spikes"]),
                }
                for queues, case in queue_cases.items()
            }
            for event_queue, queue_cases in cases.items()
        },
        "comparisons": {
            event_queue: {
                f"queues_1_vs_{queues}": compare(queue_cases[1], queue_cases[queues])
                for queues in (2, 4)
            }
            for event_queue, queue_cases in cases.items()
        },
        "cross_event_queue": compare(cases["heap"][1], cases["timing_wheel"][1]),
    }
    report["pass"] = (
        all(item["pass"] for event_cases in report["comparisons"].values() for item in event_cases.values())
        and report["cross_event_queue"]["pass"]
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
