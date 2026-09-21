from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import neuronbridge as nb


DENSE_NAMES = ["dense_a", "dense_b", "dense_c", "dense_d"]
DENSE_SIZE = 8
RELAY_SIZE = 3
OUTPUT_SIZE = 2


def all_to_all(source_begin: int, source_count: int, target_begin: int,
               target_count: int, weight: float, delay: int = 1) -> nb.Connection:
    count = source_count * target_count
    return nb.Connection(
        source=[source_begin + i for i in range(source_count) for _ in range(target_count)],
        target=[target_begin + j for _ in range(source_count) for j in range(target_count)],
        weight=[weight] * count,
        max_weight=[100.0] * count,
        delay=[delay] * count,
    )


def build_network(with_output_routes: bool = True, distribute_dense_queues: bool = True) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(3))             # 0..2
    network.add_layer(nb.NeuronLayer.lif_double(RELAY_SIZE, monitored=True))  # 3..5

    dense_begin = 3 + RELAY_SIZE
    for index, name in enumerate(DENSE_NAMES):
        network.add_layer(nb.NeuronLayer.lif_double(
            DENSE_SIZE,
            dense_name=name,
            monitored=True,
            t_ref=nb.int32(2),
            dense_queue_index=nb.int32(index if distribute_dense_queues else 0),
        ))
        begin = dense_begin + index * DENSE_SIZE
        network.connect(all_to_all(3, RELAY_SIZE, begin, DENSE_SIZE, weight=8.0, delay=1))

    output_begin = dense_begin + len(DENSE_NAMES) * DENSE_SIZE
    network.add_layer(nb.NeuronLayer.lif_decay(OUTPUT_SIZE, monitored=True, output=True))
    network.connect(all_to_all(0, 3, 3, RELAY_SIZE, weight=12.0, delay=1))
    if with_output_routes:
        for index in range(len(DENSE_NAMES)):
            begin = dense_begin + index * DENSE_SIZE
            network.connect(all_to_all(begin, DENSE_SIZE, output_begin, OUTPUT_SIZE, weight=3.0, delay=2))
    return network


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_snapshot(snapshot: dict) -> str:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def run_case(queues: int, root: Path, with_output_routes: bool, distribute_dense_queues: bool) -> dict:
    out = root / f"queues_{queues}"
    sim = nb.Simulation(build_network(with_output_routes, distribute_dense_queues), nb.SimulationConfig(steps=240, timestep=0.1, queues=queues))
    sim.enable_debug_monitor(nb.DebugMonitorConfig(
        output_dir=out,
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
    event_steps = [0, 10, 20, 40, 80, 120, 160, 200]
    for neuron_id in range(3):
        sim.add_external_spikes(event_steps, [neuron_id] * len(event_steps))
    for _ in range(12):
        sim.run(20)
    sim.flush()
    snapshots = {name: canonical_snapshot(sim.dense_subnetwork_snapshot(name)) for name in DENSE_NAMES}
    spike_rows = sorted(
        # local_neuron_id is queue/model-shard local and may legitimately
        # differ when OpenMP changes the model partition.  Global time and
        # neuron identity are the deterministic behavioral contract.
        (row.get("time_step", ""), row.get("global_neuron_id", ""))
        for row in read_rows(out / "spikes.csv")
    )
    return {
        "queues": queues,
        "dense_subnetwork_count": sim.dense_subnetwork_count,
        "snapshots": snapshots,
        "spikes": spike_rows,
        "spike_count": len(spike_rows),
        "output_spike_count": len(sim.output_spikes()),
    }


def compare(reference: dict, candidate: dict) -> dict:
    snapshot_names = sorted(reference["snapshots"])
    snapshot_equal = all(reference["snapshots"][name] == candidate["snapshots"].get(name)
                         for name in snapshot_names)
    return {
        "snapshot_equal": snapshot_equal,
        "spikes_equal": reference["spikes"] == candidate["spikes"],
        "spike_count_difference": candidate["spike_count"] - reference["spike_count"],
        "output_spike_count_difference": candidate["output_spike_count"] - reference["output_spike_count"],
        "dense_subnetwork_count_equal": reference["dense_subnetwork_count"] == candidate["dense_subnetwork_count"],
        "pass": snapshot_equal and reference["spikes"] == candidate["spikes"]
        and reference["output_spike_count"] == candidate["output_spike_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-output-routes", action="store_true")
    parser.add_argument("--same-dense-queue", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with_output_routes = not args.no_output_routes
    distribute_dense_queues = not args.same_dense_queue
    cases = {queues: run_case(queues, args.output_dir, with_output_routes, distribute_dense_queues) for queues in (1, 2, 4)}
    report = {
        "backend": nb.backend_info(),
        "network": {
            "main_input_neurons": 3,
            "main_relay_neurons": RELAY_SIZE,
            "dense_subnetworks": len(DENSE_NAMES),
            "dense_neurons_per_subnetwork": DENSE_SIZE,
            "dense_neurons_total": len(DENSE_NAMES) * DENSE_SIZE,
            "main_output_neurons": OUTPUT_SIZE,
            "total_neurons": 3 + RELAY_SIZE + len(DENSE_NAMES) * DENSE_SIZE + OUTPUT_SIZE,
            "steps": 240,
            "timestep": 0.1,
            "dense_to_main_output_routes": with_output_routes,
            "distributed_dense_queue_indices": distribute_dense_queues,
        },
        "counts": {
            str(q): {key: value for key, value in case.items()
                     if key in {"dense_subnetwork_count", "spike_count", "output_spike_count"}}
            for q, case in cases.items()
        },
        "comparisons": {
            f"queues_1_vs_{q}": compare(cases[1], cases[q]) for q in (2, 4)
        },
    }
    report["pass"] = all(item["pass"] for item in report["comparisons"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
