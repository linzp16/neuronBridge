from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import neuronbridge as nb


MODELS = [
    "TimeDrivenLIF_Exponential_double",
    "TimeDrivenLIF_Voltage_jump",
    "TimeDrivenIzhikevic_Exponential_Decay",
]


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


def build_network(model: str) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2))             # 0..1
    network.add_layer(nb.NeuronLayer(model, 8, monitored=True))  # 2..9
    network.add_layer(nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double", 2, monitored=True, output=True,
    ))                                                            # 10..11
    network.connect(all_to_all(0, 2, 2, 8, 20.0))
    network.connect(all_to_all(2, 8, 2, 8, 2.5, delay=2))
    network.connect(all_to_all(2, 8, 10, 2, 8.0, delay=2))
    return network


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_case(model: str, queues: int, root: Path) -> dict:
    out = root / model / f"queues_{queues}"
    sim = nb.Simulation(
        build_network(model),
        nb.SimulationConfig(steps=240, timestep=0.1, queues=queues),
    )
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
    ))
    sim.init()
    event_steps = [0, 10, 20, 40, 80, 120, 160, 200]
    sim.add_external_spikes(event_steps, [0] * len(event_steps))
    sim.add_external_spikes(event_steps, [1] * len(event_steps))
    for _ in range(12):
        sim.run(20)
    sim.flush()
    state_rows = read_csv(out / "neuron_state.csv")
    spike_rows = read_csv(out / "spikes.csv")
    output_spikes = sim.output_spikes()
    return {
        "model": model,
        "queues": queues,
        "state_rows": state_rows,
        "spike_rows": spike_rows,
        "output_spikes": output_spikes,
        "state_count": len(state_rows),
        "spike_count": len(spike_rows),
        "output_spike_count": len(output_spikes),
    }


def state_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["time_step"], row["global_neuron_id"], row["field_name"], row["component_name"]


def compare(reference: dict, candidate: dict) -> dict:
    ref_states = {state_key(row): float(row["value"]) for row in reference["state_rows"]}
    cand_states = {state_key(row): float(row["value"]) for row in candidate["state_rows"]}
    common = set(ref_states) & set(cand_states)
    missing = len(set(ref_states) ^ set(cand_states))
    max_abs = max((abs(ref_states[key] - cand_states[key]) for key in common), default=0.0)
    # Parallel OpenMP queues may append equivalent spike events in a different
    # CSV row order. Compare the canonical event set/order instead of the
    # producer-dependent write order.
    ref_spikes = sorted((row["time_step"], row["global_neuron_id"])
                        for row in reference["spike_rows"])
    cand_spikes = sorted((row["time_step"], row["global_neuron_id"])
                         for row in candidate["spike_rows"])
    return {
        "state_common_rows": len(common),
        "state_key_mismatch_count": missing,
        "state_max_abs_error": max_abs,
        "spike_rows_equal": ref_spikes == cand_spikes,
        "spike_count_difference": len(cand_spikes) - len(ref_spikes),
        "output_spike_count_difference": candidate["output_spike_count"] - reference["output_spike_count"],
        "pass": missing == 0 and max_abs <= 1.0e-5 and ref_spikes == cand_spikes and
                candidate["output_spike_count"] == reference["output_spike_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"backend": nb.backend_info(), "models": []}
    for model in MODELS:
        cases = {queues: run_case(model, queues, args.output_dir) for queues in (1, 2, 4)}
        comparisons = {f"queues_1_vs_{queues}": compare(cases[1], cases[queues]) for queues in (2, 4)}
        report["models"].append({
            "model": model,
            "counts": {str(q): {
                "state": cases[q]["state_count"],
                "spike": cases[q]["spike_count"],
                "output_spike": cases[q]["output_spike_count"],
            } for q in cases},
            "comparisons": comparisons,
            "pass": all(item["pass"] for item in comparisons.values()),
        })
    report["pass"] = all(item["pass"] for item in report["models"])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
