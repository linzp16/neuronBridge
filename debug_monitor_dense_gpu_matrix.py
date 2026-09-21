from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import neuronbridge as nb


MODELS = [
    "TimeDrivenLIF_Exponential_double",
    "TimeDrivenLIF_Exponential_Decay",
    "TimeDrivenLIF_Voltage_jump",
    "TimeDrivenLIF_Exponential_triple",
    "TimeDrivenIzhikevic_Exponential_Decay",
    "PoissonRate",
    "TriggerRelayNeuronModel",
]


def all_to_all(source_begin: int, source_count: int, target_begin: int,
               target_count: int, weight: float) -> nb.Connection:
    return nb.Connection(
        source=[source_begin + i for i in range(source_count) for _ in range(target_count)],
        target=[target_begin + j for _ in range(source_count) for j in range(target_count)],
        weight=[weight] * (source_count * target_count),
        max_weight=[200.0] * (source_count * target_count),
        delay=[1] * (source_count * target_count),
    )


def rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_model(model: str, root: Path) -> dict:
    dense_name = f"dense_{model}"
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))                 # ID 0
    network.add_layer(nb.NeuronLayer.lif_double(1))                  # ID 1
    params = {}
    if model == "PoissonRate":
        params = {
            "poisson_rate_bias_hz": nb.float32(60.0),
            "poisson_rate_gain_hz_per_current": nb.float32(2.0),
        }
    network.add_layer(nb.NeuronLayer(model, 4, parameters={
        **params, "dense_subnetwork_name": dense_name,
    }))                                                               # IDs 2..5
    network.add_layer(nb.NeuronLayer.lif_decay(1, output=True))      # ID 6
    network.connect(all_to_all(0, 1, 1, 1, 80.0))
    network.connect(all_to_all(1, 1, 2, 4, 40.0))
    network.connect(all_to_all(2, 4, 6, 1, 10.0))

    out = root / model
    sim = nb.Simulation(network, nb.SimulationConfig(steps=80, timestep=0.1, queues=2))
    sim.enable_debug_monitor(nb.DebugMonitorConfig(
        output_dir=out,
        sample_interval_steps=1,
        flush_interval_steps=20,
        record_spikes=True,
        record_state=True,
        record_weights=True,
        record_pending_channels=True,
        record_outer_dynamic_state=False,
        all_neurons=False,
        neuron_ids=[1, 6],
        dense_local_neuron_ids={dense_name: [0, 1, 2, 3]},
    ))
    sim.init()
    if sim.dense_subnetwork_count != 1:
        raise RuntimeError(f"expected one dense subnetwork, got {sim.dense_subnetwork_count}")
    sim.add_external_spikes([0, 0, 0, 0], [0, 10, 20, 30])
    for _ in range(80):
        sim.run(1)
    sim.flush()

    spike_rows = rows(out / "spikes.csv")
    state_rows = rows(out / "neuron_state.csv")
    weight_rows = rows(out / "weights.csv")
    pending_rows = rows(out / "pending_channels.csv")
    dense_spikes = [row for row in spike_rows if row["component_kind"] == "DenseSubnetwork"]
    main_spikes = [row for row in spike_rows if row["component_kind"] == "MainNetwork"]
    result = {
        "model": model,
        "dense_name": dense_name,
        "spike_rows": len(spike_rows),
        "main_spike_rows": len(main_spikes),
        "dense_spike_rows": len(dense_spikes),
        "state_rows": len(state_rows),
        "weight_rows": len(weight_rows),
        "pending_rows": len(pending_rows),
        "dense_snapshot_fired": sum(sim.dense_subnetwork_snapshot(dense_name)["fired"]),
        "state_expected": model != "TriggerRelayNeuronModel",
    }
    result["pass"] = (
        result["main_spike_rows"] > 0 and result["dense_spike_rows"] > 0 and result["weight_rows"] > 0 and
        (result["state_rows"] > 0 if result["state_expected"] else True)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model in MODELS:
        try:
            results.append(run_model(model, args.output_dir))
        except Exception as exc:
            results.append({"model": model, "pass": False,
                            "error": f"{type(exc).__name__}: {exc}"})
    report = {"backend": nb.backend_info(), "results": results,
              "pass": all(item["pass"] for item in results)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
