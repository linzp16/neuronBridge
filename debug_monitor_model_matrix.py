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


def connection(source: int, target: int, weight: float = 100.0) -> nb.Connection:
    return nb.Connection(
        source=[source], target=[target], weight=[weight], max_weight=[200.0],
        delay=[1], synapse_type=[0],
    )


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_model(model: str, output_root: Path) -> dict:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    params = {}
    if model == "PoissonRate":
        params = {
            "poisson_rate_bias_hz": nb.float32(40.0),
            "poisson_rate_gain_hz_per_current": nb.float32(2.0),
        }
    network.add_layer(nb.NeuronLayer(
        model, 1, monitored=True, output=True, parameters=params,
    ))
    network.connect(connection(0, 1))

    out = output_root / model
    sim = nb.Simulation(network, nb.SimulationConfig(steps=160, timestep=0.1, queues=2))
    sim.enable_debug_monitor(nb.DebugMonitorConfig(
        output_dir=out,
        sample_interval_steps=1,
        flush_interval_steps=40,
        record_spikes=True,
        record_state=True,
        record_weights=True,
        record_pending_channels=False,
        record_outer_dynamic_state=False,
        all_neurons=False,
        neuron_ids=[1],
    ))
    sim.init()
    # Use multiple run chunks to verify event-time spike buffering is not
    # dependent on run(1), while state is sampled at chunk boundaries.
    for start in (0, 20, 40, 60, 80, 100, 120, 140):
        sim.add_external_spikes([0], [start])
    sim.run(160)
    sim.flush()

    spikes = count_rows(out / "spikes.csv")
    states = count_rows(out / "neuron_state.csv")
    weights = count_rows(out / "weights.csv")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return {
        "model": model,
        "spike_rows": spikes,
        "state_rows": states,
        "weight_rows": weights,
        "spikes_bytes": (out / "spikes.csv").stat().st_size,
        "state_bytes": (out / "neuron_state.csv").stat().st_size,
        "weights_bytes": (out / "weights.csv").stat().st_size,
        "monitor_config": meta,
        "state_expected": model != "TriggerRelayNeuronModel",
        "pass": spikes > 0 and weights > 0 and (
            states > 0 if model != "TriggerRelayNeuronModel" else states == 0),
    }


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
            results.append({"model": model, "pass": False, "error": f"{type(exc).__name__}: {exc}"})
    report = {"backend": nb.backend_info(), "results": results,
              "pass": all(item["pass"] for item in results)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
