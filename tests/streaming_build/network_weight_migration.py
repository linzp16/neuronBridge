"""End-to-end migration test for a streamed network and learned weights.

The producer creates an ``.nbnet`` file incrementally, trains it through the
direct streaming path, saves a weight snapshot, and copies both artifacts to a
separate directory.  A fresh child process then loads only the copied files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import neuronbridge as nb


INITIAL_WEIGHT = 8.0
CONNECTION_COUNT = 2
TRAIN_STEPS = [1, 4, 8, 11, 15, 18]
TRAIN_NEURONS = [0, 1, 0, 1, 0, 1]
PROBE_STEPS = [1, 4, 8, 11, 15, 18]
PROBE_NEURONS = [0, 1, 0, 1, 0, 1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def lif_layer() -> nb.NeuronLayer:
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double",
        1,
        output=True,
        monitored=True,
        parameters={
            "V_rest": nb.float32(-60.0),
            "V_reset": nb.float32(-60.0),
            "V_th": nb.float32(-58.0),
            "tau": nb.float32(20.0),
            "R": nb.float32(1.0),
            "t_ref": nb.int32(1),
            "gexc_tau": nb.float32(5.0),
            "ginh_tau": nb.float32(10.0),
            "Eexc": nb.float32(0.0),
            "Einh": nb.float32(-80.0),
        },
    )


def create_network_file(path: Path) -> nb.NbnetBuildResult:
    with nb.NbnetDescriptionBuilder(
        path,
        options=nb.NbnetWriteOptions(overwrite=True),
    ) as builder:
        inputs = builder.add_layer(nb.NeuronLayer.input_spike(2))
        output = builder.add_layer(lif_layer())
        rule = builder.add_learning_rule(
            nb.LearningRule(
                "AdditiveKernalChange",
                parameters={
                    "a1pre": nb.float32(0.2),
                    "a2prepre": nb.float32(-0.05),
                    "LTP_tau": nb.float32(16.8),
                },
            )
        )
        builder.append_connections(
            [inputs.first_neuron_id],
            [output.first_neuron_id],
            weight=INITIAL_WEIGHT,
            max_weight=20.0,
            delay=1,
            synapse_rule=rule,
        )
        builder.append_connections(
            [inputs.first_neuron_id + 1],
            [output.first_neuron_id],
            weight=0.0,
            max_weight=0.0,
            delay=1,
            trigger_rule=rule,
        )
        return builder.finalize()


def config() -> nb.SimulationConfig:
    return nb.SimulationConfig(
        steps=24,
        timestep=1.0,
        event_queue="timing_wheel",
        timing_wheel_size=64,
    )


def load_streaming(network_path: Path) -> nb.Simulation:
    simulation = nb.Simulation(
        network_path,
        config(),
        build_options=nb.StreamingBuildOptions(
            memory_budget_mb=8,
            mmap=True,
            verify_checksum=True,
        ),
    )
    stats = simulation.build_stats
    assert stats["mode"] == "streaming_file_cpp", stats
    assert stats["runtime_build_path"] == "streaming_direct", stats
    assert stats["connection_count"] == CONNECTION_COUNT, stats
    return simulation.init()


def weights(simulation: nb.Simulation) -> list[float]:
    return [simulation.get_connection_weight(index) for index in range(CONNECTION_COUNT)]


def run_probe(network_path: Path, weight_path: Path) -> dict[str, object]:
    simulation = load_streaming(network_path)
    initial = weights(simulation)
    simulation.load_weights(weight_path)
    loaded = weights(simulation)
    simulation.add_external_spikes(PROBE_STEPS, PROBE_NEURONS).run()
    return {
        "initial_weights_before_snapshot": initial,
        "loaded_weights": loaded,
        "weights_after_probe": weights(simulation),
        "output_spikes": simulation.output_spikes(),
        "build_stats": simulation.build_stats,
    }


def consumer(network_path: Path, weight_path: Path, result_path: Path) -> None:
    validation = nb.nbnet.validate(network_path, verify_checksum=True)
    assert validation.valid, validation.errors
    result = run_probe(network_path, weight_path)
    result.update(
        {
            "network_path": str(network_path.resolve()),
            "weight_path": str(weight_path.resolve()),
            "network_sha256": sha256(network_path),
            "weight_sha256": sha256(weight_path),
            "process_id": __import__("os").getpid(),
        }
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def producer(output_dir: Path) -> dict[str, object]:
    source_dir = output_dir / "source"
    destination_dir = output_dir / "migrated_destination"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination_dir.mkdir(parents=True, exist_ok=True)

    source_network = source_dir / "trained_network.nbnet"
    source_weights = source_dir / "trained_weights.txt"
    build_result = create_network_file(source_network)
    validation = nb.nbnet.validate(source_network, verify_checksum=True)
    assert validation.valid, validation.errors

    training = load_streaming(source_network)
    before = weights(training)
    training.add_external_spikes(TRAIN_STEPS, TRAIN_NEURONS).run()
    after = weights(training)
    assert abs(after[0] - before[0]) > 1.0e-6, (before, after)
    training.save_weights(source_weights)
    assert source_weights.read_text(encoding="utf-8").startswith("NPGR_WEIGHT_SNAPSHOT 1\n")

    source_probe = run_probe(source_network, source_weights)
    migrated_network = destination_dir / source_network.name
    migrated_weights = destination_dir / source_weights.name
    shutil.copy2(source_network, migrated_network)
    shutil.copy2(source_weights, migrated_weights)

    assert sha256(source_network) == sha256(migrated_network)
    assert sha256(source_weights) == sha256(migrated_weights)

    consumer_result_path = destination_dir / "consumer_result.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--consumer",
            "--network",
            str(migrated_network),
            "--weights",
            str(migrated_weights),
            "--result",
            str(consumer_result_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    target_probe = json.loads(consumer_result_path.read_text(encoding="utf-8"))

    assert target_probe["loaded_weights"] == source_probe["loaded_weights"]
    assert target_probe["weights_after_probe"] == source_probe["weights_after_probe"]
    assert target_probe["output_spikes"] == source_probe["output_spikes"]

    report: dict[str, object] = {
        "pass": True,
        "python": sys.executable,
        "backend": nb.backend_info(),
        "network_generation": {
            "path": str(source_network.resolve()),
            "file_bytes": build_result.file_bytes,
            "neuron_count": build_result.neuron_count,
            "connection_count": build_result.connection_count,
            "nbnet_content_sha256": build_result.sha256,
            "file_sha256": sha256(source_network),
        },
        "training": {
            "weights_before": before,
            "weights_after": after,
            "plastic_weight_changed": True,
            "training_output_spikes": training.output_spikes(),
        },
        "weight_snapshot": {
            "path": str(source_weights.resolve()),
            "file_bytes": source_weights.stat().st_size,
            "file_sha256": sha256(source_weights),
        },
        "migration": {
            "destination": str(destination_dir.resolve()),
            "network_sha256_preserved": True,
            "weight_sha256_preserved": True,
            "fresh_process_exit_code": completed.returncode,
            "fresh_process_stderr": completed.stderr,
        },
        "source_probe": source_probe,
        "destination_probe": target_probe,
        "equivalence": {
            "loaded_weights_equal": True,
            "post_probe_weights_equal": True,
            "output_spikes_equal": True,
        },
    }
    report_path = output_dir / "migration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--consumer", action="store_true")
    parser.add_argument("--network", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()

    if args.consumer:
        if args.network is None or args.weights is None or args.result is None:
            parser.error("consumer mode requires --network, --weights, and --result")
        consumer(args.network, args.weights, args.result)
        return

    output_dir = args.output_dir or Path("artifacts") / "network_migration_test" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = producer(output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
