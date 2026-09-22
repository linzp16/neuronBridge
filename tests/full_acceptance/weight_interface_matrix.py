"""Installed-wheel acceptance matrix for weight read, mutation, reset, and snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import neuronbridge as nb


INITIAL_WEIGHTS = [1.0, 2.0, 3.0, 4.0, 5.0]
MUTABLE_INDICES = (0, 2)
INITIAL_ONLY_INDICES = (1, 3, 4)


def output_layer() -> nb.NeuronLayer:
    return nb.NeuronLayer.lif_decay(
        1,
        output=True,
        monitored=True,
        V_rest=nb.float32(-60.0),
        V_reset=nb.float32(-60.0),
        V_th=nb.float32(-50.0),
        tau=nb.float32(20.0),
        R=nb.float32(1.0),
        t_ref=nb.int32(1),
        gexc_tau=nb.float32(5.0),
        ginh_tau=nb.float32(10.0),
        Eexc=nb.float32(0.0),
        Einh=nb.float32(-80.0),
    )


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))                 # 0 main
    network.add_layer(nb.NeuronLayer.lif_double(2, dense_name="a")) # 1..2 dense a
    network.add_layer(nb.NeuronLayer.lif_double(1, dense_name="b")) # 3 dense b
    network.add_layer(output_layer())                                # 4 main
    for connection in (
        nb.Connection(0, 4, weight=1.0, max_weight=10.0, delay=1),  # main -> main, mutable
        nb.Connection(0, 1, weight=2.0, max_weight=10.0, delay=1),  # main -> dense, initial-only
        nb.Connection(1, 2, weight=3.0, max_weight=10.0, delay=1),  # dense internal, mutable
        nb.Connection(2, 3, weight=4.0, max_weight=10.0, delay=1),  # dense -> dense, initial-only
        nb.Connection(3, 4, weight=5.0, max_weight=10.0, delay=1),  # dense -> main, initial-only
    ):
        network.connect(connection)
    return network


def weights(sim: nb.Simulation) -> list[float]:
    return [sim.get_connection_weight(index) for index in range(len(INITIAL_WEIGHTS))]


def assert_weights(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual, expected):
        assert abs(lhs - rhs) <= 1.0e-6, (actual, expected)


def expect_runtime_error(function, contains: str) -> str:
    try:
        function()
    except RuntimeError as exc:
        message = str(exc)
        assert contains in message, message
        return message
    raise AssertionError("expected RuntimeError")


def make_sim(source, config: nb.SimulationConfig) -> nb.Simulation:
    return nb.Simulation(source, config).init()


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    network = build_network()
    nbnet_path = output_dir / "weight_matrix.nbnet"
    nb.nbnet.from_network(network, nbnet_path, options=nb.NbnetWriteOptions(overwrite=True))
    config = nb.SimulationConfig(steps=32, timestep=1.0, event_queue="timing_wheel", timing_wheel_size=64)
    simulations = {
        "fast": make_sim(network, config),
        "streaming": make_sim(
            nbnet_path,
            config,
        ),
    }
    report: dict = {"cases": {}, "pass": False}

    for mode, sim in simulations.items():
        initial = weights(sim)
        assert_weights(initial, INITIAL_WEIGHTS)

        sim.set_connection_weight(0, 1.5)
        sim.set_connection_weight(2, 3.5)
        changed = weights(sim)
        assert_weights(changed, [1.5, 2.0, 3.5, 4.0, 5.0])
        dense_weights = sim.dense_subnetwork_weights(sim.find_dense_subnetwork("a"))
        assert any(abs(value - 3.5) <= 1.0e-6 for value in dense_weights)

        immutable_errors = {
            str(index): expect_runtime_error(
                lambda index=index: sim.set_connection_weight(index, 9.0),
                "connection has no mutable runtime weight",
            )
            for index in INITIAL_ONLY_INDICES
        }
        range_errors = {
            "get_negative": expect_runtime_error(
                lambda: sim.get_connection_weight(-1), "out of range"
            ),
            "get_past_end": expect_runtime_error(
                lambda: sim.get_connection_weight(len(INITIAL_WEIGHTS)), "out of range"
            ),
            "set_negative": expect_runtime_error(
                lambda: sim.set_connection_weight(-1, 1.0), "out of range"
            ),
            "set_past_end": expect_runtime_error(
                lambda: sim.set_connection_weight(len(INITIAL_WEIGHTS), 1.0), "out of range"
            ),
        }

        snapshot = output_dir / f"{mode}_weights.txt"
        sim.save_weights(snapshot)
        assert snapshot.exists() and snapshot.stat().st_size > 0
        assert snapshot.read_text(encoding="utf-8").startswith("NPGR_WEIGHT_SNAPSHOT 1\n")
        sim.set_connection_weight(0, 7.25)
        sim.set_connection_weight(2, 8.5)
        sim.load_weights(snapshot)
        restored = weights(sim)
        assert_weights(restored, changed)

        sim.set_connection_weight(0, 2.25)
        sim.set_connection_weight(2, 4.25)
        sim.reset()
        preserved = weights(sim)
        assert_weights(preserved, [2.25, 2.0, 4.25, 4.0, 5.0])

        report["cases"][mode] = {
            "build_mode": sim.build_stats["mode"],
            "initial": initial,
            "changed": changed,
            "restored": restored,
            "weights_after_reset": preserved,
            "reset_preserves_weights": True,
            "dense_weights_after_change": dense_weights,
            "immutable_route_errors": immutable_errors,
            "range_errors": range_errors,
            "snapshot": str(snapshot.resolve()),
            "snapshot_bytes": snapshot.stat().st_size,
            "pass": True,
        }

    # A snapshot written by one construction path must load into an equivalent
    # simulation created by the other path.
    cross_snapshot = output_dir / "cross_mode_weights.txt"
    simulations["fast"].set_connection_weight(0, 6.25)
    simulations["fast"].set_connection_weight(2, 7.5)
    simulations["fast"].save_weights(cross_snapshot)
    simulations["streaming"].load_weights(cross_snapshot)
    cross_loaded = weights(simulations["streaming"])
    assert_weights(cross_loaded, [6.25, 2.0, 7.5, 4.0, 5.0])
    report["cross_mode"] = {
        "source": "fast",
        "target": "streaming",
        "weights": cross_loaded,
        "pass": True,
    }

    validation_sim = make_sim(network, config)
    special_values = {}
    for name, value in (
        ("nan", float("nan")),
        ("positive_infinity", float("inf")),
        ("negative_infinity", float("-inf")),
        ("above_declared_max", 11.0),
        ("negative", -1.0),
    ):
        try:
            validation_sim.set_connection_weight(0, value)
            special_values[name] = {
                "accepted": True,
                "readback": repr(validation_sim.get_connection_weight(0)),
            }
        except RuntimeError as exc:
            special_values[name] = {"accepted": False, "error": str(exc)}

    bad_save_path = output_dir / "missing_parent" / "weights.txt"
    bad_load_path = output_dir / "missing_weight_file.txt"
    io_failures = {}
    try:
        validation_sim.save_weights(bad_save_path)
        io_failures["save"] = {"raised": False, "file_exists": bad_save_path.exists()}
    except RuntimeError as exc:
        io_failures["save"] = {"raised": True, "error": str(exc)}
    try:
        validation_sim.load_weights(bad_load_path)
        io_failures["load"] = {"raised": False}
    except RuntimeError as exc:
        io_failures["load"] = {"raised": True, "error": str(exc)}

    report["input_validation"] = {
        "special_values": special_values,
        "nonfinite_rejected": all(
            not special_values[name]["accepted"]
            for name in ("nan", "positive_infinity", "negative_infinity")
        ),
        "bounds_policy_observation": "negative and above-max values are currently accepted",
    }
    report["io_error_reporting"] = {
        "cases": io_failures,
        "pass": io_failures["save"]["raised"] and io_failures["load"]["raised"],
    }
    report["pass"] = (
        all(case["pass"] for case in report["cases"].values())
        and report["input_validation"]["nonfinite_rejected"]
        and report["io_error_reporting"]["pass"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.output_dir)
    except Exception as exc:
        report = {
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    path = args.output_dir / "weight_interface_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
