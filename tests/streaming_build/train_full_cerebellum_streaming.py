"""Train the full streamed cerebellum with its built-in 2-DOF arm plant."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import time

import neuronbridge as nb

from build_full_cerebellum_streaming import FullCerebellumConfig, process_memory


@dataclass(frozen=True)
class StateSample:
    q1: float
    q2: float
    qd1: float
    qd2: float
    x: float
    y: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def hand_position(q1: float, q2: float, cfg: FullCerebellumConfig) -> tuple[float, float]:
    return (
        cfg.link1 * math.cos(q1) + cfg.link2 * math.cos(q1 + q2),
        cfg.link1 * math.sin(q1) + cfg.link2 * math.sin(q1 + q2),
    )


def desired_trajectory(cfg: FullCerebellumConfig) -> list[StateSample]:
    positions: list[tuple[float, float, float, float]] = []
    total_ms = cfg.control_interval_ms * cfg.sample_count
    for sample in range(cfg.sample_count + 1):
        phase = sample * cfg.control_interval_ms / total_ms
        x = 0.15 * math.cos(2.0 * math.pi * phase) + 0.1
        y = 0.15 * math.sin(2.0 * math.pi * phase) + 0.4
        radius = math.hypot(x, y)
        c2 = clamp(
            (x * x + y * y - cfg.link1 * cfg.link1 - cfg.link2 * cfg.link2)
            / (2.0 * cfg.link1 * cfg.link2),
            -1.0,
            1.0,
        )
        q2 = math.acos(c2)
        phi = math.acos(
            clamp(
                (cfg.link2 * cfg.link2 - cfg.link1 * cfg.link1 - x * x - y * y)
                / (-2.0 * cfg.link1 * radius),
                -1.0,
                1.0,
            )
        )
        q1 = math.atan2(y, x) - phi
        positions.append((q1, q2, x, y))

    dt_s = cfg.control_interval_ms * 0.001
    velocities = [
        (
            (positions[index + 1][0] - positions[index][0]) / dt_s,
            (positions[index + 1][1] - positions[index][1]) / dt_s,
        )
        for index in range(cfg.sample_count)
    ]
    velocities.append(velocities[-1])
    return [
        StateSample(q1=q1, q2=q2, qd1=qd1, qd2=qd2, x=x, y=y)
        for (q1, q2, x, y), (qd1, qd2) in zip(positions, velocities)
    ]


def sampled_weight_stats(sim: nb.Simulation, plastic_connection_count: int, sample_count: int) -> dict[str, float | int]:
    count = min(max(2, sample_count), plastic_connection_count)
    indices = [index * (plastic_connection_count - 1) // (count - 1) for index in range(count)]
    weights = [sim.get_connection_weight(index) for index in indices]
    return {
        "sample_count": count,
        "mean": statistics.fmean(weights),
        "minimum": min(weights),
        "maximum": max(weights),
        "stdev": statistics.pstdev(weights),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def train(args: argparse.Namespace) -> dict[str, object]:
    cfg = FullCerebellumConfig()
    info = nb.nbnet.inspect(args.network, verify_checksum=False)
    expected_neurons = 2 * cfg.n_gc + 4 * cfg.n_cf + 4 * cfg.n_pc + 4 * cfg.n_dcn
    expected_connections = 4 * cfg.n_gc * (cfg.n_pc + cfg.n_dcn) + 4 * cfg.n_cf + 4 * cfg.n_pc
    if info.neuron_count != expected_neurons or info.connection_count != expected_connections:
        raise RuntimeError(
            f"network dimensions do not match full cerebellum: "
            f"{info.neuron_count} neurons, {info.connection_count} connections"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "training_report.json"
    trajectory_path = args.output_dir / "trajectory.jsonl"
    weights_path = args.output_dir / "weights_final.bin"
    reference = desired_trajectory(cfg)
    report: dict[str, object] = {
        "status": "constructing",
        "network": str(args.network.resolve()),
        "configuration": asdict(cfg),
        "epochs_requested": args.epochs,
        "windows_per_epoch": cfg.sample_count - 2,
        "steps_per_window": cfg.steps_per_control,
        "steps_per_epoch": (cfg.sample_count - 2) * cfg.steps_per_control,
        "total_training_steps": args.epochs * (cfg.sample_count - 2) * cfg.steps_per_control,
        "neuron_count": info.neuron_count,
        "connection_count": info.connection_count,
        "plastic_connection_count": 4 * cfg.n_gc * cfg.n_pc,
        "epochs": [],
        "started_unix": time.time(),
    }
    write_report(report_path, report)

    construct_started = time.perf_counter()
    sim = nb.Simulation(
        args.network,
        nb.SimulationConfig(
            steps=cfg.total_steps,
            timestep=cfg.dt_ms,
            queues=args.queues,
            event_queue=args.event_queue,
            timing_wheel_size=args.timing_wheel_size,
        ),
        build_options=nb.StreamingBuildOptions(
            memory_budget_mb=args.memory_budget_mb,
            mmap=True,
            verify_checksum=True,
        ),
    )
    report["construct_seconds"] = time.perf_counter() - construct_started
    report["build_stats"] = sim.build_stats
    report["memory_after_construct"] = process_memory()
    report["status"] = "training"
    sim.init()
    initial_weight_stats = sampled_weight_stats(sim, int(report["plastic_connection_count"]), args.weight_samples)
    report["initial_sampled_weight_stats"] = initial_weight_stats
    write_report(report_path, report)

    epochs: list[dict[str, object]] = report["epochs"]  # type: ignore[assignment]
    training_started = time.perf_counter()
    with trajectory_path.open("w", encoding="utf-8", newline="\n", buffering=1) as trajectory:
        for epoch in range(args.epochs):
            epoch_started = time.perf_counter()
            if epoch > 0:
                sim.reset()
            initial = reference[1]
            sim.reset_outer_dynamic_state(
                "cerebellum_arm",
                [initial.q1, initial.q2],
                [initial.qd1, initial.qd2],
            )
            sim.set_outer_dynamic_desired_state(
                "cerebellum_arm",
                [initial.q1, initial.q2],
                [initial.qd1, initial.qd2],
            )
            errors: list[float] = []
            for sample in range(2, cfg.sample_count):
                target = reference[sample]
                sim.set_outer_dynamic_desired_state(
                    "cerebellum_arm",
                    [target.q1, target.q2],
                    [target.qd1, target.qd2],
                )
                sim.run(cfg.steps_per_control)
                state = sim.outer_dynamic_state()
                x, y = hand_position(float(state["q"][0]), float(state["q"][1]), cfg)
                error = math.hypot(x - target.x, y - target.y)
                errors.append(error)
                trajectory.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "sample": sample,
                            "simulation_step": int(state["time_step"]),
                            "x": x,
                            "y": y,
                            "desired_x": target.x,
                            "desired_y": target.y,
                            "q1": float(state["q"][0]),
                            "q2": float(state["q"][1]),
                            "error_m": error,
                        }
                    )
                    + "\n"
                )
            epoch_report: dict[str, object] = {
                "epoch": epoch + 1,
                "windows": len(errors),
                "mean_error_m": statistics.fmean(errors),
                "rmse_m": math.sqrt(statistics.fmean(value * value for value in errors)),
                "minimum_error_m": min(errors),
                "maximum_error_m": max(errors),
                "final_error_m": errors[-1],
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "sampled_weight_stats": sampled_weight_stats(
                    sim,
                    int(report["plastic_connection_count"]),
                    args.weight_samples,
                ),
            }
            epochs.append(epoch_report)
            report["epochs_completed"] = epoch + 1
            report["training_elapsed_seconds"] = time.perf_counter() - training_started
            report["memory_during_training"] = process_memory()
            write_report(report_path, report)
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"mean_error_m={epoch_report['mean_error_m']:.9f} "
                f"rmse_m={epoch_report['rmse_m']:.9f} "
                f"elapsed_s={epoch_report['elapsed_seconds']:.3f}",
                flush=True,
            )

    save_started = time.perf_counter()
    sim.save_weights(weights_path)
    report["weight_save_seconds"] = time.perf_counter() - save_started
    report["weights"] = str(weights_path.resolve())
    report["trajectory"] = str(trajectory_path.resolve())
    report["final_sampled_weight_stats"] = sampled_weight_stats(
        sim,
        int(report["plastic_connection_count"]),
        args.weight_samples,
    )
    means = [float(item["mean_error_m"]) for item in epochs]
    x_mean = (len(means) - 1) / 2.0
    y_mean = statistics.fmean(means)
    denominator = sum((index - x_mean) ** 2 for index in range(len(means)))
    slope = (
        sum((index - x_mean) * (value - y_mean) for index, value in enumerate(means)) / denominator
        if denominator
        else 0.0
    )
    report["convergence"] = {
        "first_epoch_mean_error_m": means[0],
        "last_epoch_mean_error_m": means[-1],
        "last_to_first_ratio": means[-1] / means[0] if means[0] else None,
        "linear_trend_m_per_epoch": slope,
        "decreased_from_first_to_last": means[-1] < means[0],
        "best_epoch": min(range(len(means)), key=means.__getitem__) + 1,
        "best_mean_error_m": min(means),
    }
    report["training_elapsed_seconds"] = time.perf_counter() - training_started
    report["completed_unix"] = time.time()
    report["status"] = "complete"
    report["pass"] = len(epochs) == args.epochs and all(int(item["windows"]) == cfg.sample_count - 2 for item in epochs)
    write_report(report_path, report)
    print(json.dumps({"pass": report["pass"], "convergence": report["convergence"]}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("artifacts/full_cerebellum_streaming/full_cerebellum.nbnet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full_cerebellum_training_100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--memory-budget-mb", type=int, default=64)
    parser.add_argument("--queues", type=int, default=1)
    parser.add_argument("--event-queue", choices=("heap", "timing_wheel"), default="timing_wheel")
    parser.add_argument("--timing-wheel-size", type=int, default=256)
    parser.add_argument("--weight-samples", type=int, default=256)
    args = parser.parse_args()
    if args.epochs <= 0 or args.weight_samples <= 1:
        parser.error("epochs must be positive and weight-samples must exceed one")
    result = train(args)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
