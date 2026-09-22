"""Train the full cerebellum against a blocking REQ/REP arm server."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import statistics
import struct
import threading
import time

import neuronbridge as nb
import zmq

from build_full_cerebellum_streaming import FullCerebellumConfig, process_memory
from train_full_cerebellum_streaming import (
    desired_trajectory,
    hand_position,
    sampled_weight_stats,
    write_report,
)


REQUEST_HEADER = struct.Struct("<II")
SPIKE_RECORD = struct.Struct("<iff")
REPLY_COUNT = struct.Struct("<I")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def uniform_bin(value: float, low: float, high: float, count: int) -> int:
    if count <= 1 or high <= low:
        return 0
    ratio = (clamp(value, low, high) - low) / (high - low)
    return min(count - 1, max(0, int(math.floor(ratio * count))))


def state_product_index(
    joint: int,
    state: tuple[float, float, float, float],
    target,
    cfg: FullCerebellumConfig,
) -> int:
    q = state[joint]
    qd = state[2 + joint]
    q_des = (target.q1, target.q2)[joint]
    qd_des = (target.qd1, target.qd2)[joint]
    angle_min = (-30.0, 0.1)[joint]
    angle_max = (90.0, 150.0)[joint]
    values = [math.degrees(q_des), math.degrees(q), math.degrees(qd_des), math.degrees(qd)]
    minimums = [angle_min, angle_min, -400.0, -400.0]
    maximums = [angle_max, angle_max, 400.0, 400.0]
    bins = [cfg.nn, cfg.nn, cfg.nv, cfg.nv]
    result = 0
    for value, minimum, maximum, count in zip(values, minimums, maximums, bins):
        result = result * count + uniform_bin(value, minimum, maximum, count)
    return result


class ReqRepArmServer:
    def __init__(self, cfg: FullCerebellumConfig, port: int):
        self.cfg = cfg
        self.port = port
        self.reference = desired_trajectory(cfg)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread = threading.Thread(target=self._serve, name="reqrep-arm-server", daemon=True)
        self.protocol_errors: list[str] = []
        self.rows: list[dict[str, float | int]] = []
        self.request_count = 0
        self.reply_spike_count = 0
        self.output_spike_count = 0
        self.request_service_seconds: list[float] = []
        self.epoch = 0
        self.last_step = -1
        self.q = [0.0, 0.0]
        self.qd = [0.0, 0.0]
        self.rng = random.Random(17)

    def start(self) -> None:
        self.thread.start()
        if not self.ready_event.wait(5.0):
            raise RuntimeError("REQ/REP arm server did not bind")

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)

    def _reset_epoch(self) -> None:
        self.epoch += 1
        initial = self.reference[1]
        self.q = [initial.q1, initial.q2]
        self.qd = [initial.qd1, initial.qd2]

    def _decode_torque(self, payload: bytes, count: int) -> tuple[float, float]:
        dcn_begin = 2 * self.cfg.n_gc + 4 * self.cfg.n_cf + 4 * self.cfg.n_pc
        counts = [0, 0, 0, 0]
        for index in range(count):
            neuron, _spike_time, _dt = SPIKE_RECORD.unpack_from(payload, index * SPIKE_RECORD.size)
            group = (neuron - dcn_begin) // self.cfg.n_dcn
            if 0 <= group < 4:
                counts[group] += 1
        return (
            self.cfg.dcn_torque_gain_1 * (counts[0] - counts[1]),
            self.cfg.dcn_torque_gain_2 * (counts[2] - counts[3]),
        )

    def _step_arm(self, torque: tuple[float, float]) -> None:
        theta2 = self.q[1]
        d1 = self.cfg.link1 * 0.5
        d2 = self.cfg.link2 * 0.5
        i1 = self.cfg.mass1 * self.cfg.link1**2 / 3.0
        i2 = self.cfg.mass2 * self.cfg.link2**2 / 12.0
        c2 = math.cos(theta2)
        s2 = math.sin(theta2)
        m11 = (
            i1
            + i2
            + 2.0 * self.cfg.mass2 * self.cfg.link1 * d2 * c2
            + self.cfg.mass1 * d1 * d1
            + self.cfg.mass2 * (d2 * d2 + self.cfg.link1**2)
        )
        m12 = i2 + self.cfg.mass2 * self.cfg.link1 * d2 * c2 + self.cfg.mass2 * d2 * d2
        m22 = i2 + self.cfg.mass2 * d2 * d2
        rhs1 = (
            torque[0]
            + 2.0 * self.cfg.mass2 * self.cfg.link1 * d2 * s2 * self.qd[0] * self.qd[1]
            + self.cfg.mass2 * self.cfg.link1 * d2 * s2 * self.qd[1] ** 2
        )
        rhs2 = torque[1] - self.cfg.mass2 * self.cfg.link1 * d2 * s2 * self.qd[0] ** 2
        determinant = m11 * m22 - m12 * m12
        if abs(determinant) < 1.0e-12:
            qdd = (0.0, 0.0)
        else:
            qdd = ((rhs1 * m22 - rhs2 * m12) / determinant, (m11 * rhs2 - m12 * rhs1) / determinant)
        dt_s = self.cfg.control_interval_ms * 0.001
        old_qd = tuple(self.qd)
        self.qd[0] += qdd[0] * dt_s
        self.qd[1] += qdd[1] * dt_s
        self.q[0] = clamp(self.q[0] + old_qd[0] * dt_s, math.radians(-30.0), math.radians(90.0))
        self.q[1] = clamp(self.q[1] + old_qd[1] * dt_s, math.radians(0.1), math.radians(150.0))

    def _feedback_spikes(self, step: int, phase: int) -> list[tuple[int, int]]:
        target = self.reference[phase]
        state = (self.q[0], self.q[1], self.qd[0], self.qd[1])
        base_step = step + 1
        repeat_period = max(1, int(round(4.0 / self.cfg.dt_ms)))
        spikes: list[tuple[int, int]] = []
        for repeat in range(self.cfg.gc_repeats_per_control):
            event_step = base_step + repeat * repeat_period
            spikes.append((state_product_index(0, state, target, self.cfg), event_step))
            spikes.append((self.cfg.n_gc + state_product_index(1, state, target, self.cfg), event_step))

        cf_starts = [
            2 * self.cfg.n_gc,
            2 * self.cfg.n_gc + self.cfg.n_cf,
            2 * self.cfg.n_gc + 2 * self.cfg.n_cf,
            2 * self.cfg.n_gc + 3 * self.cfg.n_cf,
        ]
        for joint, (positive_start, negative_start) in enumerate(
            ((cf_starts[0], cf_starts[1]), (cf_starts[2], cf_starts[3]))
        ):
            target_q = (target.q1, target.q2)[joint]
            target_qd = (target.qd1, target.qd2)[joint]
            position_error = math.degrees(target_q - self.q[joint])
            velocity_error = math.degrees(target_qd - self.qd[joint])
            angle_norm = clamp(position_error / (60.0, 75.0)[joint], -1.0, 1.0)
            velocity_norm = clamp(velocity_error / 400.0, -1.0, 1.0)
            mixed = self.cfg.cf_mix_position * angle_norm + (1.0 - self.cfg.cf_mix_position) * velocity_norm
            count = int(round(abs(self.cfg.spike_cf_max * self.cfg.n_cf / self.cfg.sample_count * mixed)))
            population_start = positive_start if mixed > 0.0 else negative_start
            seen: set[tuple[int, int]] = set()
            for _ in range(count):
                neuron = population_start + self.rng.randrange(self.cfg.n_cf)
                local_step = self.rng.randrange(self.cfg.steps_per_control)
                key = (neuron, local_step)
                if key not in seen:
                    seen.add(key)
                    spikes.append((neuron, base_step + local_step))
        return spikes

    def _handle_request(self, request: bytes) -> tuple[bytes, bytes | None]:
        if len(request) < REQUEST_HEADER.size:
            raise ValueError("truncated REQ/REP spike request")
        step, count = REQUEST_HEADER.unpack_from(request)
        expected = REQUEST_HEADER.size + count * SPIKE_RECORD.size
        if len(request) != expected:
            raise ValueError(f"REQ/REP request size mismatch: expected {expected}, got {len(request)}")
        # CommunicationEvent is first scheduled at communication_interval, not
        # at zero.  A reset is therefore observed as the step counter wrapping
        # from the previous epoch's final boundary back to the first boundary.
        if self.request_count == 0 or step <= self.last_step:
            self._reset_epoch()
        self.last_step = step
        payload = request[REQUEST_HEADER.size:]
        self.output_spike_count += count
        phase = 1 if step == 0 else min(self.cfg.sample_count - 1, step // self.cfg.steps_per_control + 1)
        if step > 0:
            torque = self._decode_torque(payload, count)
            self._step_arm(torque)
            target = self.reference[phase]
            x, y = hand_position(self.q[0], self.q[1], self.cfg)
            self.rows.append(
                {
                    "epoch": self.epoch,
                    "sample": phase,
                    "simulation_step": int(step),
                    "x": x,
                    "y": y,
                    "desired_x": target.x,
                    "desired_y": target.y,
                    "q1": self.q[0],
                    "q2": self.q[1],
                    "error_m": math.hypot(x - target.x, y - target.y),
                    "output_spikes": count,
                }
            )
        feedback = self._feedback_spikes(step, phase)
        self.reply_spike_count += len(feedback)
        count_frame = REPLY_COUNT.pack(len(feedback))
        if not feedback:
            return count_frame, None
        spike_frame = b"".join(
            SPIKE_RECORD.pack(neuron, event_step * self.cfg.dt_ms, self.cfg.dt_ms)
            for neuron, event_step in feedback
        )
        return count_frame, spike_frame

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.linger = 0
        socket.bind(f"tcp://127.0.0.1:{self.port}")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self.ready_event.set()
        try:
            while not self.stop_event.is_set():
                events = dict(poller.poll(50))
                if socket not in events:
                    continue
                request = socket.recv()
                started = time.perf_counter()
                try:
                    count_frame, spike_frame = self._handle_request(request)
                except Exception as exc:
                    self.protocol_errors.append(repr(exc))
                    count_frame, spike_frame = REPLY_COUNT.pack(0), None
                if spike_frame is None:
                    socket.send(count_frame)
                else:
                    socket.send_multipart([count_frame, spike_frame])
                self.request_count += 1
                self.request_service_seconds.append(time.perf_counter() - started)
        finally:
            socket.close(0)
            context.term()


def train(args: argparse.Namespace) -> dict[str, object]:
    cfg = FullCerebellumConfig()
    info = nb.nbnet.inspect(args.network, verify_checksum=False)
    expected_neurons = 2 * cfg.n_gc + 4 * cfg.n_cf + 4 * cfg.n_pc + 4 * cfg.n_dcn
    expected_connections = 4 * cfg.n_gc * (cfg.n_pc + cfg.n_dcn) + 4 * cfg.n_cf + 4 * cfg.n_pc
    if info.neuron_count != expected_neurons or info.connection_count != expected_connections:
        raise RuntimeError("REQ/REP network dimensions do not match the full cerebellum")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "training_report.json"
    trajectory_path = args.output_dir / "trajectory.jsonl"
    weights_path = args.output_dir / "weights_final.bin"
    report: dict[str, object] = {
        "status": "constructing",
        "mode": "external_arm_reqrep",
        "network": str(args.network.resolve()),
        "configuration": asdict(cfg),
        "epochs_requested": args.epochs,
        "windows_per_epoch": cfg.sample_count - 2,
        "steps_per_window": cfg.steps_per_control,
        "total_training_steps": args.epochs * (cfg.sample_count - 2) * cfg.steps_per_control,
        "neuron_count": info.neuron_count,
        "connection_count": info.connection_count,
        "plastic_connection_count": 4 * cfg.n_gc * cfg.n_pc,
        "epochs": [],
        "started_unix": time.time(),
    }
    write_report(report_path, report)

    server = ReqRepArmServer(cfg, args.port)
    server.start()
    try:
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
        dcn_begin = 2 * cfg.n_gc + 4 * cfg.n_cf + 4 * cfg.n_pc
        dcn_state = sim.neuron_state(dcn_begin)
        if not bool(dcn_state["is_output"]):
            raise RuntimeError(
                "REQ/REP control requires DCN layers with output=True; "
                "rebuild the external-arm .nbnet with the corrected layer metadata"
            )
        sim.add_zmq_input_output_spike_driver(
            server_address="127.0.0.1",
            server_port=args.port,
            communication_interval=cfg.steps_per_control,
        )
        sim.init()
        initial_weights = sampled_weight_stats(sim, int(report["plastic_connection_count"]), args.weight_samples)
        report["initial_sampled_weight_stats"] = initial_weights
        report["status"] = "training"
        write_report(report_path, report)

        epochs: list[dict[str, object]] = report["epochs"]  # type: ignore[assignment]
        training_started = time.perf_counter()
        for epoch in range(args.epochs):
            epoch_started = time.perf_counter()
            if epoch > 0:
                sim.reset()
            sim.run((cfg.sample_count - 2) * cfg.steps_per_control)
            rows = [row for row in server.rows if int(row["epoch"]) == epoch + 1]
            errors = [float(row["error_m"]) for row in rows]
            expected_requests_so_far = (epoch + 1) * (cfg.sample_count - 2)
            if len(errors) != cfg.sample_count - 2 or server.request_count != expected_requests_so_far:
                raise RuntimeError(
                    f"REQ/REP epoch {epoch + 1} incomplete: windows={len(errors)}, "
                    f"requests={server.request_count}, expected_requests={expected_requests_so_far}, "
                    f"protocol_errors={server.protocol_errors}"
                )
            epoch_report: dict[str, object] = {
                "epoch": epoch + 1,
                "windows": len(errors),
                "mean_error_m": statistics.fmean(errors) if errors else None,
                "rmse_m": math.sqrt(statistics.fmean(value * value for value in errors)) if errors else None,
                "minimum_error_m": min(errors) if errors else None,
                "maximum_error_m": max(errors) if errors else None,
                "final_error_m": errors[-1] if errors else None,
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
            write_report(report_path, report)
            print(
                f"epoch={epoch + 1}/{args.epochs} windows={len(errors)} requests={server.request_count} "
                f"mean_error_m={epoch_report['mean_error_m']} "
                f"elapsed_s={epoch_report['elapsed_seconds']:.3f}",
                flush=True,
            )

        save_started = time.perf_counter()
        sim.save_weights(weights_path)
        report["weight_save_seconds"] = time.perf_counter() - save_started
        report["weights"] = str(weights_path.resolve())
        report["final_sampled_weight_stats"] = sampled_weight_stats(
            sim,
            int(report["plastic_connection_count"]),
            args.weight_samples,
        )
    finally:
        server.close()

    with trajectory_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in server.rows:
            handle.write(json.dumps(row) + "\n")
    report["trajectory"] = str(trajectory_path.resolve())
    report["server"] = {
        "request_count": server.request_count,
        "output_spike_count": server.output_spike_count,
        "reply_spike_count": server.reply_spike_count,
        "protocol_errors": server.protocol_errors,
        "mean_service_seconds": statistics.fmean(server.request_service_seconds),
        "maximum_service_seconds": max(server.request_service_seconds),
    }
    epochs = report["epochs"]  # type: ignore[assignment]
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
    expected_requests = args.epochs * (cfg.sample_count - 2)
    report["pass"] = (
        len(epochs) == args.epochs
        and all(int(item["windows"]) == cfg.sample_count - 2 for item in epochs)
        and server.request_count == expected_requests
        and server.output_spike_count > 0
        and report["final_sampled_weight_stats"] != report["initial_sampled_weight_stats"]
        and not server.protocol_errors
    )
    report["expected_request_count"] = expected_requests
    report["training_elapsed_seconds"] = time.perf_counter() - training_started
    report["memory_after_training"] = process_memory()
    report["completed_unix"] = time.time()
    report["status"] = "complete"
    write_report(report_path, report)
    print(json.dumps({"pass": report["pass"], "server": report["server"], "convergence": report["convergence"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("artifacts/full_cerebellum_reqrep_network/full_cerebellum.nbnet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full_cerebellum_reqrep_training_100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--port", type=int, default=5645)
    parser.add_argument("--memory-budget-mb", type=int, default=64)
    parser.add_argument("--queues", type=int, default=1)
    parser.add_argument("--event-queue", choices=("heap", "timing_wheel"), default="timing_wheel")
    parser.add_argument("--timing-wheel-size", type=int, default=256)
    parser.add_argument("--weight-samples", type=int, default=256)
    args = parser.parse_args()
    result = train(args)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
