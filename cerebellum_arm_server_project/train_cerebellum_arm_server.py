from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import neuronbridge as nb
import zmq


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "python"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from cerebellum_framework import (  # noqa: E402
    CerebellumConfig,
    LayerIndex,
    add_connections,
    add_layers,
    add_learning_rule,
    hand_position,
)


MAGIC = 0x53504B31
VERSION = 1
HEADER = struct.Struct("<IIII")
SPIKE = struct.Struct("<iff")
OUTPUT_TOPIC = b"/snn/output_spikes"
INPUT_TOPIC = b"/robot/input_spikes"


@dataclass(frozen=True)
class ProjectConfig:
    epochs: int = 3
    phases: int = 8
    dt_ms: float = 0.1
    control_interval_ms: float = 20.0
    queues: int = 2
    publish_port: int = 5627
    subscribe_port: int = 5626
    link1: float = 0.34
    link2: float = 0.32
    inertia1: float = 1.8
    inertia2: float = 1.6
    damping: float = 0.7
    torque_scale: float = 0.25

    @property
    def steps_per_phase(self) -> int:
        return int(round(self.control_interval_ms / self.dt_ms))

    @property
    def epoch_steps(self) -> int:
        return self.phases * self.steps_per_phase


def build_network(cfg: ProjectConfig) -> tuple[nb.Network, LayerIndex, CerebellumConfig]:
    # Keep the published cerebellum topology and learning-rule parameters.
    base = CerebellumConfig(
        sample_count=cfg.phases,
        dt_ms=cfg.dt_ms,
        control_interval_ms=cfg.control_interval_ms,
        link1=cfg.link1,
        link2=cfg.link2,
        mass1=cfg.inertia1,
        mass2=cfg.inertia2,
    )
    network = nb.Network()
    index = add_layers(network, base)
    add_learning_rule(network, base)
    add_connections(network, base, index)
    # Deliberately no add_outer_dynamic(): the external ArmServer owns plant
    # dynamics and sends state/error spikes over asynchronous PUB/SUB.
    return network, index, base


def desired_state(phase: int, cfg: ProjectConfig) -> tuple[float, float, float, float, float, float]:
    fraction = phase / float(cfg.phases)
    x = 0.10 + 0.15 * math.cos(2.0 * math.pi * fraction)
    y = 0.40 + 0.15 * math.sin(2.0 * math.pi * fraction)
    c2 = max(-1.0, min(1.0, (x * x + y * y - cfg.link1**2 - cfg.link2**2) / (2.0 * cfg.link1 * cfg.link2)))
    q2 = math.acos(c2)
    phi = math.acos(max(-1.0, min(1.0, (cfg.link2**2 - cfg.link1**2 - x * x - y * y) / (-2.0 * cfg.link1 * math.hypot(x, y)))))
    q1 = math.atan2(y, x) - phi
    return q1, q2, x, y, 0.0, 0.0


class ArmServer:
    """External arm plant and cerebellar feedback encoder."""

    def __init__(self, cfg: ProjectConfig, base: CerebellumConfig, index: LayerIndex):
        self.cfg = cfg
        self.base = base
        self.index = index
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.received_steps: list[int] = []
        self.trajectory: list[dict] = []
        self.phase_errors: list[float] = []
        self.protocol_errors: list[str] = []
        self.q = [0.0, 0.0]
        self.qd = [0.0, 0.0]
        self.epoch = 0
        self.feedback_lead_steps = 0

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("arm server did not bind ZMQ sockets")
        time.sleep(0.30)

    def begin_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        target = desired_state(0, self.cfg)
        self.q = [target[0], target[1]]
        self.qd = [0.0, 0.0]
        self.feedback_lead_steps = self.cfg.steps_per_phase * 4

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5.0)

    def _run(self) -> None:
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        subscriber = context.socket(zmq.SUB)
        publisher.linger = 0
        subscriber.linger = 0
        publisher.bind(f"tcp://127.0.0.1:{self.cfg.subscribe_port}")
        subscriber.connect(f"tcp://127.0.0.1:{self.cfg.publish_port}")
        subscriber.setsockopt(zmq.SUBSCRIBE, OUTPUT_TOPIC)
        self.ready.set()
        try:
            while not self.stop.is_set():
                if not subscriber.poll(10, zmq.POLLIN):
                    continue
                parts = subscriber.recv_multipart()
                if len(parts) not in (2, 3) or parts[0] != OUTPUT_TOPIC or len(parts[1]) != HEADER.size:
                    self.protocol_errors.append("invalid output message framing")
                    continue
                magic, version, step, count = HEADER.unpack(parts[1])
                if magic != MAGIC or version != VERSION:
                    self.protocol_errors.append("invalid output protocol header")
                    continue
                payload = parts[2] if len(parts) == 3 else b""
                if len(payload) != count * SPIKE.size:
                    self.protocol_errors.append("invalid output payload length")
                    continue
                self.received_steps.append(step)
                self._apply_dcn_spikes(payload)
                target_epoch = max(0, step // self.cfg.epoch_steps)
                if target_epoch != self.epoch:
                    self.begin_epoch(target_epoch)
                phase = (step // self.cfg.steps_per_phase) % self.cfg.phases
                self._publish_feedback(publisher, step, phase)
        except Exception as exc:  # pragma: no cover - diagnostic path
            if not self.stop.is_set():
                self.protocol_errors.append(repr(exc))
        finally:
            subscriber.close(0)
            publisher.close(0)
            context.term()

    def _apply_dcn_spikes(self, payload: bytes) -> None:
        dcn_begin = 2 * self.base.n_gc + 4 * self.base.n_cf + 4 * self.base.n_pc
        torque = [0.0, 0.0]
        for offset in range(0, len(payload), SPIKE.size):
            neuron, _time_ms, _dt = SPIKE.unpack_from(payload, offset)
            local = neuron - dcn_begin
            group = local // self.base.n_dcn if local >= 0 else -1
            if group == 0:
                torque[0] += self.cfg.torque_scale
            elif group == 1:
                torque[0] -= self.cfg.torque_scale
            elif group == 2:
                torque[1] += self.cfg.torque_scale
            elif group == 3:
                torque[1] -= self.cfg.torque_scale
        dt = self.cfg.control_interval_ms * 0.001
        self.qd[0] += dt * (torque[0] - self.cfg.damping * self.qd[0]) / self.cfg.inertia1
        self.qd[1] += dt * (torque[1] - self.cfg.damping * self.qd[1]) / self.cfg.inertia2
        self.q[0] += dt * self.qd[0]
        self.q[1] += dt * self.qd[1]

    def _publish_feedback(self, publisher, step: int, phase: int) -> None:
        qd1, qd2, xd, yd, _v1, _v2 = desired_state(phase, self.cfg)
        x, y = hand_position(self.q[0], self.q[1], self.base)
        error_norm = math.hypot(x - xd, y - yd)
        self.phase_errors.append(error_norm)
        self.trajectory.append({"epoch": self.epoch, "step": step, "phase": phase,
                                "q1": self.q[0], "q2": self.q[1], "x": x, "y": y,
                                "desired_x": xd, "desired_y": yd, "error_norm": error_norm})

        def gc_index(q1: float, q2: float, v1: float, v2: float) -> int:
            n, nv = self.base.nn, self.base.nv
            b1 = max(0, min(n - 1, int((q1 + 0.6) / 1.8 * n)))
            b2 = max(0, min(n - 1, int(q2 / 2.5 * n)))
            bv1 = max(0, min(nv - 1, int((v1 + 4.0) / 8.0 * nv)))
            bv2 = max(0, min(nv - 1, int((v2 + 4.0) / 8.0 * nv)))
            return ((b1 * n + b2) * nv + bv1) * nv + bv2

        error1 = qd1 - self.q[0]
        error2 = qd2 - self.q[1]
        feedback_step = step + self.feedback_lead_steps
        spikes = [
            (gc_index(self.q[0], self.q[1], self.qd[0], self.qd[1]), feedback_step),
            (self.base.n_gc + gc_index(self.q[0], self.q[1], self.qd[0], self.qd[1]), feedback_step),
        ]
        cf_starts = [2 * self.base.n_gc, 2 * self.base.n_gc + self.base.n_cf,
                     2 * self.base.n_gc + 2 * self.base.n_cf, 2 * self.base.n_gc + 3 * self.base.n_cf]
        if error1 >= 0:
            spikes.append((cf_starts[0] + min(self.base.n_cf - 1, phase), feedback_step))
        else:
            spikes.append((cf_starts[1] + min(self.base.n_cf - 1, phase), feedback_step))
        if error2 >= 0:
            spikes.append((cf_starts[2] + min(self.base.n_cf - 1, phase), feedback_step))
        else:
            spikes.append((cf_starts[3] + min(self.base.n_cf - 1, phase), feedback_step))

        header = HEADER.pack(MAGIC, VERSION, feedback_step, len(spikes))
        payload = b"".join(SPIKE.pack(neuron, spike_step * self.cfg.dt_ms, self.cfg.dt_ms) for neuron, spike_step in spikes)
        publisher.send_multipart([INPUT_TOPIC, header, payload])


def run_training(args: argparse.Namespace) -> dict:
    cfg = ProjectConfig(epochs=args.epochs, phases=args.phases, queues=args.queues,
                        publish_port=args.publish_port, subscribe_port=args.subscribe_port)
    network, index, base = build_network(cfg)
    sim = nb.Simulation(network, nb.SimulationConfig(
        steps=cfg.epoch_steps * cfg.epochs, timestep=cfg.dt_ms, queues=cfg.queues,
        event_queue="timing_wheel", timing_wheel_size=256,
    ))
    server = ArmServer(cfg, base, index)
    server.start()
    sim.add_zmq_async_input_output_spike_driver(
        subscribe_address="127.0.0.1",
        publish_port=cfg.publish_port,
        subscribe_port=cfg.subscribe_port,
        publish_topic=OUTPUT_TOPIC.decode(),
        subscribe_topic=INPUT_TOPIC.decode(),
        communication_interval=cfg.steps_per_phase,
    )
    time.sleep(0.30)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_reports: list[dict] = []
    try:
        sim.init()
        server.begin_epoch(0)
        sim.enable_realtime(slot_steps=cfg.steps_per_phase, max_advance_seconds=0.05,
                            first_section=0.25, second_section=0.5, third_section=0.75)
        sim.run_realtime(cfg.epoch_steps * cfg.epochs)
        sim.disable_realtime()
        for epoch in range(cfg.epochs):
            errors = [row["error_norm"] for row in server.trajectory if row["epoch"] == epoch]
            epoch_reports.append({
                "epoch": epoch,
                "phase_count": len(errors),
                "mean_position_error_m": sum(errors) / len(errors) if errors else None,
                "max_position_error_m": max(errors) if errors else None,
                "min_position_error_m": min(errors) if errors else None,
            })
        weights_path = output_dir / "cerebellum_weights.bin"
        sim.save_weights(str(weights_path))
    finally:
        server.close()

    trajectory_path = output_dir / "arm_trajectory.jsonl"
    with trajectory_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in server.trajectory:
            handle.write(json.dumps(row) + "\n")
    report = {
        "project": "cerebellum_arm_server_project",
        "communication": "asynchronous ZMQ PUB/SUB",
        "control": "synchronous fixed phase boundaries paced by native realtime runtime",
        "error_definition": "Euclidean end-effector position error in metres",
        "configuration": {"epochs": cfg.epochs, "phases": cfg.phases, "steps_per_phase": cfg.steps_per_phase,
                          "epoch_steps": cfg.epoch_steps, "queues": cfg.queues, "dt_ms": cfg.dt_ms},
        "network": {"neurons": network.neuron_count, "connections": len(network.connections),
                    "learning_rules": len(network.learning_rules), "outer_dynamics": len(network.outer_dynamics)},
        "epochs": epoch_reports,
        "server": {"received_batches": len(server.received_steps),
                    "unique_steps": len(set(server.received_steps)),
                    "protocol_errors": server.protocol_errors},
        "weights": str(weights_path),
        "trajectory": str(trajectory_path),
    }
    means = [row["mean_position_error_m"] for row in epoch_reports if row["mean_position_error_m"] is not None]
    if len(means) >= 2:
        x_mean = sum(range(len(means))) / len(means)
        y_mean = sum(means) / len(means)
        slope = sum((x - x_mean) * (y - y_mean) for x, y in enumerate(means)) / sum((x - x_mean) ** 2 for x in range(len(means)))
        first_error = means[0]
        last_error = means[-1]
        convergence = {
            "epoch_mean_errors_m": means,
            "first_epoch_mean_error_m": first_error,
            "last_epoch_mean_error_m": last_error,
            "last_to_first_ratio": last_error / first_error if first_error > 0 else None,
            "linear_trend_m_per_epoch": slope,
            "decreased_from_first_to_last": last_error < first_error,
            "converged": slope < 0.0 and last_error < first_error * 0.95,
        }
    else:
        convergence = {"epoch_mean_errors_m": means, "converged": False}
    report["convergence"] = convergence
    report["pass"] = bool(epoch_reports) and not server.protocol_errors and len(server.received_steps) >= cfg.epochs * cfg.phases
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--phases", type=int, default=8)
    parser.add_argument("--queues", type=int, default=2)
    parser.add_argument("--publish-port", type=int, default=5627)
    parser.add_argument("--subscribe-port", type=int, default=5626)
    args = parser.parse_args()
    report = run_training(args)
    print(json.dumps({"pass": report["pass"], "epochs": report["epochs"],
                      "convergence": report["convergence"], "server": report["server"]}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
