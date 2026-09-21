from __future__ import annotations

import argparse
import json
import math
import random
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
class Config:
    epochs: int = 3
    sample_count: int = 50
    dt_ms: float = 0.1
    control_interval_ms: float = 20.0
    nn: int = 15
    nv: int = 10
    n_pc: int = 200
    n_cf: int = 200
    n_dcn: int = 200
    gc_repeats_per_control: int = 5
    link1: float = 0.34
    link2: float = 0.32
    mass1: float = 1.8
    mass2: float = 1.6
    torque_gain_1: float = 0.2
    torque_gain_2: float = 0.05
    cf_mix_position: float = 0.8
    spike_cf_max: float = 15.0
    pub_port: int = 5637
    sub_port: int = 5636
    control_port: int = 5638

    @property
    def n_gc(self) -> int:
        return self.nn * self.nn * self.nv * self.nv

    @property
    def steps_per_control(self) -> int:
        return int(round(self.control_interval_ms / self.dt_ms))

    @property
    def epoch_steps(self) -> int:
        return self.sample_count * self.steps_per_control


def make_network(cfg: Config) -> tuple[nb.Network, LayerIndex, CerebellumConfig]:
    base = CerebellumConfig(
        sample_count=cfg.sample_count,
        dt_ms=cfg.dt_ms,
        control_interval_ms=cfg.control_interval_ms,
        nn=cfg.nn,
        nv=cfg.nv,
        n_pc=cfg.n_pc,
        n_cf=cfg.n_cf,
        n_dcn=cfg.n_dcn,
        gc_repeats_per_control=cfg.gc_repeats_per_control,
        link1=cfg.link1,
        link2=cfg.link2,
        mass1=cfg.mass1,
        mass2=cfg.mass2,
        dcn_torque_gain_1=cfg.torque_gain_1,
        dcn_torque_gain_2=cfg.torque_gain_2,
        cf_mix_position=cfg.cf_mix_position,
        spike_cf_max=cfg.spike_cf_max,
    )
    network = nb.Network()
    index = add_layers(network, base)
    add_learning_rule(network, base)
    add_connections(network, base, index)
    return network, index, base


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def desired_trajectory(cfg: Config) -> list[tuple[float, float, float, float, float, float]]:
    values: list[tuple[float, float, float, float, float, float]] = []
    total_ms = cfg.control_interval_ms * cfg.sample_count
    for i in range(cfg.sample_count + 1):
        phase = i * cfg.control_interval_ms / total_ms
        x = 0.15 * math.cos(2.0 * math.pi * phase) + 0.1
        y = 0.15 * math.sin(2.0 * math.pi * phase) + 0.4
        c2 = clamp((x * x + y * y - cfg.link1**2 - cfg.link2**2) /
                   (2.0 * cfg.link1 * cfg.link2), -1.0, 1.0)
        q2 = math.acos(c2)
        phi = math.acos(clamp((cfg.link2**2 - cfg.link1**2 - x*x - y*y) /
                              (-2.0 * cfg.link1 * math.hypot(x, y)), -1.0, 1.0))
        q1 = math.atan2(y, x) - phi
        values.append((q1, q2, x, y, 0.0, 0.0))
    dt_s = cfg.control_interval_ms * 0.001
    velocities = []
    for i in range(cfg.sample_count):
        velocities.append(((values[i + 1][0] - values[i][0]) / dt_s,
                           (values[i + 1][1] - values[i][1]) / dt_s))
    velocities.append(velocities[-1])
    return [(q1, q2, x, y, velocities[i][0], velocities[i][1])
            for i, (q1, q2, x, y, _, _) in enumerate(values)]


def uniform_bin(value: float, low: float, high: float, count: int) -> int:
    if count <= 1 or high <= low:
        return 0
    ratio = (clamp(value, low, high) - low) / (high - low)
    return min(count - 1, max(0, int(math.floor(ratio * count))))


def state_product_index(joint: int, state: tuple[float, float, float, float], target: tuple[float, float, float, float], cfg: Config) -> int:
    q, qd = state[joint], state[2 + joint]
    qdes, qddes = target[joint], target[4 + joint]
    angle_min = (-30.0, 0.1)[joint]
    angle_max = (90.0, 150.0)[joint]
    values = [math.degrees(qdes), math.degrees(q), math.degrees(qddes), math.degrees(qd)]
    mins = [angle_min, angle_min, -400.0, -400.0]
    maxs = [angle_max, angle_max, 400.0, 400.0]
    bins = [cfg.nn, cfg.nn, cfg.nv, cfg.nv]
    result = 0
    for value, low, high, count in zip(values, mins, maxs, bins):
        result = result * count + uniform_bin(value, low, high, count)
    return result


class ArmServer:
    def __init__(self, cfg: Config, base: CerebellumConfig, index: LayerIndex, trajectory):
        self.cfg, self.base, self.index, self.reference = cfg, base, index, trajectory
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.rng = random.Random(17)
        self.received_steps: list[int] = []
        self.trajectory_log: list[dict] = []
        self.protocol_errors: list[str] = []
        self.q = [0.0, 0.0]
        self.qd = [0.0, 0.0]
        self.epoch = 0
        self.stream_id = 1
        self.accept_data = False
        self.last_sequence = 0
        self.pending_resets: list[dict] = []
        # PUB/SUB is intentionally asynchronous.  Reserve future simulation
        # time for the round trip so a delayed feedback packet is never queued
        # behind the native current time.
        self.feedback_lead_steps = cfg.steps_per_control * 4

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("arm server failed to bind")
        time.sleep(0.35)

    def begin_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        target = self.reference[1]
        self.q = [target[0], target[1]]
        self.qd = [target[4], target[5]]

    def _handle_control(self, rep, message: dict) -> None:
        message_type = message.get("type")
        request_id = int(message.get("request_id", 0))
        stream_id = int(message.get("stream_id", self.stream_id))
        if message_type == "RESET_PREPARE":
            self.accept_data = False
            rep.send_json({"version": 1, "type": "RESET_PREPARED",
                           "request_id": request_id + 1,
                           "correlation_id": request_id,
                           "stream_id": self.stream_id,
                           "simulation_step": int(message.get("simulation_step", 0)),
                           "payload": {"last_consumed_sequence": self.last_sequence}})
            return
        if message_type == "SCHEDULE_RESET":
            payload = message.get("payload", {})
            self.pending_resets.append({
                "apply_at_step": int(message.get("simulation_step", 0)),
                "stream_id": stream_id,
                "initial_q": payload.get("initial_q", [self.reference[1][0], self.reference[1][1]]),
                "initial_qd": payload.get("initial_qd", [self.reference[1][4], self.reference[1][5]]),
            })
            self.pending_resets.sort(key=lambda item: item["apply_at_step"])
            rep.send_json({"version": 1, "type": "RESET_SCHEDULED",
                           "request_id": request_id + 1,
                           "correlation_id": request_id,
                           "stream_id": stream_id,
                           "simulation_step": int(message.get("simulation_step", 0)),
                           "payload": {"accepted": True}})
            return
        if message_type == "RESET_COMMIT":
            payload = message.get("payload", {})
            q = payload.get("initial_q", [self.reference[1][0], self.reference[1][1]])
            qd = payload.get("initial_qd", [self.reference[1][4], self.reference[1][5]])
            self.q = [float(q[0]), float(q[1])]
            self.qd = [float(qd[0]), float(qd[1])]
            self.stream_id = stream_id
            self.accept_data = True
            rep.send_json({"version": 1, "type": "RESET_READY",
                           "request_id": request_id + 1,
                           "correlation_id": request_id,
                           "stream_id": self.stream_id,
                           "simulation_step": 0,
                           "payload": {"accepted": True}})
            return
        if message_type == "PING":
            rep.send_json({"version": 1, "type": "PONG",
                           "request_id": request_id + 1,
                           "correlation_id": request_id,
                           "stream_id": self.stream_id,
                           "simulation_step": 0, "payload": {}})
            return
        rep.send_json({"version": 1, "type": "ERROR",
                       "request_id": request_id + 1,
                       "correlation_id": request_id,
                       "stream_id": self.stream_id,
                       "simulation_step": 0,
                       "payload": {"message": "unknown control type"}})

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5.0)

    def _run(self) -> None:
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        sub = context.socket(zmq.SUB)
        control = context.socket(zmq.REP)
        pub.linger = sub.linger = 0
        control.linger = 0
        pub.bind(f"tcp://127.0.0.1:{self.cfg.sub_port}")
        sub.connect(f"tcp://127.0.0.1:{self.cfg.pub_port}")
        sub.setsockopt(zmq.SUBSCRIBE, OUTPUT_TOPIC)
        control.bind(f"tcp://127.0.0.1:{self.cfg.control_port}")
        self.ready.set()
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        poller.register(control, zmq.POLLIN)
        try:
            while not self.stop.is_set():
                events = dict(poller.poll(20))
                if control in events:
                    self._handle_control(control, control.recv_json())
                if sub not in events:
                    continue
                parts = sub.recv_multipart()
                if len(parts) not in (2, 3) or parts[0] != OUTPUT_TOPIC or len(parts[1]) != HEADER.size:
                    self.protocol_errors.append("invalid output framing")
                    continue
                magic, version, step, count = HEADER.unpack(parts[1])
                payload = parts[2] if len(parts) == 3 else b""
                if magic != MAGIC or version != VERSION or len(payload) != count * SPIKE.size:
                    self.protocol_errors.append("invalid output header or payload")
                    continue
                while self.pending_resets and step >= self.pending_resets[0]["apply_at_step"]:
                    reset = self.pending_resets.pop(0)
                    self.q = [float(reset["initial_q"][0]), float(reset["initial_q"][1])]
                    self.qd = [float(reset["initial_qd"][0]), float(reset["initial_qd"][1])]
                    self.stream_id = int(reset["stream_id"])
                    self.epoch = self.stream_id - 1
                    self.accept_data = True
                if not self.accept_data:
                    continue
                self.received_steps.append(step)
                self.last_sequence += 1
                torque = ControlClient._decode_torque(self, payload)
                ControlClient._step_arm(self, torque)
                # The C++ reproduction initializes at sample 1 and evaluates
                # control windows from sample 2 through sample_count-1.
                phase = min(self.cfg.sample_count - 1,
                            max(2, (step // self.cfg.steps_per_control) % self.cfg.sample_count))
                ControlClient._publish_feedback(self, pub, step, phase)
        except Exception as exc:
            if not self.stop.is_set():
                self.protocol_errors.append(repr(exc))
        finally:
            sub.close(0)
            pub.close(0)
            control.close(0)
            context.term()


class ControlClient:
    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.connect(f"tcp://127.0.0.1:{port}")
        self.request_id = 1

    def request(self, message_type: str, stream_id: int, simulation_step: int, payload: dict | None = None) -> dict:
        request_id = self.request_id
        self.request_id += 1
        message = {"version": 1, "type": message_type, "request_id": request_id,
                   "correlation_id": 0, "stream_id": stream_id,
                   "simulation_step": simulation_step, "payload": payload or {}}
        self.socket.send_json(message)
        reply = self.socket.recv_json()
        if reply.get("correlation_id") != request_id:
            raise RuntimeError(f"control correlation mismatch: {reply}")
        if reply.get("type") == "ERROR":
            raise RuntimeError(f"control error: {reply}")
        return reply

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()

    def _decode_torque(self, payload: bytes) -> tuple[float, float]:
        dcn_begin = 2 * self.base.n_gc + 4 * self.base.n_cf + 4 * self.base.n_pc
        counts = [0, 0, 0, 0]
        for offset in range(0, len(payload), SPIKE.size):
            neuron, _, _ = SPIKE.unpack_from(payload, offset)
            group = (neuron - dcn_begin) // self.base.n_dcn
            if 0 <= group < 4:
                counts[group] += 1
        return (self.cfg.torque_gain_1 * (counts[0] - counts[1]),
                self.cfg.torque_gain_2 * (counts[2] - counts[3]))

    def _step_arm(self, torque: tuple[float, float]) -> None:
        theta2 = self.q[1]
        d1, d2 = self.cfg.link1 * 0.5, self.cfg.link2 * 0.5
        i1 = self.cfg.mass1 * self.cfg.link1**2 / 3.0
        i2 = self.cfg.mass2 * self.cfg.link2**2 / 12.0
        c2, s2 = math.cos(theta2), math.sin(theta2)
        m11 = i1 + i2 + 2.0*self.cfg.mass2*self.cfg.link1*d2*c2 + self.cfg.mass1*d1*d1 + self.cfg.mass2*(d2*d2 + self.cfg.link1**2)
        m12 = i2 + self.cfg.mass2*self.cfg.link1*d2*c2 + self.cfg.mass2*d2*d2
        m22 = i2 + self.cfg.mass2*d2*d2
        rhs1 = torque[0] + 2.0*self.cfg.mass2*self.cfg.link1*d2*s2*self.qd[0]*self.qd[1] + self.cfg.mass2*self.cfg.link1*d2*s2*self.qd[1]**2
        rhs2 = torque[1] - self.cfg.mass2*self.cfg.link1*d2*s2*self.qd[0]**2
        det = m11*m22 - m12*m12
        qdd = ((rhs1*m22-rhs2*m12)/det, (m11*rhs2-m12*rhs1)/det) if abs(det) >= 1e-12 else (0.0, 0.0)
        dt = self.cfg.steps_per_control * self.cfg.dt_ms * 0.001
        old_qd = tuple(self.qd)
        self.qd[0] += qdd[0] * dt
        self.qd[1] += qdd[1] * dt
        self.q[0] = clamp(self.q[0] + old_qd[0] * dt, math.radians(-30.0), math.radians(90.0))
        self.q[1] = clamp(self.q[1] + old_qd[1] * dt, math.radians(0.1), math.radians(150.0))

    def _publish_feedback(self, pub, step: int, phase: int) -> None:
        target = self.reference[phase]
        state = (self.q[0], self.q[1], self.qd[0], self.qd[1])
        x, y = hand_position(self.q[0], self.q[1], self.base)
        error = math.hypot(x - target[2], y - target[3])
        self.trajectory_log.append({"epoch": self.epoch, "step": step, "phase": phase,
                                    "x": x, "y": y, "desired_x": target[2], "desired_y": target[3],
                                    "q1": self.q[0], "q2": self.q[1], "error_m": error})
        emit_step = step + self.feedback_lead_steps
        spikes = []
        for joint, start in ((0, self.index.gc1), (1, self.index.gc2)):
            spikes.append((start + state_product_index(joint, state, target, self.cfg), emit_step))
        cf_starts = [2*self.base.n_gc, 2*self.base.n_gc+self.base.n_cf,
                     2*self.base.n_gc+2*self.base.n_cf, 2*self.base.n_gc+3*self.base.n_cf]
        for joint, (pos_start, neg_start) in enumerate(((cf_starts[0], cf_starts[1]), (cf_starts[2], cf_starts[3]))):
            pos_error = math.degrees(target[joint] - self.q[joint])
            vel_error = math.degrees(target[4+joint] - self.qd[joint])
            angle_norm = min(1.0, abs(pos_error) / (60.0, 75.0)[joint]) * (-1.0 if pos_error < 0 else 1.0)
            vel_norm = min(1.0, abs(vel_error) / 400.0) * (-1.0 if vel_error < 0 else 1.0)
            mixed = self.cfg.cf_mix_position * angle_norm + (1.0-self.cfg.cf_mix_position) * vel_norm
            count = int(round(abs(self.cfg.spike_cf_max * self.base.n_cf / self.cfg.sample_count * mixed)))
            targets = list(range(pos_start, pos_start+self.base.n_cf)) if mixed > 0 else list(range(neg_start, neg_start+self.base.n_cf))
            seen = set()
            for _ in range(count):
                neuron = self.rng.choice(targets)
                local = self.rng.randrange(self.cfg.steps_per_control)
                key = (neuron, local)
                if key not in seen:
                    seen.add(key)
                    spikes.append((neuron, emit_step + 1 + local))
        header = HEADER.pack(MAGIC, VERSION, emit_step, len(spikes))
        payload = b"".join(SPIKE.pack(n, t*self.cfg.dt_ms, self.cfg.dt_ms) for n, t in spikes)
        pub.send_multipart([INPUT_TOPIC, header, payload])


def train(args: argparse.Namespace) -> dict:
    cfg = Config(epochs=args.epochs, sample_count=args.sample_count, nn=args.nn, nv=args.nv,
                 n_pc=args.n_pc, n_cf=args.n_cf, n_dcn=args.n_dcn)
    network, index, base = make_network(cfg)
    reference = desired_trajectory(cfg)
    sim = nb.Simulation(network, nb.SimulationConfig(steps=cfg.epoch_steps*cfg.epochs,
                                                     timestep=cfg.dt_ms, queues=2,
                                                     event_queue="timing_wheel", timing_wheel_size=256))
    server = ArmServer(cfg, base, index, reference)
    control = ControlClient(cfg.control_port)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    try:
        server.start()
        sim.add_zmq_async_input_output_spike_driver(
            subscribe_address="127.0.0.1", publish_port=cfg.pub_port,
            subscribe_port=cfg.sub_port, publish_topic=OUTPUT_TOPIC.decode(),
            subscribe_topic=INPUT_TOPIC.decode(), communication_interval=cfg.steps_per_control)
        sim.init()
        sim.reset_realtime_skip_counters()
        sim.reset_realtime_restriction_counts()
        for round_index in range(cfg.epochs):
            stream_id = round_index + 1
            control.request("SCHEDULE_RESET", stream_id, round_index * cfg.epoch_steps,
                            {"initial_q": [reference[1][0], reference[1][1]],
                             "initial_qd": [reference[1][4], reference[1][5]]})
        # The C++-sized network needs a larger wall-clock allowance.  With a
        # 50 ms allowance the native watchdog disables output events after a
        # few long epochs while the asynchronous peer is still draining.
        sim.enable_realtime(slot_steps=args.slot_steps or cfg.steps_per_control,
                            max_advance_seconds=args.max_advance_seconds,
                            first_section=args.first_section,
                            second_section=args.second_section,
                            third_section=args.third_section)
        # One native realtime run; the server applies the pre-scheduled
        # control messages when the corresponding simulation step arrives.
        sim.run_realtime(cfg.epoch_steps * cfg.epochs)
        sim.disable_realtime()
        # PUB/SUB delivery is asynchronous; allow the server to consume the
        # final output batches before closing its subscriber.
        time.sleep(0.75)
        realtime_skip_counters = sim.realtime_skip_counters()
        realtime_restriction_counts = sim.realtime_restriction_counts()
        reports = []
        for epoch in range(cfg.epochs):
            rows = [row for row in server.trajectory_log if row["epoch"] == epoch]
            errors = [row["error_m"] for row in rows]
            reports.append({"epoch": epoch, "windows": len(errors),
                            "mean_error_m": sum(errors)/len(errors) if errors else None,
                            "max_error_m": max(errors) if errors else None})
        weights = out / "weights_final.bin"
        sim.save_weights(str(weights))
    finally:
        server.close()
        control.close()
    trajectory = out / "trajectory.jsonl"
    with trajectory.open("w", encoding="utf-8") as handle:
        for row in server.trajectory_log:
            handle.write(json.dumps(row) + "\n")
    means = [r["mean_error_m"] for r in reports if r["mean_error_m"] is not None]
    result = {"project": "cerebellum_arm_realtime_async_cpp_aligned", "communication": "async ZMQ PUB/SUB",
              "control": "native realtime control windows", "configuration": cfg.__dict__,
              "network": {"neurons": network.neuron_count, "connections": len(network.connections),
                          "learning_rules": len(network.learning_rules)}, "epochs": reports,
              "realtime_parameters": {"slot_steps": args.slot_steps or cfg.steps_per_control,
                                      "max_advance_seconds": args.max_advance_seconds,
                                      "first_section": args.first_section,
                                      "second_section": args.second_section,
                                      "third_section": args.third_section},
              "realtime_skip_counters": realtime_skip_counters,
              "realtime_restriction_counts": realtime_restriction_counts,
              "received_batches": len(server.received_steps), "protocol_errors": server.protocol_errors,
              "weights": str(weights), "trajectory": str(trajectory),
              "pass": bool(reports) and not server.protocol_errors and all(
                  r["windows"] >= cfg.sample_count - 1 for r in reports),
              "error_decreased": len(means) > 1 and means[-1] < means[0]}
    (out / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--nn", type=int, default=15)
    parser.add_argument("--nv", type=int, default=10)
    parser.add_argument("--n-pc", type=int, default=200)
    parser.add_argument("--n-cf", type=int, default=200)
    parser.add_argument("--n-dcn", type=int, default=200)
    parser.add_argument("--slot-steps", type=int, default=0,
                        help="Watchdog/control slot in simulation steps; 0 uses steps_per_control")
    parser.add_argument("--max-advance-seconds", type=float, default=5.0)
    parser.add_argument("--first-section", type=float, default=0.25)
    parser.add_argument("--second-section", type=float, default=0.5)
    parser.add_argument("--third-section", type=float, default=0.75)
    result = train(parser.parse_args())
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
