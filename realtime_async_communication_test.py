from __future__ import annotations

import argparse
import json
import struct
import threading
import time
from pathlib import Path

import neuronbridge as nb
import zmq


STEPS = 200
TIMESTEP_MS = 1.0
COMMUNICATION_INTERVAL = 5
MAGIC = 0x53504B31  # SPK1
VERSION = 1
OUTPUT_TOPIC = b"/snn/output_spikes"
INPUT_TOPIC = b"/robot/input_spikes"


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1, output=True))
    network.connect(nb.Connection(source=[0], target=[1], weight=[25.0], max_weight=[100.0], delay=[1]))
    return network


class AsyncProbe:
    def __init__(self, publish_port: int, subscribe_port: int):
        self.publish_port = publish_port
        self.subscribe_port = subscribe_port
        self.received: list[dict] = []
        self.errors: list[str] = []
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("async PUB/SUB probe did not bind within 5 seconds")
        # Allow both ZeroMQ subscriptions to complete before the simulation starts.
        time.sleep(0.25)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5.0)

    def _serve(self) -> None:
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        subscriber = context.socket(zmq.SUB)
        publisher.linger = 0
        subscriber.linger = 0
        publisher.bind(f"tcp://127.0.0.1:{self.subscribe_port}")
        subscriber.connect(f"tcp://127.0.0.1:{self.publish_port}")
        subscriber.setsockopt(zmq.SUBSCRIBE, OUTPUT_TOPIC)
        self.ready.set()
        last_input = 0.0
        try:
            while not self.stop.is_set():
                now = time.perf_counter()
                if now - last_input >= 0.02:
                    header = struct.pack("<IIII", MAGIC, VERSION, 0, 0)
                    publisher.send_multipart([INPUT_TOPIC, header])
                    last_input = now
                if subscriber.poll(5, zmq.POLLIN):
                    parts = subscriber.recv_multipart()
                    if len(parts) not in (2, 3) or parts[0] != OUTPUT_TOPIC or len(parts[1]) != 16:
                        self.errors.append(f"invalid output multipart message: {len(parts)} parts")
                        continue
                    magic, version, time_step, spike_count = struct.unpack("<IIII", parts[1])
                    expected_payload = spike_count * 12
                    payload_size = len(parts[2]) if len(parts) == 3 else 0
                    if magic != MAGIC or version != VERSION or payload_size != expected_payload:
                        self.errors.append("invalid output header or payload size")
                    self.received.append({
                        "time_step": time_step,
                        "spike_count": spike_count,
                        "payload_bytes": payload_size,
                        "wall_time": now,
                    })
        except Exception as exc:  # pragma: no cover - diagnostic path
            if not self.stop.is_set():
                self.errors.append(repr(exc))
        finally:
            subscriber.close(0)
            publisher.close(0)
            context.term()


def run_case(queues: int, publish_port: int, subscribe_port: int) -> dict:
    probe = AsyncProbe(publish_port, subscribe_port)
    probe.start()
    try:
        sim = nb.Simulation(
            build_network(),
            nb.SimulationConfig(steps=STEPS, timestep=TIMESTEP_MS, queues=queues, event_queue="heap"),
        )
        sim.add_external_spikes([1, 30, 60, 90, 120, 150], [0] * 6)
        sim.add_zmq_async_input_output_spike_driver(
            subscribe_address="127.0.0.1",
            publish_port=publish_port,
            subscribe_port=subscribe_port,
            publish_topic="/snn/output_spikes",
            subscribe_topic="/robot/input_spikes",
            communication_interval=COMMUNICATION_INTERVAL,
        )
        # PUB/SUB has no handshake at the application-message level. Give the
        # native SUB socket time to finish connecting before the first flush.
        time.sleep(0.30)
        started = time.perf_counter()
        sim.enable_realtime(
            slot_steps=COMMUNICATION_INTERVAL,
            max_advance_seconds=0.05,
            first_section=0.25,
            second_section=0.5,
            third_section=0.75,
        )
        sim.run_realtime(STEPS)
        elapsed = time.perf_counter() - started
        sim.disable_realtime()
        # Drain messages already queued by the PUB socket before closing the probe.
        time.sleep(0.30)
        return {
            "queues": queues,
            "elapsed_seconds": elapsed,
            "simulated_seconds": STEPS * TIMESTEP_MS / 1000.0,
            "received": probe.received,
            "errors": probe.errors,
        }
    finally:
        probe.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-port", type=int, default=5605)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        str(queues): run_case(queues, args.base_port + queues * 2, args.base_port + queues * 2 + 1)
        for queues in (1, 2, 4)
    }
    simulated = STEPS * TIMESTEP_MS / 1000.0
    checks = {}
    for key, case in cases.items():
        steps = [item["time_step"] for item in case["received"]]
        checks[f"queues_{key}_received_messages"] = len(case["received"]) >= STEPS // COMMUNICATION_INTERVAL - 2
        checks[f"queues_{key}_no_protocol_errors"] = not case["errors"]
        checks[f"queues_{key}_time_steps_monotonic"] = steps == sorted(steps)
        checks[f"queues_{key}_time_steps_unique"] = len(steps) == len(set(steps))
        checks[f"queues_{key}_realtime_paced"] = case["elapsed_seconds"] >= simulated * 0.25

    report = {
        "mode": "native_realtime_plus_async_zmq_pub_sub",
        "configuration": {
            "steps": STEPS,
            "timestep_ms": TIMESTEP_MS,
            "communication_interval_steps": COMMUNICATION_INTERVAL,
            "topics": {"output": OUTPUT_TOPIC.decode(), "input": INPUT_TOPIC.decode()},
            "protocol": "SPK1 v1: topic + 16-byte header + optional OutputSpikeIO payload",
        },
        "cases": cases,
        "checks": checks,
    }
    report["pass"] = all(checks.values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "checks": checks,
                      "received": {key: len(value["received"]) for key, value in cases.items()}}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
