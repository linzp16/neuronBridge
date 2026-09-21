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


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.lif_decay(1, output=True, monitored=True))
    network.connect(nb.Connection(
        source=[0], target=[1], weight=[25.0], max_weight=[100.0], delay=[1],
    ))
    return network


def run_local(realtime: bool, queues: int = 1) -> dict:
    sim = nb.Simulation(
        build_network(),
        nb.SimulationConfig(steps=STEPS, timestep=TIMESTEP_MS, queues=queues, event_queue="heap"),
    )
    sim.add_external_spikes([1, 30, 60, 90, 120, 150], [0] * 6)
    started = time.perf_counter()
    if realtime:
        sim.enable_realtime(
            slot_steps=COMMUNICATION_INTERVAL,
            max_advance_seconds=0.05,
            first_section=0.25,
            second_section=0.5,
            third_section=0.75,
        )
        sim.run_realtime(STEPS)
        sim.disable_realtime()
    else:
        sim.run(STEPS)
    elapsed = time.perf_counter() - started
    return {
        "mode": "realtime" if realtime else "normal",
        "queues": queues,
        "elapsed_seconds": elapsed,
        "simulated_seconds": STEPS * TIMESTEP_MS / 1000.0,
        "output_spikes": sim.output_spikes(),
    }


class RepProbe:
    def __init__(self, port: int):
        self.port = port
        self.requests: list[dict] = []
        self.errors: list[str] = []
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("REP probe did not bind within 5 seconds")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5.0)

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.linger = 0
        socket.bind(f"tcp://127.0.0.1:{self.port}")
        self.ready.set()
        try:
            while not self.stop.is_set():
                if not socket.poll(100, zmq.POLLIN):
                    continue
                request = socket.recv()
                if len(request) < 8:
                    self.errors.append(f"request too short: {len(request)}")
                    socket.send(struct.pack("<I", 0))
                    continue
                time_step, output_count = struct.unpack_from("<II", request, 0)
                self.requests.append({
                    "time_step": time_step,
                    "output_count": output_count,
                    "wall_time": time.perf_counter(),
                    "bytes": len(request),
                })
                socket.send(struct.pack("<I", 0))
        except Exception as exc:  # pragma: no cover - diagnostic path
            if not self.stop.is_set():
                self.errors.append(repr(exc))
        finally:
            socket.close(0)
            context.term()


def run_realtime_req_rep(port: int, queues: int = 2) -> dict:
    probe = RepProbe(port)
    probe.start()
    try:
        sim = nb.Simulation(
            build_network(),
            nb.SimulationConfig(steps=STEPS, timestep=TIMESTEP_MS, queues=queues, event_queue="heap"),
        )
        sim.add_external_spikes([1, 30, 60, 90, 120, 150], [0] * 6)
        sim.add_zmq_input_output_spike_driver(
            server_address="127.0.0.1",
            server_port=port,
            communication_interval=COMMUNICATION_INTERVAL,
        )
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
        return {
            "mode": "realtime_req_rep",
            "queues": queues,
            "elapsed_seconds": elapsed,
            "simulated_seconds": STEPS * TIMESTEP_MS / 1000.0,
            "communication_interval_steps": COMMUNICATION_INTERVAL,
            "requests": probe.requests,
            "errors": probe.errors,
            "output_spikes": sim.output_spikes(),
        }
    finally:
        probe.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5595)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normal = run_local(False, queues=1)
    realtime_cases = {str(queues): run_local(True, queues=queues) for queues in (1, 2, 4)}
    realtime = realtime_cases["1"]
    req_rep = run_realtime_req_rep(args.port, queues=2)

    simulated = normal["simulated_seconds"]
    request_steps = [item["time_step"] for item in req_rep["requests"]]
    expected_min = max(1, STEPS // COMMUNICATION_INTERVAL - 2)
    report = {
        "configuration": {
            "steps": STEPS,
            "timestep_ms": TIMESTEP_MS,
            "communication_interval_steps": COMMUNICATION_INTERVAL,
            "realtime_watchdog": {
                "max_advance_seconds": 0.05,
                "sections": [0.25, 0.5, 0.75],
            },
        },
        "normal": normal,
        "realtime": realtime,
        "realtime_by_queues": realtime_cases,
        "realtime_req_rep": req_rep,
        "checks": {
            "realtime_api_available": True,
            "normal_and_realtime_output_equal": normal["output_spikes"] == realtime["output_spikes"],
            "realtime_wall_time_not_zero": all(
                case["elapsed_seconds"] >= simulated * 0.25 for case in realtime_cases.values()
            ),
            "realtime_queue_outputs_equal": all(
                case["output_spikes"] == realtime["output_spikes"]
                for case in realtime_cases.values()
            ),
            "req_rep_has_enough_windows": len(req_rep["requests"]) >= expected_min,
            "req_rep_all_replied": not req_rep["errors"],
            "req_rep_time_steps_monotonic": request_steps == sorted(request_steps),
            "req_rep_time_steps_unique": len(request_steps) == len(set(request_steps)),
        },
    }
    report["pass"] = all(report["checks"].values())
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
