"""Verify that synchronous REQ/REP reply spikes re-enter the native event queue."""

from __future__ import annotations

import argparse
import json
import struct
import threading

import neuronbridge as nb
import zmq


REQUEST_HEADER = struct.Struct("<II")
SPIKE_RECORD = struct.Struct("<iff")
REPLY_COUNT = struct.Struct("<I")


class ProbeServer:
    def __init__(self, port: int, timestep_ms: float):
        self.port = port
        self.timestep_ms = timestep_ms
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.requests: list[dict[str, object]] = []
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("probe server did not bind")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(5.0)

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.linger = 0
        socket.bind(f"tcp://127.0.0.1:{self.port}")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self.ready.set()
        try:
            while not self.stop.is_set():
                if socket not in dict(poller.poll(50)):
                    continue
                request = socket.recv()
                step, count = REQUEST_HEADER.unpack_from(request)
                returned = [
                    SPIKE_RECORD.unpack_from(request, REQUEST_HEADER.size + i * SPIKE_RECORD.size)
                    for i in range(count)
                ]
                self.requests.append({"step": step, "returned_spikes": returned})
                # Inject one spike one step after every communication boundary.
                payload = SPIKE_RECORD.pack(0, (step + 1) * self.timestep_ms, self.timestep_ms)
                socket.send_multipart([REPLY_COUNT.pack(1), payload])
        finally:
            socket.close(0)
            context.term()


def run(event_queue: str, port: int) -> dict[str, object]:
    timestep_ms = 0.1
    communication_interval = 5
    network = nb.Network()
    network.add_layer(nb.NeuronLayer("InputSpikeNeuronModel", 1, output=True))
    server = ProbeServer(port, timestep_ms)
    server.start()
    try:
        sim = nb.Simulation(
            network,
            nb.SimulationConfig(
                steps=25,
                timestep=timestep_ms,
                event_queue=event_queue,
                timing_wheel_size=64,
            ),
        )
        sim.add_zmq_input_output_spike_driver(
            server_address="127.0.0.1",
            server_port=port,
            communication_interval=communication_interval,
        )
        sim.init()
        sim.run(25)
    finally:
        server.close()
    returned_counts = [len(item["returned_spikes"]) for item in server.requests]
    result = {
        "event_queue": event_queue,
        "request_steps": [item["step"] for item in server.requests],
        "returned_counts": returned_counts,
        "pass": len(server.requests) == 5 and returned_counts == [0, 1, 1, 1, 1],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5655)
    args = parser.parse_args()
    results = [run("heap", args.port), run("timing_wheel", args.port + 1)]
    print(json.dumps(results, indent=2))
    if not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
