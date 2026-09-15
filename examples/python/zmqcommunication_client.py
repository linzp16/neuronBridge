"""Python migration for the C++ zmqcommunication_client example.

The simulation network is built through the public neuronbridge API. The ZMQ
driver is optional because communication validation needs an external compatible
PUB/SUB peer.
"""

from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import neuronbridge as nb
from _example_paths import artifact_dir


ASYNC_SPIKE_MAGIC = 0x53504B31
ASYNC_SPIKE_PROTOCOL_VERSION = 1
DEFAULT_SNN_OUTPUT_TOPIC = "/snn/output_spikes"
DEFAULT_ROBOT_INPUT_TOPIC = "/robot/input_spikes"
ASYNC_ROBOT_PUB_PORT = 5566
ASYNC_SNN_PUB_PORT = 5567
HEADER_STRUCT = struct.Struct("<IIII")
SPIKE_STRUCT = struct.Struct("<iff")


@dataclass(frozen=True)
class SpikeRecord:
    neuron: int
    time: float
    base_timestep: float = 0.2


def pack_spike_batch(time_step: int, spikes: list[SpikeRecord]) -> tuple[bytes, bytes]:
    header = HEADER_STRUCT.pack(ASYNC_SPIKE_MAGIC, ASYNC_SPIKE_PROTOCOL_VERSION, int(time_step), len(spikes))
    payload = b"".join(SPIKE_STRUCT.pack(int(spike.neuron), float(spike.time), float(spike.base_timestep)) for spike in spikes)
    return header, payload


def unpack_spike_batch(header: bytes, payload: bytes) -> tuple[int, list[SpikeRecord]]:
    magic, version, time_step, spike_count = HEADER_STRUCT.unpack(header)
    if magic != ASYNC_SPIKE_MAGIC:
        raise ValueError(f"invalid ZMQ spike magic: {magic:#x}")
    if version != ASYNC_SPIKE_PROTOCOL_VERSION:
        raise ValueError(f"unsupported ZMQ spike protocol version: {version}")
    expected = spike_count * SPIKE_STRUCT.size
    if len(payload) != expected:
        raise ValueError(f"invalid ZMQ spike payload size: {len(payload)}; expected {expected}")
    spikes = [
        SpikeRecord(*SPIKE_STRUCT.unpack_from(payload, offset))
        for offset in range(0, len(payload), SPIKE_STRUCT.size)
    ]
    return time_step, spikes


def build_network() -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2, monitored=True))
    network.add_layer(nb.NeuronLayer("TimeDrivenLIF_Exponential_triple_GPU", 2, monitored=True, output=True))
    network.add_layer(nb.NeuronLayer.input_spike(1))
    network.add_layer(nb.NeuronLayer.input_current(1))
    network.connect(nb.Connection(source=[0, 1], target=[2, 2], weight=[1.0, 1.0], max_weight=10.0, delay=1))
    network.connect(nb.Connection(source=4, target=2, weight=0.0, max_weight=10.0, delay=1, trigger_rule=0))
    network.connect(nb.Connection(source=5, target=3, synapse_type=3, weight=1.0, max_weight=10.0, delay=1))
    network.add_learning_rule(
        nb.LearningRule(
            "CerebullarLearningRule",
            parameters={"fixweightchange": nb.float32(0.2), "kernalchange": nb.float32(-0.3)},
        )
    )
    return network


def add_demo_inputs(sim: nb.Simulation) -> None:
    spike_times: list[int] = []
    spike_ids: list[int] = []
    for value in range(15, 500, 150):
        spike_times.append(value)
        spike_ids.append(0)
    for value in range(10, 500, 400):
        spike_times.append(value)
        spike_ids.append(1)
    for value in range(0, 500, 250):
        spike_times.append(value)
        spike_ids.append(4)
    sim.add_external_spikes(spike_times, spike_ids)
    sim.add_external_currents([0], [5], [12.8])


def run_example(output_dir: Path, *, enable_zmq: bool = False, steps: int = 500) -> dict:
    if not nb.backend_info()["native_extension_loaded"]:
        raise RuntimeError("neuronbridge native extension is not built")
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = output_dir / "monitor"
    sim = nb.Simulation(build_network(), nb.SimulationConfig(steps=steps, timestep=0.2))
    if enable_zmq:
        sim.add_zmq_async_input_output_spike_driver(
            subscribe_address="127.0.0.1",
            publish_port=ASYNC_SNN_PUB_PORT,
            subscribe_port=ASYNC_ROBOT_PUB_PORT,
            publish_topic=DEFAULT_SNN_OUTPUT_TOPIC,
            subscribe_topic=DEFAULT_ROBOT_INPUT_TOPIC,
            communication_interval=10,
        )
    sim.enable_debug_monitor(
        nb.DebugMonitorConfig(
            output_dir=monitor_dir,
            sample_interval_steps=1,
            flush_interval_steps=100,
            record_state=True,
            record_spikes=True,
            record_pending_channels=False,
            record_weights=False,
            record_outer_dynamic_state=False,
            all_neurons=False,
        )
    )
    sim.init()
    if enable_zmq:
        time.sleep(0.3)
    add_demo_inputs(sim)
    sim.run(steps)
    if enable_zmq:
        sim.publish_output()
        time.sleep(0.2)
    sim.flush()
    output_spikes = sim.output_spikes()
    result = sim.result()
    summary = {
        "steps": steps,
        "enable_zmq": int(enable_zmq),
        "buffered_output_spikes": len(output_spikes),
        "monitor_spike_bytes": result.file_sizes().get("spikes", 0),
        "monitor_state_bytes": result.file_sizes().get("neuron_state", 0),
    }
    with (output_dir / "summary.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("metric\tvalue\n")
        for key, value in summary.items():
            handle.write(f"{key}\t{value}\n")
    return summary


def run_python_test_server(duration_ms: int, log_path: Path) -> int:
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError("pyzmq is required for the Python ZMQ test server; install it as an optional communication dependency") from exc

    context = zmq.Context.instance()
    robot_publisher = context.socket(zmq.PUB)
    snn_subscriber = context.socket(zmq.SUB)
    robot_publisher.linger = 0
    snn_subscriber.linger = 0
    snn_subscriber.setsockopt_string(zmq.SUBSCRIBE, DEFAULT_SNN_OUTPUT_TOPIC)
    robot_publisher.bind(f"tcp://127.0.0.1:{ASYNC_ROBOT_PUB_PORT}")
    snn_subscriber.connect(f"tcp://127.0.0.1:{ASYNC_SNN_PUB_PORT}")

    import time

    log_path.parent.mkdir(parents=True, exist_ok=True)
    received_batches = 0
    start = time.monotonic()
    next_publish = start + 0.3
    time_step = 0
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("server_ready\n")
        while (time.monotonic() - start) * 1000.0 < duration_ms:
            now = time.monotonic()
            if now >= next_publish:
                header, payload = pack_spike_batch(
                    time_step,
                    [
                        SpikeRecord(0, time_step * 0.2, 0.2),
                        SpikeRecord(1, (time_step + 1) * 0.2, 0.2),
                    ],
                )
                robot_publisher.send_string(DEFAULT_ROBOT_INPUT_TOPIC, zmq.SNDMORE)
                robot_publisher.send(header, zmq.SNDMORE)
                robot_publisher.send(payload)
                time_step += 10
                next_publish = now + 0.02
            try:
                topic = snn_subscriber.recv_string(zmq.NOBLOCK)
                header = snn_subscriber.recv()
                if len(header) != HEADER_STRUCT.size:
                    handle.write(f"invalid_header_size={len(header)}\n")
                    continue
                _magic, _version, _batch_step, spike_count = HEADER_STRUCT.unpack(header)
                payload = snn_subscriber.recv() if spike_count > 0 else b""
                batch_step, spikes = unpack_spike_batch(header, payload)
                received_batches += 1
                handle.write(f"received topic={topic} time_step={batch_step} spike_count={len(spikes)}\n")
            except zmq.Again:
                pass
            time.sleep(0.001)
        handle.write(f"server_done received_batches={received_batches}\n")
    robot_publisher.close()
    snn_subscriber.close()
    return received_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=artifact_dir("zmqcommunication_client"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--enable-zmq", action="store_true", help="attach the native async ZMQ driver")
    parser.add_argument("--server", action="store_true", help="run a Python compatibility test server; requires pyzmq")
    parser.add_argument("--duration-ms", type=int, default=15000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.server:
        received = run_python_test_server(args.duration_ms, args.output_dir / "zmqcommunication_test_server.log")
        print(f"received_batches={received}")
        return
    summary = run_example(args.output_dir, enable_zmq=args.enable_zmq, steps=args.steps)
    for key, value in summary.items():
        print(f"{key}={value}")
    print(f"output_dir={args.output_dir}")
    print("zmqcommunication_client Python migration finished.")


if __name__ == "__main__":
    main()
