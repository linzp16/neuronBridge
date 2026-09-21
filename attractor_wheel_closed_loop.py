"""Portable wheel-side closed loop for the H:\\attractor R-STDP ring task.

The original project refers to generated target/configuration headers that are
not present in the supplied H:\\attractor tree.  This fixture therefore keeps
the original topology and protocol, while making the missing target sequence
explicit and reproducible in JSON.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import threading
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import neuronbridge as nb


MAGIC = 0x53504B31
VERSION = 1
HEADER = struct.Struct("<IIII")
SPIKE = struct.Struct("<iff")
TOPIC_OUT = "/snn/output_spikes"
TOPIC_IN = "/robot/input_spikes"


@dataclass(frozen=True)
class Spike:
    neuron: int
    time: float
    base_timestep: float = 0.1


def pack_batch(step: int, spikes: list[Spike]) -> tuple[bytes, bytes]:
    return (
        HEADER.pack(MAGIC, VERSION, int(step), len(spikes)),
        b"".join(SPIKE.pack(s.neuron, s.time, s.base_timestep) for s in spikes),
    )


def unpack_batch(header: bytes, payload: bytes) -> tuple[int, list[Spike]]:
    magic, version, step, count = HEADER.unpack(header)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"invalid spike protocol magic/version: {magic:#x}/{version}")
    if len(payload) != count * SPIKE.size:
        raise ValueError(f"invalid payload size {len(payload)} for {count} spikes")
    return step, [Spike(*SPIKE.unpack_from(payload, i)) for i in range(0, len(payload), SPIKE.size)]


@dataclass
class FixtureConfig:
    # Restored in Python; each input neuron owns exactly one phase.
    input_groups: int = 64
    ring_size: int = 32
    phases: int = 64
    phase_steps: int = 10000
    timestep_ms: float = 0.1
    communication_interval: int = 1000
    reward_delay_steps: int = 80
    input_weight_random_seed: int = 20260425
    input_weight_mean: float = 1.0
    epochs: int = 1

    def __post_init__(self) -> None:
        if self.phases != self.input_groups:
            raise ValueError("phases must equal input_groups: one input neuron per phase")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

    @property
    def ring_start(self) -> int:
        return self.input_groups

    @property
    def ring_count(self) -> int:
        return 2 * self.ring_size

    @property
    def feedback_start(self) -> int:
        return self.ring_start + self.ring_count

    @property
    def feedback_count(self) -> int:
        return self.input_groups * 2 * 2

    @property
    def steps(self) -> int:
        return self.epochs * self.phases * self.phase_steps + self.reward_delay_steps


def build_network(cfg: FixtureConfig) -> nb.Network:
    n = nb.Network()
    n.add_layer(nb.NeuronLayer("InputSpikeNeuronModel", cfg.input_groups,
                                monitored=True, communication_input=True))
    n.add_layer(
        nb.NeuronLayer(
            "TimeDrivenLIF_Exponential_double",
            cfg.ring_count,
            monitored=True,
            output=True,
            parameters={
                "V_rest": nb.float32(-60.0), "V_reset": nb.float32(-60.0),
                "V_th": nb.float32(-40.0), "tau": nb.float32(15.0),
                "R": nb.float32(1.0), "t_ref": nb.int32(60),
                "gexc_tau": nb.float32(8.0), "ginh_tau": nb.float32(15.0),
                "Eexc": nb.float32(0.0), "Einh": nb.float32(-80.0),
            },
        )
    )
    n.add_layer(nb.NeuronLayer("InputSpikeNeuronModel", cfg.feedback_count))
    learning_parameters = {
        "Max_LTP": nb.float32(0.08),
        "Max_LTD": nb.float32(0.05),
        "LTP_tau": nb.float32(20.0),
        "LTD_tau": nb.float32(30.0),
        "RewardFactor": nb.float32(1.0),
        "PunishmentFactor": nb.float32(-1.0),
        "ClearEligibilityAfterTrigger": True,
    }
    # One R-STDP rule for each input-neuron/ring pair, matching the 5 x 2
    # rule layout in the original C++ source.
    for _ in range(cfg.input_groups * 2):
        n.add_learning_rule(nb.LearningRule("R_STDP", parameters=learning_parameters))

    # Match the reference project's seeded input-weight jitter.  The input
    # still projects to every neuron in both rings, but exact symmetry is
    # intentionally broken so the ring can form a winner.
    rng = random.Random(cfg.input_weight_random_seed)
    for source in range(cfg.input_groups):
        for ring in range(2):
            rule = source * 2 + ring
            for target in range(ring * cfg.ring_size, (ring + 1) * cfg.ring_size):
                n.connect(nb.Connection(source, cfg.ring_start + target, synapse_type=0,
                                        weight=cfg.input_weight_mean + rng.uniform(-0.35, 0.35),
                                        max_weight=8.0, delay=1,
                                        synapse_rule=rule))

    # Distance-dependent inhibitory connections within each ring; no
    # self-connections, matching AddRingInhibitoryConnections().
    for ring in range(2):
        start = cfg.ring_start + ring * cfg.ring_size
        for source_slot in range(cfg.ring_size):
            for target_slot in range(cfg.ring_size):
                if source_slot == target_slot:
                    continue
                distance = min(abs(source_slot - target_slot),
                               cfg.ring_size - abs(source_slot - target_slot))
                normalized_distance = distance / max(1, cfg.ring_size // 2)
                weight = 1.2 + (5.0 - 1.2) * normalized_distance * normalized_distance
                n.connect(nb.Connection(start + source_slot, start + target_slot,
                                        synapse_type=1, weight=weight,
                                        max_weight=weight, delay=1))
    # First feedback block is reward (excitatory), second is punishment
    # (inhibitory). Each group/ring pair owns one trigger channel.
    for ring in range(2):
        for group in range(cfg.input_groups):
            rule = group * 2 + ring
            reward_id = cfg.feedback_start + ring * cfg.input_groups + group
            punishment_id = cfg.feedback_start + 2 * cfg.input_groups + ring * cfg.input_groups + group
            for target in range(ring * cfg.ring_size, (ring + 1) * cfg.ring_size):
                n.connect(nb.Connection(reward_id, cfg.ring_start + target, synapse_type=0,
                                        weight=0.0, max_weight=0.0, delay=1, trigger_rule=rule))
                n.connect(nb.Connection(punishment_id, cfg.ring_start + target, synapse_type=1,
                                        weight=0.0, max_weight=0.0, delay=1, trigger_rule=rule))
    return n


class AttractorPeer:
    def __init__(self, cfg: FixtureConfig, pub_port: int, sub_port: int, log_path: Path):
        self.cfg = cfg
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.log_path = log_path
        self.ready = threading.Event()
        self.go = threading.Event()
        self.done = threading.Event()
        self.error: str | None = None
        self.received_batches = 0
        self.received_spikes = 0
        self.feedback_batches = 0
        self.feedback_spikes = 0
        self.output_by_ring = Counter()

    def run(self) -> None:
        try:
            import zmq
            ctx = zmq.Context.instance()
            pub = ctx.socket(zmq.PUB)
            sub = ctx.socket(zmq.SUB)
            pub.linger = 0
            sub.linger = 0
            sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_OUT)
            pub.bind(f"tcp://127.0.0.1:{self.pub_port}")
            sub.connect(f"tcp://127.0.0.1:{self.sub_port}")
            self.ready.set()
            # Let both PUB/SUB handshakes settle before the first stimulus.
            self.go.wait(5)
            time.sleep(0.15)
            with self.log_path.open("w", encoding="utf-8") as log:
                log.write("peer_ready\n")
                for phase in range(self.cfg.phases):
                    start = phase * self.cfg.phase_steps
                    group = phase % self.cfg.input_groups
                    for offset in range(10, self.cfg.phase_steps, 40):
                        step = start + offset
                        h, p = pack_batch(step, [Spike(group, step * self.cfg.timestep_ms)])
                        pub.send_string(TOPIC_IN, zmq.SNDMORE)
                        pub.send(h, zmq.SNDMORE)
                        pub.send(p)
                        self._drain(sub, log, zmq)
                        time.sleep(0.004)
                    log.write(f"input phase={phase} group={group} step={start}\n")
                    ring = phase % 2
                    is_reward = (phase % 2) == 0
                    feedback = self.cfg.feedback_start + (
                        ring * self.cfg.input_groups + group
                        if is_reward else
                        2 * self.cfg.input_groups + ring * self.cfg.input_groups + group
                    )
                    fh, fp = pack_batch(start + self.cfg.reward_delay_steps,
                                        [Spike(feedback, (start + self.cfg.reward_delay_steps) * self.cfg.timestep_ms)])
                    pub.send_string(TOPIC_IN, zmq.SNDMORE)
                    pub.send(fh, zmq.SNDMORE)
                    pub.send(fp)
                    self.feedback_batches += 1
                    self.feedback_spikes += 1
                    log.write(f"feedback phase={phase} kind={'reward' if is_reward else 'punishment'} neuron={feedback}\n")
                # Continue consuming output long enough to observe final feedback.
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    self._drain(sub, log, zmq)
                    time.sleep(0.001)
            pub.close()
            sub.close()
        except Exception as exc:  # pragma: no cover - surfaced in result JSON
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            self.done.set()

    def _drain(self, sub, log, zmq) -> None:
        while True:
            try:
                topic = sub.recv_string(zmq.NOBLOCK)
                header = sub.recv()
                payload = sub.recv() if HEADER.unpack(header)[3] else b""
            except zmq.Again:
                return
            step, spikes = unpack_batch(header, payload)
            self.received_batches += 1
            self.received_spikes += len(spikes)
            for spike in spikes:
                ring = 0 if spike.neuron < self.cfg.ring_start + self.cfg.ring_size else 1
                self.output_by_ring[ring] += 1
            log.write(f"output topic={topic} step={step} spikes={len(spikes)}\n")


def run(output_dir: Path, cfg: FixtureConfig) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg) | {
        "ring_start": cfg.ring_start, "feedback_start": cfg.feedback_start,
        "feedback_count": cfg.feedback_count, "steps": cfg.steps,
    }, indent=2), encoding="utf-8")
    peer = AttractorPeer(cfg, pub_port=5566, sub_port=5567, log_path=output_dir / "peer.log")
    thread = threading.Thread(target=peer.run, name="attractor-peer", daemon=True)
    thread.start()
    if not peer.ready.wait(5):
        raise RuntimeError("peer did not become ready")
    if peer.error:
        raise RuntimeError(peer.error)
    sim = nb.Simulation(build_network(cfg), nb.SimulationConfig(
        steps=cfg.steps, timestep=cfg.timestep_ms, queues=2,
        event_queue="timing_wheel", timing_wheel_size=128,
    ))
    sim.add_zmq_async_input_output_spike_driver(
        subscribe_address="127.0.0.1", publish_port=5567, subscribe_port=5566,
        publish_topic=TOPIC_OUT, subscribe_topic=TOPIC_IN,
        communication_interval=cfg.communication_interval,
    )
    sim.init()
    peer.go.set()
    # Give the PUB/SUB peer time to enqueue the complete staged input stream.
    # The native driver consumes those timestamped batches at communication
    # events, so this does not advance simulation time or alter causality.
    time.sleep(0.4)
    weight_before = sim.get_connection_weight(0)
    sim.run(cfg.steps)
    sim.publish_output()
    sim.flush()
    weight_after = sim.get_connection_weight(0)
    thread.join(timeout=5)
    if thread.is_alive():
        raise RuntimeError("peer thread did not finish")
    result = {
        "backend": nb.backend_info(),
        "config": asdict(cfg),
        "network_neurons": cfg.input_groups + cfg.ring_count + cfg.feedback_count,
        "simulation_steps": cfg.steps,
        "received_output_batches": peer.received_batches,
        "received_output_spikes": peer.received_spikes,
        "feedback_batches": peer.feedback_batches,
        "feedback_spikes": peer.feedback_spikes,
        "output_by_ring": dict(peer.output_by_ring),
        "plastic_weight_before": weight_before,
        "plastic_weight_after": weight_after,
        "plastic_weight_delta": weight_after - weight_before,
        "peer_error": peer.error,
        "pass": peer.error is None and peer.feedback_batches == cfg.phases,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, FixtureConfig()), indent=2, default=str))


if __name__ == "__main__":
    main()
