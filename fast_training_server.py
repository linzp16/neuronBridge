"""Mechanical-arm-free REQ/REP trainer for the dual-ring attractor.

The protocol is identical to the Coppelia peer: the wheel sends a packed
output-spike batch and receives a packed input-spike batch containing the
next phase inputs and reward/punishment trigger spikes.
"""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import zmq

import attractor_wheel_closed_loop as wheel


class FastTrainingServer:
    def __init__(self, cfg: wheel.FixtureConfig, target_path: Path | None,
                 port: int = 5565, learning_enabled: bool = True):
        self.cfg = cfg
        self.target_path = target_path
        self.port = port
        self.learning_enabled = learning_enabled
        if target_path is None:
            import reference_delivery_config as reference
            data = {
                "ring0_target_slots": reference.RING0_TARGET_SLOTS_ONE_BASED,
                "ring1_target_slots": reference.RING1_TARGET_SLOTS_ONE_BASED,
                "ik_angle_ranges": reference.RING_ANGLE_RANGES,
            }
        else:
            data = json.loads(target_path.read_text(encoding="utf-8"))
        self.target_slots = [
            [int(value) - 1 for value in data["ring0_target_slots"]],
            [int(value) - 1 for value in data["ring1_target_slots"]],
        ]
        for ring in self.target_slots:
            if len(ring) != cfg.phases or any(value < 0 or value >= cfg.ring_size for value in ring):
                raise ValueError("target slot sequences must contain one valid slot per phase")
        ranges = data.get("ik_angle_ranges")
        if ranges is None:
            ranges = [[0.0, float(cfg.ring_size - 1)] for _ in range(2)]
        self.ik_angle_ranges = [(float(item[0]), float(item[1])) for item in ranges]
        self.target_angles = [
            [low + (high - low) * slot / max(1, cfg.ring_size - 1)
             for slot in slots]
            for slots, (low, high) in zip(self.target_slots, self.ik_angle_ranges)
        ]
        self.joint_positions = [angles[0] for angles in self.target_angles]
        self.ready = threading.Event()
        self.go = threading.Event()
        self.stop_requested = threading.Event()
        self.error: str | None = None
        self.received_batches = 0
        self.received_spikes = 0
        self.feedback_batches = 0
        self.rewards = 0
        self.punishments = 0
        self.rewards_by_epoch = [0] * cfg.epochs
        self.punishments_by_epoch = [0] * cfg.epochs
        self.phase_control_correct_by_epoch = [0] * cfg.epochs
        self.window_winners: list[dict] = []
        self.phase_decisions: list[dict] = []
        self.winners_by_epoch: list[list[list[int]]] = [
            [[], []] for _ in range(cfg.epochs)
        ]
        self.window_counts = [[0] * cfg.ring_size for _ in range(2)]
        self.phase_counts = [[0] * cfg.ring_size for _ in range(2)]
        self.last_finalized_phase = -1
        self.phase_control_count = 0
        self.next_input_step = 10
        self.last_phase_number = -1

    def run(self) -> None:
        context = zmq.Context.instance()
        rep = context.socket(zmq.REP)
        rep.linger = 0
        rep.setsockopt(zmq.RCVTIMEO, 1000)
        rep.bind(f"tcp://*:{self.port}")
        self.ready.set()
        try:
            self.go.wait(10)
            while not self.stop_requested.is_set():
                try:
                    request = rep.recv()
                except zmq.Again:
                    continue
                if len(request) < 8:
                    self._send_reply(rep, [])
                    continue
                time_step, output_count = struct.unpack_from("<II", request, 0)
                if len(request) != 8 + output_count * 12:
                    self._send_reply(rep, [])
                    continue

                counts = [[0] * self.cfg.ring_size for _ in range(2)]
                offset = 8
                for _ in range(output_count):
                    neuron, _, _ = struct.unpack_from("<iff", request, offset)
                    offset += 12
                    if self.cfg.ring_start <= neuron < self.cfg.ring_start + self.cfg.ring_count:
                        relative = neuron - self.cfg.ring_start
                        ring = relative // self.cfg.ring_size
                        slot = relative % self.cfg.ring_size
                        counts[ring][slot] += 1
                self.received_batches += 1
                self.received_spikes += output_count
                phase_number = time_step // self.cfg.phase_steps
                response: list[wheel.Spike] = []
                if phase_number != self.last_phase_number:
                    if self.last_phase_number >= 0:
                        self._finish_phase(self.last_phase_number, time_step)
                    self.phase_counts = [[0] * self.cfg.ring_size for _ in range(2)]
                    self.last_phase_number = phase_number

                for ring in range(2):
                    for slot in range(self.cfg.ring_size):
                        self.phase_counts[ring][slot] += counts[ring][slot]

                # Match RStdpWtaServer::generateInputSpikes(): one feedback
                # decision per ring, per communication request, based on the
                # current request window rather than the phase accumulator.
                phase = phase_number % self.cfg.phases
                group = phase % self.cfg.input_groups
                for ring in range(2):
                    peak = max(counts[ring], default=0)
                    if peak <= 0:
                        continue
                    winner = counts[ring].index(peak)
                    target = self.target_slots[ring][phase]
                    self.window_winners.append({
                        "epoch": min(phase_number // self.cfg.phases, self.cfg.epochs - 1) + 1,
                        "phase": phase,
                        "time_step": time_step,
                        "ring": ring,
                        "winner": winner,
                        "target": target,
                        "spike_count": peak,
                        "correct": int(winner == target),
                    })
                    if winner == target:
                        neuron = self.cfg.feedback_start + ring * self.cfg.input_groups + group
                        if self.learning_enabled:
                            self.rewards += 1
                            self.rewards_by_epoch[min(phase_number // self.cfg.phases, self.cfg.epochs - 1)] += 1
                    else:
                        neuron = (self.cfg.feedback_start + 2 * self.cfg.input_groups
                                  + ring * self.cfg.input_groups + group)
                        if self.learning_enabled:
                            self.punishments += 1
                            self.punishments_by_epoch[min(phase_number // self.cfg.phases, self.cfg.epochs - 1)] += 1
                    if self.learning_enabled:
                        self.feedback_batches += 1
                        response.append(wheel.Spike(neuron, time_step * self.cfg.timestep_ms))

                end_step = min(self.cfg.steps, time_step + self.cfg.communication_interval)
                if self.next_input_step <= time_step:
                    self.next_input_step = time_step + 10
                while self.next_input_step < end_step:
                    input_phase = (self.next_input_step // self.cfg.phase_steps) % self.cfg.phases
                    response.append(wheel.Spike(
                        input_phase % self.cfg.input_groups,
                        self.next_input_step * self.cfg.timestep_ms,
                    ))
                    self.next_input_step += 100
                final_phase_start = self.cfg.epochs * self.cfg.phases * self.cfg.phase_steps - self.cfg.phase_steps
                if (time_step + self.cfg.communication_interval >=
                        self.cfg.epochs * self.cfg.phases * self.cfg.phase_steps and
                        time_step >= final_phase_start):
                    self._finish_phase(self.cfg.epochs * self.cfg.phases - 1, time_step)
                self._send_reply(rep, response)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            rep.close()

    def _finish_phase(self, phase_number: int, time_step: int) -> None:
        completed_phase = phase_number
        if completed_phase < 0 or completed_phase <= self.last_finalized_phase:
            return
        self.last_finalized_phase = completed_phase
        epoch = min(completed_phase // self.cfg.phases, self.cfg.epochs - 1)
        phase = completed_phase % self.cfg.phases
        for ring in range(2):
            counts = self.phase_counts[ring]
            peak = max(counts, default=0)
            winner = counts.index(peak) if peak > 0 else -1
            self.winners_by_epoch[epoch][ring].append(winner)
            if winner >= 0:
                low, high = self.ik_angle_ranges[ring]
                command = low + (high - low) * winner / max(1, self.cfg.ring_size - 1)
                self.joint_positions[ring] = command
        phase_correct = all(
            self.winners_by_epoch[epoch][ring][-1] == self.target_slots[ring][phase]
            for ring in range(2)
        )
        if phase_correct:
            self.phase_control_correct_by_epoch[epoch] += 1
        self.phase_decisions.append({
            "epoch": epoch + 1,
            "phase": phase,
            "ring0_winner": self.winners_by_epoch[epoch][0][-1],
            "ring0_target": self.target_slots[0][phase],
            "ring1_winner": self.winners_by_epoch[epoch][1][-1],
            "ring1_target": self.target_slots[1][phase],
            "both_correct": int(phase_correct),
        })
        self.phase_control_count += 1

    @staticmethod
    def _send_reply(rep, spikes: list[wheel.Spike]) -> None:
        rep.send(struct.pack("<I", len(spikes)), zmq.SNDMORE if spikes else 0)
        if spikes:
            payload = b"".join(
                struct.pack("<iff", spike.neuron, spike.time, spike.base_timestep)
                for spike in spikes
            )
            rep.send(payload)
