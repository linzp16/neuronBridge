"""CoppeliaSim-backed controller peer for the NeuronBridge attractor fixture.

This replaces the deterministic AttractorPeer with a Remote API peer.  It is
intentionally optional: CoppeliaSim and its Python zmqRemoteApi client must be
provided by the user because the H:\\attractor tree contains only the C++
Remote API client and no scene file.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import struct
import threading
import time
from pathlib import Path

import zmq

import attractor_wheel_closed_loop as wheel


class CoppeliaAdapter:
    def __init__(self, host: str, port: int, joint_paths: list[str], phases: int,
                 ring_size: int):
        self.client = self._make_client(host, port)
        self.sim = self.client.getObject("sim")
        self.handles = [self.sim.getObject(path) for path in joint_paths]
        self.joint_modes = [self.sim.getJointMode(handle)[0] for handle in self.handles]
        self.end_handle = self.sim.getObject("/MTB/suctionPad/Body")
        self.script_handle = self.sim.getObject("/MTB/Script")
        self.external_control = self.client.getScriptFunctions(self.script_handle)
        # Keep a persistent red world-space trace in the CoppeliaSim scene.
        # A line strip accepts one 3D point per addDrawingObjectItem call.
        self.trajectory_drawing = self.sim.addDrawingObject(
            self.sim.drawing_linestrip,
            3,
            0,
            -1,
            10000,
            [1, 0, 0],
        )
        self.last_end_position: list[float] | None = None
        self.ring_size = ring_size
        self.target_slots: list[list[int]] = [[], []]
        self.ik_angle_ranges: list[tuple[float, float]] = []
        self._phase = 0
        self.trajectory_rows: list[dict] = []
        self.target_points: list[tuple[float, float, float]] = []
        self.simulation_started = False
        # These angles are reference values for phase evaluation and range
        # calibration only. They are not used to drive the joints.
        self.reference_angles = self._plan_figure_eight(phases)
        if phases == 100:
            try:
                import reference_delivery_config as reference
                self.target_slots = [list(reference.RING_TARGET_SLOTS[0]),
                                     list(reference.RING_TARGET_SLOTS[1])]
                self.ik_angle_ranges = [tuple(item) for item in reference.RING_ANGLE_RANGES]
            except ImportError:
                pass
        self._set_start_pose()
        self.sim.startSimulation()
        self.simulation_started = True
        time.sleep(0.3)

    def _set_start_pose(self) -> None:
        """Place both joints at the phase-0 population-code start pose."""
        if not self.target_slots or not all(self.target_slots):
            return
        for ring, handle in enumerate(self.handles):
            low, high = self.ik_angle_ranges[ring]
            slot = self.target_slots[ring][0]
            angle = low + (high - low) * slot / max(1, self.ring_size - 1)
            try:
                self.sim.setJointPosition(handle, angle)
            except Exception:
                # The scene script may require simulation time before direct
                # joint writes; the normal closed-loop command will then take
                # over after startSimulation().
                pass

    def _plan_figure_eight(self, phases: int) -> list[list[float]]:
        """Plan a planar x-y figure-eight using the scene's measured geometry."""
        base = self.sim.getObjectPosition(self.handles[0], self.sim.handle_world)
        elbow = self.sim.getObjectPosition(self.handles[1], self.sim.handle_world)
        end = self.sim.getObjectPosition(self.end_handle, self.sim.handle_world)
        l1 = ((elbow[0] - base[0]) ** 2 + (elbow[1] - base[1]) ** 2) ** 0.5
        l2 = ((end[0] - elbow[0]) ** 2 + (end[1] - elbow[1]) ** 2) ** 0.5
        # Keep the path comfortably inside the measured reachable annulus.
        cx, cy = end[0] - 0.05, end[1]
        rx, ry = 0.07, 0.055
        q1, q2 = [], []
        for index in range(phases):
            theta = 2.0 * 3.141592653589793 * index / phases
            x = cx + rx * __import__("math").sin(theta)
            y = cy + ry * __import__("math").sin(2.0 * theta)
            dx, dy = x - base[0], y - base[1]
            cosine = max(-1.0, min(1.0, (dx * dx + dy * dy - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)))
            angle2 = math.acos(cosine)
            angle1 = math.atan2(dy, dx) - math.atan2(
                l2 * math.sin(angle2), l1 + l2 * math.cos(angle2))
            q1.append(angle1)
            q2.append(angle2)
            self.target_points.append((x, y, end[2]))

        # Quantize the IK solution into ring-neuron slots. The ranges are
        # derived from the complete IK trajectory, not guessed joint limits.
        # Continuous IK angles remain the actuator target; slots are the SNN
        # population-code representation used for each phase.
        self.ik_angle_ranges = []
        self.target_slots = [[], []]
        for ring, angles in enumerate((q1, q2)):
            low, high = min(angles), max(angles)
            if math.isclose(low, high):
                high = low + 1.0
            self.ik_angle_ranges.append((low, high))
            slots = []
            for angle in angles:
                normalized = (angle - low) / (high - low)
                slots.append(max(0, min(self.ring_size - 1,
                                        int(round(normalized * (self.ring_size - 1))))))
            self.target_slots[ring] = slots
        return [q1, q2]

    @staticmethod
    def _make_client(host: str, port: int):
        errors = []
        for module_name in ("coppeliasim_zmqremoteapi_client", "zmqRemoteApi"):
            try:
                module = importlib.import_module(module_name)
                return module.RemoteAPIClient(host=host, port=port)
            except Exception as exc:  # import and constructor diagnostics are reported together
                errors.append(f"{module_name}: {exc}")
        raise RuntimeError(
            "CoppeliaSim Python Remote API is unavailable. Install/copy "
            "coppeliasim_zmqremoteapi_client or add its client directory to PYTHONPATH. "
            + " | ".join(errors)
        )

    def apply_ring_action(self, ring: int, spike_counts: list[int], phase: int | None = None) -> None:
        """Decode the most active ring neuron into an IK joint command."""
        if phase is not None:
            self._phase = int(phase)
        if len(spike_counts) != self.ring_size:
            raise ValueError(f"expected {self.ring_size} ring counts, got {len(spike_counts)}")
        peak = max(spike_counts, default=0)
        if peak <= 0:
            # No activity means no new decoded command; hold the joint.
            winner = -1
        else:
            # Deterministic tie break: retain the previous winner when tied;
            # otherwise choose the first maximum slot.
            candidates = [index for index, count in enumerate(spike_counts) if count == peak]
            previous = getattr(self, "last_winner", [-1, -1])[ring]
            winner = previous if previous in candidates else candidates[0]
        if not hasattr(self, "last_winner"):
            self.last_winner = [-1, -1]
        if winner >= 0:
            self.last_winner[ring] = winner
            low, high = self.ik_angle_ranges[ring]
            command = low + (high - low) * winner / max(1, self.ring_size - 1)
        else:
            command = None
        before = float(self.sim.getJointPosition(self.handles[ring]))
        if command is None:
            next_position = before
        else:
            # Decode the winner into a smooth actuator movement.
            next_position = before + 0.18 * (command - before)
        targets = [float(self.sim.getJointPosition(handle)) for handle in self.handles]
        targets[ring] = next_position
        try:
            self.external_control.setExternalJointTargets(targets)
        except Exception:
            # Keep a direct fallback for an unpatched scene, but the patched
            # scene should always use its actuation callback.
            self.sim.setJointPosition(self.handles[ring], next_position)
        # Let CoppeliaSim propagate the kinematic transform through the MTB
        # hierarchy before sampling the end-effector pose.
        time.sleep(0.02)
        # A readback also flushes the Remote API request queue for passive
        # joints in this scene version.
        self.sim.getJointPosition(self.handles[0])
        self.sim.getJointPosition(self.handles[1])
        end_position = [float(value) for value in self.sim.getObjectPosition(
            self.end_handle, self.sim.handle_world)]
        self.sim.addDrawingObjectItem(self.trajectory_drawing, end_position)
        self.last_end_position = end_position
        self.trajectory_rows.append({
            "time": time.monotonic(), "phase": self._phase, "ring": ring,
            "target_slot": self.target_slots[ring][self._phase % len(self.target_slots[ring])]
            if self.target_slots[ring] else "",
            "decoded_slot": winner,
            "peak_count": peak,
            "ik_angle_min": self.ik_angle_ranges[ring][0],
            "ik_angle_max": self.ik_angle_ranges[ring][1],
            "before": before, "command": next_position if command is None else command,
            "after": next_position,
            "end_x": end_position[0],
            "end_y": end_position[1],
            "end_z": end_position[2],
        })

    def evaluate_phase(self, phase: int) -> bool:
        self._phase = phase
        actual = [float(self.sim.getJointPosition(handle)) for handle in self.handles]
        target = [self.reference_angles[i][phase] for i in range(len(self.handles))]
        error = max(abs(actual[i] - target[i]) for i in range(len(actual)))
        return error <= 0.15

    def stop(self) -> None:
        if self.simulation_started:
            try:
                self.sim.stopSimulation()
            except Exception:
                pass

    def reset_trace(self) -> None:
        """Discard startup settling motion and start a clean measured trace."""
        try:
            self.sim.removeDrawingObject(self.trajectory_drawing)
        except Exception:
            pass
        self.trajectory_drawing = self.sim.addDrawingObject(
            self.sim.drawing_linestrip, 3, 0, -1, 10000, [1, 0, 0])
        self.trajectory_rows.clear()


class CoppeliaPeer:
    def __init__(self, cfg: wheel.FixtureConfig, host: str, port: int,
                 joint_paths: list[str], pub_port: int, sub_port: int, log_path: Path):
        self.cfg = cfg
        self.host = host
        self.port = port
        self.joint_paths = joint_paths
        self.adapter: CoppeliaAdapter | None = None
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.log_path = log_path
        self.ready = threading.Event()
        self.go = threading.Event()
        self.done = threading.Event()
        self.stop_requested = threading.Event()
        self.error: str | None = None
        self.received_batches = 0
        self.received_spikes = 0
        self.feedback_batches = 0
        self.reward_count = 0
        self.punishment_count = 0
        self.reward_by_epoch = [0] * cfg.epochs
        self.punishment_by_epoch = [0] * cfg.epochs
        self.current_phase = 0
        self.ring_activity = [
            [0] * self.cfg.ring_size,
            [0] * self.cfg.ring_size,
        ]
        self.phase_activity = [
            [0] * self.cfg.ring_size,
            [0] * self.cfg.ring_size,
        ]
        self.ring_spike_rows: list[dict] = []
        self.last_phase = -1
        self.last_phase_number = -1
        self.last_control_phase_number = -1
        self.phase_control_count = 0
        # The reference server injects the active phase input every 100
        # simulation steps, beginning 10 steps into each phase.  The REQ/REP
        # protocol allows us to queue the next communication window in one
        # response, so the wheel sees the same repeated drive.
        self.next_input_step = 10
        self.final_feedback_sent = False
        # Settle at phase 0 before recording the trajectory.  Without this
        # pre-roll, the first sample includes the arm's scene-start pose and
        # cannot be compared with the final phase-0 closure pose.
        self.warmup_phases = 10
        # Run several additional phase-0 cycles after the requested epochs.
        # The actuator uses a 0.18 smoothing factor, so one repeated phase is
        # not enough to return from the final phase to the initial pose.
        self.closure_phases = 10
        self.total_steps = (self.warmup_phases * cfg.phase_steps + cfg.steps
                            + self.closure_phases * cfg.phase_steps)
        self.timing = {
            "peer_request_wait_seconds": 0.0,
            "peer_parse_seconds": 0.0,
            "coppeliasim_control_seconds": 0.0,
            "phase_evaluation_seconds": 0.0,
            "reply_send_seconds": 0.0,
            "request_count": 0,
        }

    def _logical_phase(self, phase_number: int) -> int:
        """Map the post-epoch closure cycles back to the phase-0 input."""
        main_start = self.warmup_phases
        closure_start = main_start + self.cfg.epochs * self.cfg.phases
        if phase_number < main_start or phase_number >= closure_start:
            return 0
        return (phase_number - main_start) % self.cfg.phases

    def run(self) -> None:
        ctx = zmq.Context.instance()
        rep = ctx.socket(zmq.REP)
        rep.linger = 0
        rep.setsockopt(zmq.RCVTIMEO, 1000)
        rep.bind("tcp://*:5565")
        self.adapter = CoppeliaAdapter(self.host, self.port, self.joint_paths,
                                       self.cfg.phases, self.cfg.ring_size)
        self.ready.set()
        try:
            self.go.wait(10)
            with self.log_path.open("w", encoding="utf-8") as log:
                while not self.done.is_set():
                    wait_started = time.perf_counter()
                    try:
                        request = rep.recv()
                    except zmq.Again:
                        self.timing["peer_request_wait_seconds"] += time.perf_counter() - wait_started
                        if self.stop_requested.is_set():
                            break
                        continue
                    self.timing["peer_request_wait_seconds"] += time.perf_counter() - wait_started
                    self.timing["request_count"] += 1
                    parse_started = time.perf_counter()
                    if len(request) < 8:
                        rep.send(struct.pack("<I", 0))
                        self.timing["peer_parse_seconds"] += time.perf_counter() - parse_started
                        continue
                    time_step, output_count = struct.unpack_from("<II", request, 0)
                    expected = 8 + output_count * 12
                    if len(request) != expected:
                        rep.send(struct.pack("<I", 0))
                        self.timing["peer_parse_seconds"] += time.perf_counter() - parse_started
                        continue

                    self.received_batches += 1
                    self.received_spikes += output_count
                    self.ring_activity = [
                        [0] * self.cfg.ring_size,
                        [0] * self.cfg.ring_size,
                    ]
                    offset = 8
                    for _ in range(output_count):
                        neuron, spike_time, base_timestep = struct.unpack_from("<iff", request, offset)
                        offset += 12
                        if self.cfg.ring_start <= neuron < self.cfg.ring_start + self.cfg.ring_size:
                            self.ring_activity[0][neuron - self.cfg.ring_start] += 1
                        elif neuron < self.cfg.ring_start + 2 * self.cfg.ring_size:
                            self.ring_activity[1][neuron - self.cfg.ring_start - self.cfg.ring_size] += 1
                        if self.cfg.ring_start <= neuron < self.cfg.ring_start + 2 * self.cfg.ring_size:
                            self.ring_spike_rows.append({
                                "time_step": int(time_step),
                                "neuron_id": int(neuron),
                                "spike_time": float(spike_time),
                                "base_timestep": float(base_timestep),
                            })
                    self.timing["peer_parse_seconds"] += time.perf_counter() - parse_started

                    phase_number = time_step // self.cfg.phase_steps
                    phase = self._logical_phase(phase_number)
                    response = []
                    if phase_number != self.last_phase_number:
                        if self.last_phase_number >= 0:
                            control_started = time.perf_counter()
                            self._finalize_phase_control(self.last_phase_number, log)
                            self.timing["coppeliasim_control_seconds"] += time.perf_counter() - control_started
                        self.phase_activity = [
                            [0] * self.cfg.ring_size,
                            [0] * self.cfg.ring_size,
                        ]
                        self.last_phase_number = phase_number
                        self.last_phase = phase
                        if phase_number == self.warmup_phases:
                            self.adapter.reset_trace()

                    for ring in range(2):
                        for slot in range(self.cfg.ring_size):
                            self.phase_activity[ring][slot] += self.ring_activity[ring][slot]

                    # C++ RStdpWtaServer::generateInputSpikes(): feedback is
                    # produced on every communication for each ring winner.
                    feedback_started = time.perf_counter()
                    for ring in range(2):
                        peak = max(self.ring_activity[ring], default=0)
                        if peak <= 0:
                            continue
                        winner = self.ring_activity[ring].index(peak)
                        target = self.adapter.target_slots[ring][phase]
                        group = phase % self.cfg.input_groups
                        epoch = min(phase_number // self.cfg.phases, self.cfg.epochs - 1)
                        if winner == target:
                            feedback = self.cfg.feedback_start + ring * self.cfg.input_groups + group
                            self.reward_count += 1
                            self.reward_by_epoch[epoch] += 1
                        else:
                            feedback = self.cfg.feedback_start + 2 * self.cfg.input_groups + ring * self.cfg.input_groups + group
                            self.punishment_count += 1
                            self.punishment_by_epoch[epoch] += 1
                        self.feedback_batches += 1
                        response.append(wheel.Spike(feedback, time_step * self.cfg.timestep_ms))
                    self.timing["phase_evaluation_seconds"] += time.perf_counter() - feedback_started

                    input_window_end = min(
                        self.total_steps,
                        time_step + self.cfg.communication_interval,
                    )
                    # The first REQ can arrive after the native simulation
                    # has already advanced several event times. Never queue
                    # an input event in the past; start the first available
                    # reference pulse just after the current native time.
                    if self.next_input_step <= time_step:
                        self.next_input_step = time_step + 10
                    while self.next_input_step < input_window_end:
                        input_phase_number = self.next_input_step // self.cfg.phase_steps
                        input_phase = self._logical_phase(input_phase_number)
                        group = input_phase % self.cfg.input_groups
                        response.append(wheel.Spike(
                            group,
                            self.next_input_step * self.cfg.timestep_ms,
                        ))
                        log.write(
                            f"input phase={input_phase} group={group} "
                            f"step={self.next_input_step}\n"
                        )
                        self.next_input_step += 100

                    send_started = time.perf_counter()
                    self._send_reply(rep, response)
                    self.timing["reply_send_seconds"] += time.perf_counter() - send_started
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
        finally:
            if self.adapter is not None:
                if self.last_phase_number >= 0:
                    final_phase = (self.warmup_phases + self.cfg.epochs * self.cfg.phases
                                   + self.closure_phases - 1)
                    self._finalize_phase_control(final_phase, None)
                self.adapter.stop()
            rep.close()
            self.done.set()

    def _finalize_phase_control(self, phase_number: int, log) -> None:
        """Apply the phase winner after the phase accumulator is complete."""
        if phase_number <= self.last_control_phase_number:
            return
        if self.adapter is None:
            return
        phase = self._logical_phase(phase_number)
        for ring in range(2):
            counts = self.phase_activity[ring]
            peak = max(counts, default=0)
            if peak <= 0:
                continue
            winner = counts.index(peak)
            self.adapter.apply_ring_action(ring, counts, phase=phase)
            if log is not None:
                log.write(f"phase_control phase={phase} ring={ring} winner={winner}\n")
        self.last_control_phase_number = phase_number
        self.phase_control_count += 1

    def _feedback_spike(self, phase: int, time_step: int, log, epoch: int):
        assert self.adapter is not None
        correct = self.adapter.evaluate_phase(phase)
        ring = phase % 2
        group = phase % self.cfg.input_groups
        if correct:
            feedback = self.cfg.feedback_start + ring * self.cfg.input_groups + group
            self.reward_count += 1
            self.reward_by_epoch[epoch] += 1
            kind = "reward"
        else:
            feedback = self.cfg.feedback_start + 2 * self.cfg.input_groups + ring * self.cfg.input_groups + group
            self.punishment_count += 1
            self.punishment_by_epoch[epoch] += 1
            kind = "punishment"
        self.feedback_batches += 1
        log.write(f"feedback phase={phase} kind={kind} neuron={feedback} step={time_step}\n")
        return wheel.Spike(feedback, time_step * self.cfg.timestep_ms)

    @staticmethod
    def _send_reply(rep, spikes) -> None:
        rep.send(struct.pack("<I", len(spikes)), zmq.SNDMORE if spikes else 0)
        if spikes:
            payload = b"".join(struct.pack("<iff", spike.neuron, spike.time, spike.base_timestep)
                                for spike in spikes)
            rep.send(payload)


def run(args: argparse.Namespace) -> dict:
    if args.reference_delivery:
        import reference_delivery_config as reference
        cfg = wheel.FixtureConfig(
            input_groups=reference.INPUT_GROUPS,
            phases=reference.PHASES,
            ring_size=reference.RING_SIZE,
            phase_steps=reference.PHASE_STEPS,
            communication_interval=reference.COMMUNICATION_INTERVAL,
            timestep_ms=reference.TIMESTEP_MS,
            epochs=args.epochs,
        )
    else:
        cfg = wheel.FixtureConfig(
            phases=args.phases,
            communication_interval=args.communication_interval,
            epochs=args.epochs,
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    peer = CoppeliaPeer(cfg, args.host, args.port, args.joint_path,
                        5566, 5567, output_dir / "coppelia_peer.log")
    thread = threading.Thread(target=peer.run, daemon=True)
    thread.start()
    if not peer.ready.wait(10):
        raise RuntimeError("Coppelia peer did not become ready")
    if peer.error:
        raise RuntimeError(peer.error)
    sim = wheel.nb.Simulation(
        wheel.build_network(cfg),
        wheel.nb.SimulationConfig(steps=peer.total_steps, timestep=cfg.timestep_ms, queues=2,
                                  event_queue="timing_wheel", timing_wheel_size=128),
    )
    # The original attractor uses blocking REQ/REP. The native wheel now
    # exposes that driver, so each communication event pauses here until the
    # CoppeliaSim peer returns its input/reward response.
    sim.add_zmq_input_output_spike_driver(
        server_address="127.0.0.1",
        server_port=5565,
        communication_interval=cfg.communication_interval,
    )
    # Explicitly monitor the two attractor rings. Their global IDs are
    # computed from the configured ring size.
    # The simulation is run in communication-sized chunks below because the
    # native DebugMonitor captures at the end of each Simulation::run call.
    monitor_dir = output_dir / "debug_monitor_ring"
    ring_neuron_ids = list(range(cfg.ring_start, cfg.ring_start + 2 * cfg.ring_size))
    sim.enable_debug_monitor(
        wheel.nb.DebugMonitorConfig(
            output_dir=monitor_dir,
            sample_interval_steps=1,
            flush_interval_steps=cfg.communication_interval,
            record_spikes=True,
            record_state=False,
            record_weights=False,
            record_pending_channels=False,
            record_outer_dynamic_state=False,
            all_neurons=False,
            neuron_ids=ring_neuron_ids,
        )
    )
    sim.init()
    if args.load_weights is not None:
        sim.load_weights(args.load_weights)
    peer.go.set()
    time.sleep(0.4)
    total_started = time.perf_counter()
    sim_run_started = time.perf_counter()
    remaining = peer.total_steps
    while remaining > 0:
        chunk = min(cfg.communication_interval, remaining)
        sim.run(chunk)
        remaining -= chunk
        sim.flush()
    sim_run_seconds = time.perf_counter() - sim_run_started
    sim.publish_output()
    sim.flush()
    peer.stop_requested.set()
    thread.join(10)
    result = {
        "pass": peer.error is None and peer.phase_control_count == (
            peer.warmup_phases + cfg.phases * cfg.epochs + peer.closure_phases),
        "received_output_batches": peer.received_batches,
        "received_output_spikes": peer.received_spikes,
        "feedback_batches": peer.feedback_batches,
        "phase_control_count": peer.phase_control_count,
        "rewards": peer.reward_count,
        "punishments": peer.punishment_count,
        "rewards_by_epoch": peer.reward_by_epoch,
        "punishments_by_epoch": peer.punishment_by_epoch,
        "peer_error": peer.error,
        "trajectory_samples": len(peer.adapter.trajectory_rows) if peer.adapter else 0,
        "debug_monitor_dir": str(monitor_dir),
        "debug_monitor_neuron_ids": [ring_neuron_ids[0], ring_neuron_ids[-1]],
        "trajectory_max_step": max(
            (abs(row["after"] - row["before"])
             for row in (peer.adapter.trajectory_rows if peer.adapter else [])),
            default=0.0,
        ),
        "timing_seconds": {
            "total_run_wall_seconds": time.perf_counter() - total_started,
            "native_sim_run_seconds": sim_run_seconds,
            **peer.timing,
        },
    }
    with (output_dir / "joint_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "time", "phase", "ring", "target_slot", "ik_angle_min",
            "ik_angle_max", "decoded_slot", "peak_count", "before", "command", "after", "end_x",
            "end_y", "end_z"])
        writer.writeheader()
        writer.writerows(peer.adapter.trajectory_rows if peer.adapter else [])
    with (output_dir / "ring_spikes_from_zmq.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "time_step", "neuron_id", "spike_time", "base_timestep"])
        writer.writeheader()
        writer.writerows(peer.ring_spike_rows)
    result["ring_spikes_from_zmq"] = len(peer.ring_spike_rows)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--phases", type=int, default=64)
    parser.add_argument("--communication-interval", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--reference-delivery", action="store_true")
    parser.add_argument("--load-weights", type=Path, default=None)
    parser.add_argument("--joint-path", action="append", default=["/MTB/axis", "/MTB/link/axis"])
    parser.add_argument("--target-angles", default=None, help="optional JSON joint targets; omit to use planned figure-eight")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
