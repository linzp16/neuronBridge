from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import neuronbridge as nb

import attractor_wheel_closed_loop as fixture


def main() -> None:
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = fixture.FixtureConfig(phases=64)
    network = fixture.build_network(cfg)
    sim = nb.Simulation(
        network,
        nb.SimulationConfig(
            steps=cfg.steps,
            timestep=cfg.timestep_ms,
            queues=2,
            event_queue="timing_wheel",
            timing_wheel_size=128,
        ),
    )

    times: list[int] = []
    neuron_ids: list[int] = []
    for phase in range(cfg.phases):
        start = phase * cfg.phase_steps
        group = phase % cfg.input_groups
        times.append(start + 10)
        neuron_ids.append(group)
        ring = phase % 2
        feedback = cfg.feedback_start + (
            ring * cfg.input_groups + group
            if phase % 2 == 0
            else 2 * cfg.input_groups + ring * cfg.input_groups + group
        )
        times.append(start + cfg.reward_delay_steps)
        neuron_ids.append(feedback)

    sim.init()
    sim.add_external_spikes(times, neuron_ids)
    sim.reset_bench_profiling()
    started = time.perf_counter()
    sim.run(cfg.steps)
    wall_seconds = time.perf_counter() - started
    snapshot = sim.bench_profiling_snapshot()

    def ms(name: str) -> float:
        return float(snapshot.get(name, 0)) / 1_000_000.0

    report = {
        "pass": True,
        "steps": cfg.steps,
        "neurons": network.neuron_count,
        "connections": len(network.connections),
        "learning_rules": len(network.learning_rules),
        "wall_seconds": wall_seconds,
        "snapshot": snapshot,
        "derived_ms": {
            "event_queue_scheduling": ms("event_remove_ns") + ms("wheel_insert_ns") + ms("wheel_remove_ns") + ms("wheel_advance_queue_ns") + ms("wheel_migrate_queue_ns") + ms("wheel_pop_ready_ns"),
            "event_processing": ms("run_step_process_event_ns"),
            "neuron_state_update": ms("time_event_update_state_ns"),
            "synaptic_propagation": ms("internal_spike_write_spike_ns") + ms("internal_spike_include_ns") + ms("internal_spike_insert_ready_ns") + ms("internal_spike_rotate_group_ns") + ms("internal_spike_finalize_group_ns"),
            "learning_rule_update": ms("internal_spike_learning_ns"),
            "time_event_reschedule": ms("time_event_reschedule_ns"),
        },
    }
    (output_dir / "native_internal_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
