"""Fast multi-epoch dual-ring training without CoppeliaSim or robot I/O."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path

import attractor_wheel_closed_loop as wheel
from fast_training_server import FastTrainingServer


def run(args: argparse.Namespace) -> dict:
    target_path = args.target_slots
    if args.reference_delivery:
        import reference_delivery_config as reference
        cfg = wheel.FixtureConfig(
            input_groups=reference.INPUT_GROUPS,
            phases=reference.PHASES,
            ring_size=reference.RING_SIZE,
            phase_steps=args.phase_steps if args.phase_steps is not None else reference.PHASE_STEPS,
            communication_interval=reference.COMMUNICATION_INTERVAL,
            timestep_ms=reference.TIMESTEP_MS,
            input_weight_mean=args.input_weight_mean,
            epochs=args.epochs,
        )
        target_path = None
    else:
        cfg = wheel.FixtureConfig(
            phases=args.phases,
            epochs=args.epochs,
            phase_steps=args.phase_steps if args.phase_steps is not None else 10000,
            communication_interval=args.communication_interval,
            input_weight_mean=args.input_weight_mean,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server = FastTrainingServer(
        cfg, target_path, args.port, learning_enabled=not args.disable_learning
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    if not server.ready.wait(10):
        raise RuntimeError("fast training server did not become ready")
    if server.error:
        raise RuntimeError(server.error)

    sim = wheel.nb.Simulation(
        wheel.build_network(cfg),
        wheel.nb.SimulationConfig(
            steps=cfg.steps,
            timestep=cfg.timestep_ms,
            queues=2,
            event_queue="timing_wheel",
            timing_wheel_size=128,
        ),
    )
    sim.add_zmq_input_output_spike_driver(
        server_address="127.0.0.1",
        server_port=args.port,
        communication_interval=cfg.communication_interval,
    )
    sim.init()
    if args.load_weights is not None:
        sim.load_weights(args.load_weights)
    server.go.set()
    time.sleep(0.1)
    started = time.perf_counter()
    remaining = cfg.steps
    while remaining > 0:
        chunk = min(cfg.communication_interval, remaining)
        sim.run(chunk)
        remaining -= chunk
    sim.publish_output()
    weights_path = args.output_dir / "final_weights.dat"
    sim.save_weights(weights_path)
    sim.flush()
    server.stop_requested.set()
    server_thread.join(10)

    result = {
        "pass": server.error is None and server.phase_control_count == cfg.phases * cfg.epochs,
        "epochs": cfg.epochs,
        "phases": cfg.phases,
        "ring_size": cfg.ring_size,
        "communication_interval": cfg.communication_interval,
        "received_output_batches": server.received_batches,
        "received_output_spikes": server.received_spikes,
        "feedback_batches": server.feedback_batches,
        "phase_control_count": server.phase_control_count,
        "rewards": server.rewards,
        "punishments": server.punishments,
        "rewards_by_epoch": server.rewards_by_epoch,
        "punishments_by_epoch": server.punishments_by_epoch,
        "phase_control_correct_by_epoch": server.phase_control_correct_by_epoch,
        "winners_by_epoch": server.winners_by_epoch,
        "peer_error": server.error,
        "weights_path": str(weights_path),
        "wall_seconds": time.perf_counter() - started,
        "target_slots_path": str(target_path) if target_path is not None else "reference_delivery_config.py",
    }
    epoch_metrics = []
    for index, (rewards, punishments) in enumerate(
        zip(server.rewards_by_epoch, server.punishments_by_epoch)
    ):
        total = rewards + punishments
        phase_correct = server.phase_control_correct_by_epoch[index]
        epoch_metrics.append({
            "epoch": index + 1,
            "feedback_rewards": rewards,
            "feedback_punishments": punishments,
            "feedback_decisions": total,
            "feedback_reward_rate": rewards / total if total else 0.0,
            "phase_control_correct": phase_correct,
            "phase_decisions": cfg.phases,
            "phase_control_accuracy": phase_correct / cfg.phases,
        })
    result["epoch_metrics"] = epoch_metrics
    with (args.output_dir / "epoch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "feedback_rewards", "feedback_punishments",
            "feedback_decisions", "feedback_reward_rate",
            "phase_control_correct", "phase_decisions", "phase_control_accuracy"
        ])
        writer.writeheader()
        writer.writerows(epoch_metrics)
    with (args.output_dir / "window_winners.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["epoch", "phase", "time_step", "ring", "winner",
                  "target", "spike_count", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(server.window_winners)
    with (args.output_dir / "phase_decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["epoch", "phase", "ring0_winner", "ring0_target",
                  "ring1_winner", "ring1_target", "both_correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(server.phase_decisions)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-slots", type=Path, default=None)
    parser.add_argument("--reference-delivery", action="store_true")
    parser.add_argument("--load-weights", type=Path, default=None)
    parser.add_argument("--input-weight-mean", type=float, default=1.0)
    parser.add_argument(
        "--disable-learning", action="store_true",
        help="Do not send reward/punishment trigger spikes; keep initial weights fixed.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--phase-steps", type=int, default=None)
    parser.add_argument("--phases", type=int, default=64)
    parser.add_argument("--communication-interval", type=int, default=2000)
    parser.add_argument("--port", type=int, default=5565)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
