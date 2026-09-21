"""Local closed-loop trainer used when the installed wheel lacks sync ZMQ."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import neuronbridge as nb
import attractor_wheel_closed_loop as wheel
import reference_delivery_config as ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-weight-mean', type=float, default=1.5)
    ap.add_argument('--phase-steps', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = wheel.FixtureConfig(input_groups=100, phases=100, ring_size=32,
                              phase_steps=args.phase_steps,
                              communication_interval=1000,
                              timestep_ms=0.1,
                              input_weight_mean=args.input_weight_mean,
                              epochs=args.epochs)
    net = wheel.build_network(cfg)
    sim = nb.Simulation(net, nb.SimulationConfig(
        steps=cfg.steps, timestep=cfg.timestep_ms, queues=2,
        event_queue='timing_wheel', timing_wheel_size=128))
    sim.init()
    target = [[x - 1 for x in ref.RING0_TARGET_SLOTS_ONE_BASED],
              [x - 1 for x in ref.RING1_TARGET_SLOTS_ONE_BASED]]
    phase_counts = [[0] * 32 for _ in range(2)]
    epoch_reward = [0] * args.epochs
    epoch_punish = [0] * args.epochs
    phase_correct = [0] * args.epochs
    step = 0
    phase_rows = []
    while step < cfg.steps - cfg.reward_delay_steps:
        phase_number = step // cfg.phase_steps
        phase = phase_number % cfg.phases
        end = min(step + cfg.communication_interval, cfg.steps - cfg.reward_delay_steps)
        input_times = list(range(step + 10, end, 100))
        if input_times:
            sim.add_external_spikes(input_times, [phase] * len(input_times))
        sim.run(end - step)
        out = sim.output_spikes()
        counts = [[0] * 32 for _ in range(2)]
        for spike in out:
            nid = int(spike['neuron_id'])
            if 100 <= nid < 164:
                ring = (nid - 100) // 32
                counts[ring][(nid - 100) % 32] += 1
        for ring in range(2):
            for slot in range(32):
                phase_counts[ring][slot] += counts[ring][slot]
            peak = max(counts[ring])
            if peak > 0:
                winner = counts[ring].index(peak)
                correct = winner == target[ring][phase]
                epoch = min(phase_number // cfg.phases, args.epochs - 1)
                if correct:
                    fb = cfg.feedback_start + ring * cfg.input_groups + phase
                    epoch_reward[epoch] += 1
                else:
                    fb = (cfg.feedback_start + 2 * cfg.input_groups +
                          ring * cfg.input_groups + phase)
                    epoch_punish[epoch] += 1
                sim.add_external_spikes([end + 1], [fb])
                phase_rows.append({'phase': phase, 'ring': ring,
                                   'time_step': end, 'winner': winner,
                                   'target': target[ring][phase],
                                   'spike_count': peak,
                                   'correct': int(correct)})
        next_phase_number = end // cfg.phase_steps
        if next_phase_number != phase_number or end >= cfg.steps - cfg.reward_delay_steps:
            epoch = min(phase_number // cfg.phases, args.epochs - 1)
            winners = []
            for ring in range(2):
                peak = max(phase_counts[ring])
                winners.append(phase_counts[ring].index(peak) if peak > 0 else -1)
            if winners[0] == target[0][phase] and winners[1] == target[1][phase]:
                phase_correct[epoch] += 1
            phase_counts = [[0] * 32 for _ in range(2)]
        step = end
    args.out.mkdir(parents=True, exist_ok=True)
    sim.save_weights(args.out / 'final_weights.dat')
    result = {'input_weight_mean': args.input_weight_mean,
              'phase_steps': args.phase_steps, 'epochs': args.epochs,
              'rewards_by_epoch': epoch_reward,
              'punishments_by_epoch': epoch_punish,
              'phase_control_correct_by_epoch': phase_correct,
              'phase_control_accuracy': [x / ref.PHASES for x in phase_correct],
              'window_rows': len(phase_rows)}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    import csv
    with (args.out / 'window_winners.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['phase','ring','time_step','winner','target','spike_count','correct'])
        w.writeheader(); w.writerows(phase_rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
