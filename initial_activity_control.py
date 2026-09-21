"""One-phase no-learning activity control for the dual-ring attractor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import neuronbridge as nb


def build_network(input_groups: int = 100, ring_size: int = 32,
                  seed: int = 20260425) -> nb.Network:
    n = nb.Network()
    n.add_layer(nb.NeuronLayer("InputSpikeNeuronModel", input_groups,
                               monitored=True, communication_input=True))
    n.add_layer(nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double", 2 * ring_size,
        monitored=True, output=True,
        parameters={
            "V_rest": nb.float32(-60.0), "V_reset": nb.float32(-60.0),
            "V_th": nb.float32(-40.0), "tau": nb.float32(15.0),
            "R": nb.float32(1.0), "t_ref": nb.int32(60),
            "gexc_tau": nb.float32(8.0), "ginh_tau": nb.float32(15.0),
            "Eexc": nb.float32(0.0), "Einh": nb.float32(-80.0),
        }))
    rng = random.Random(seed)
    for source in range(input_groups):
        for ring in range(2):
            for target in range(ring * ring_size, (ring + 1) * ring_size):
                n.connect(nb.Connection(
                    source, input_groups + target, synapse_type=0,
                    weight=1.0 + rng.uniform(-0.35, 0.35),
                    max_weight=8.0, delay=1))
    for ring in range(2):
        start = input_groups + ring * ring_size
        for source_slot in range(ring_size):
            for target_slot in range(ring_size):
                if source_slot == target_slot:
                    continue
                distance = min(abs(source_slot - target_slot),
                               ring_size - abs(source_slot - target_slot))
                normalized = distance / max(1, ring_size // 2)
                weight = 1.2 + (5.0 - 1.2) * normalized * normalized
                n.connect(nb.Connection(start + source_slot,
                                        start + target_slot,
                                        synapse_type=1, weight=weight,
                                        max_weight=weight, delay=1))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n = build_network()
    sim = nb.Simulation(
        n, nb.SimulationConfig(steps=args.steps, timestep=0.1,
                               queues=2, event_queue="timing_wheel",
                               timing_wheel_size=128))
    # Phase 0 owns input neuron 0. Use the same 100-step stimulation cadence
    # as the training server, but never send reward/punishment spikes.
    times = list(range(10, args.steps, 100))
    sim.init()
    sim.add_external_spikes(times, [0] * len(times))
    sim.run(args.steps)
    spikes = sim.output_spikes()
    ring0 = [s for s in spikes if 100 <= int(s["neuron_id"]) < 132]
    ring1 = [s for s in spikes if 132 <= int(s["neuron_id"]) < 164]
    result = {
        "steps": args.steps,
        "input_spikes": len(times),
        "total_output_spikes": len(spikes),
        "ring0_spikes": len(ring0),
        "ring1_spikes": len(ring1),
        "ring0_active": bool(ring0),
        "ring1_active": bool(ring1),
        "ring0_neurons": sorted({int(s["neuron_id"]) - 100 for s in ring0}),
        "ring1_neurons": sorted({int(s["neuron_id"]) - 132 for s in ring1}),
        "ring0_first_last_ms": ([ring0[0]["time"], ring0[-1]["time"]]
                                 if ring0 else None),
        "ring1_first_last_ms": ([ring1[0]["time"], ring1[-1]["time"]]
                                 if ring1 else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
