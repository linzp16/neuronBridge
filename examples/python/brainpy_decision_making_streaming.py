"""BrainPy-style two-choice decision network built with NeuronBridge.

This reproduces the population sizes, connection strengths, stimulus protocol,
and 0.1 ms integration step from BrainPy's "Building a decision-making
network" quickstart.  The 7.2 million synapses are emitted in bounded batches
to an .nbnet file so Python never materializes the complete graph in memory.

The biological-neuron backend is selectable.  The CPU and GPU triple LIF
implementations both use independent AMPA/GABA/NMDA states, voltage-dependent
magnesium block, and a single 100 ms NMDA decay.  This aligns the two
NeuronBridge backends, although BrainPy's reference uses a dual-exponential
2 ms rise/100 ms decay NMDA channel.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import neuronbridge as nb


DT_MS = 0.1
PRE_MS = 100.0
STIMULUS_MS = 1000.0
DELAY_MS = 500.0
TOTAL_MS = PRE_MS + STIMULUS_MS + DELAY_MS

N_E = 1600
N_A = 240
N_B = 240
N_N = 1120
N_I = 400
N_STIMULUS = N_A + N_B
N_NOISE_E = N_E
N_NOISE_I = N_I
N_CURRENT = 2

F = 0.15
W_POS = 1.7
W_NEG = 1.0 - F * (W_POS - 1.0) / (1.0 - F)

G_EXT_E_AMPA = 2.1
G_EXT_I_AMPA = 1.62
G_E2E_AMPA = 0.05
G_E2I_AMPA = 0.04
G_E2E_NMDA = 0.165
G_E2I_NMDA = 0.13
G_I2E_GABAA = 1.3
G_I2I_GABAA = 1.0

NOISE_RATE_HZ = 2400.0
MU0_HZ = 40.0
COHERENCE_PERCENT = 25.6
STIMULUS_INTERVAL_MS = 50.0
STIMULUS_SIGMA_HZ = 10.0


def lif_layer(
    count: int,
    *,
    tau_ms: float,
    resistance: float,
    refractory_ms: float,
    backend: str,
) -> nb.NeuronLayer:
    model = (
        "TimeDrivenLIF_Exponential_triple"
        if backend == "cpu"
        else "TimeDrivenLIF_Exponential_triple_GPU"
    )
    return nb.NeuronLayer(
        model,
        count,
        output=True,
        parameters={
            "V_rest": nb.float32(-70.0),
            "V_th": nb.float32(-50.0),
            "V_reset": nb.float32(-55.0),
            "tau": nb.float32(tau_ms),
            "R": nb.float32(resistance),
            "t_ref": nb.int32(round(refractory_ms / DT_MS)),
            "E_ampa": nb.float32(0.0),
            "ampa_tau": nb.float32(2.0),
            "E_gaba": nb.float32(-70.0),
            "gaba_tau": nb.float32(5.0),
            "nmda_tau": nb.float32(100.0),
        },
    )


def append_all_to_all(
    builder: nb.NbnetDescriptionBuilder,
    *,
    source_begin: int,
    source_count: int,
    target_begin: int,
    target_count: int,
    synapse_type: int,
    weight: float,
    delay: int,
    batch_size: int,
) -> int:
    """Append a complete bipartite projection in bounded batches."""

    pair_count = source_count * target_count
    written = 0
    for batch_begin in range(0, pair_count, batch_size):
        batch_end = min(pair_count, batch_begin + batch_size)
        flat = range(batch_begin, batch_end)
        sources = [source_begin + index // target_count for index in flat]
        targets = [target_begin + index % target_count for index in range(batch_begin, batch_end)]
        written += builder.append_connections(
            sources,
            targets,
            synapse_type=synapse_type,
            weight=weight,
            max_weight=weight,
            delay=delay,
        )
    return written


def append_one_to_one(
    builder: nb.NbnetDescriptionBuilder,
    *,
    source_begin: int,
    target_begin: int,
    count: int,
    synapse_type: int,
    weight: float,
    delay: int,
) -> int:
    return builder.append_connections(
        list(range(source_begin, source_begin + count)),
        list(range(target_begin, target_begin + count)),
        synapse_type=synapse_type,
        weight=weight,
        max_weight=weight,
        delay=delay,
    )


def build_network(
    path: Path, batch_size: int, backend: str
) -> tuple[nb.NbnetBuildResult, dict[str, int], dict[str, int]]:
    projection_counts: dict[str, int] = {}
    recurrent_delay = round(0.5 / DT_MS)

    with nb.NbnetDescriptionBuilder(path, options=nb.NbnetWriteOptions(overwrite=True)) as builder:
        e = builder.add_layer(
            lif_layer(
                N_E,
                tau_ms=20.0,
                resistance=0.04,
                refractory_ms=2.0,
                backend=backend,
            )
        )
        inhibitory = builder.add_layer(
            lif_layer(
                N_I,
                tau_ms=10.0,
                resistance=0.05,
                refractory_ms=1.0,
                backend=backend,
            )
        )
        stimulus = builder.add_layer(
            nb.NeuronLayer.poisson_rate(
                N_STIMULUS, rate_bias_hz=0.0, rate_gain_hz_per_current=1.0
            )
        )
        noise_e = builder.add_layer(
            nb.NeuronLayer.poisson_rate(N_NOISE_E, rate_bias_hz=NOISE_RATE_HZ)
        )
        noise_i = builder.add_layer(
            nb.NeuronLayer.poisson_rate(N_NOISE_I, rate_bias_hz=NOISE_RATE_HZ)
        )
        current = builder.add_layer(nb.NeuronLayer.input_current(N_CURRENT))

        ranges = {
            "A": e.first_neuron_id,
            "B": e.first_neuron_id + N_A,
            "N": e.first_neuron_id + N_A + N_B,
            "I": inhibitory.first_neuron_id,
            "IA": stimulus.first_neuron_id,
            "IB": stimulus.first_neuron_id + N_A,
            "noise_E": noise_e.first_neuron_id,
            "noise_I": noise_i.first_neuron_id,
            "current_A": current.first_neuron_id,
            "current_B": current.first_neuron_id + 1,
        }
        population_sizes = {"A": N_A, "B": N_B, "N": N_N, "I": N_I}

        excitatory_sources = (("A", N_A), ("B", N_B), ("N", N_N))
        excitatory_targets = (("A", N_A), ("B", N_B), ("N", N_N), ("I", N_I))
        for source_name, source_count in excitatory_sources:
            for target_name, target_count in excitatory_targets:
                if target_name == "I":
                    ampa_weight = G_E2I_AMPA
                    nmda_weight = G_E2I_NMDA
                else:
                    multiplier = 1.0
                    if source_name == target_name and source_name in {"A", "B"}:
                        multiplier = W_POS
                    elif target_name in {"A", "B"} and source_name != target_name:
                        multiplier = W_NEG
                    ampa_weight = G_E2E_AMPA * multiplier
                    nmda_weight = G_E2E_NMDA * multiplier

                for channel_name, synapse_type, weight in (
                    ("AMPA", 0, ampa_weight),
                    ("NMDA", 2, nmda_weight),
                ):
                    name = f"{source_name}2{target_name}_{channel_name}"
                    projection_counts[name] = append_all_to_all(
                        builder,
                        source_begin=ranges[source_name],
                        source_count=source_count,
                        target_begin=ranges[target_name],
                        target_count=target_count,
                        synapse_type=synapse_type,
                        weight=weight,
                        delay=recurrent_delay,
                        batch_size=batch_size,
                    )
                    print(f"{name}: {projection_counts[name]:,}", flush=True)

        for target_name, target_count in excitatory_targets:
            weight = G_I2I_GABAA if target_name == "I" else G_I2E_GABAA
            name = f"I2{target_name}_GABAA"
            projection_counts[name] = append_all_to_all(
                builder,
                source_begin=ranges["I"],
                source_count=N_I,
                target_begin=ranges[target_name],
                target_count=target_count,
                synapse_type=1,
                weight=weight,
                delay=recurrent_delay,
                batch_size=batch_size,
            )
            print(f"{name}: {projection_counts[name]:,}", flush=True)

        for name, source_begin, target_begin, count, weight in (
            ("IA2A_AMPA", ranges["IA"], ranges["A"], N_A, G_EXT_E_AMPA),
            ("IB2B_AMPA", ranges["IB"], ranges["B"], N_B, G_EXT_E_AMPA),
            ("noiseE2E_AMPA", ranges["noise_E"], ranges["A"], N_E, G_EXT_E_AMPA),
            ("noiseI2I_AMPA", ranges["noise_I"], ranges["I"], N_I, G_EXT_I_AMPA),
        ):
            projection_counts[name] = append_one_to_one(
                builder,
                source_begin=source_begin,
                target_begin=target_begin,
                count=count,
                synapse_type=0,
                weight=weight,
                delay=1,
            )

        for label, current_id, target_begin in (
            ("currentA2IA", ranges["current_A"], ranges["IA"]),
            ("currentB2IB", ranges["current_B"], ranges["IB"]),
        ):
            projection_counts[label] = builder.append_connections(
                [current_id] * N_A,
                list(range(target_begin, target_begin + N_A)),
                synapse_type=3,
                weight=1.0,
                max_weight=1.0,
                delay=1,
            )

        result = builder.finalize()
    return result, projection_counts, ranges


def stimulus_schedule(seed: int, ranges: dict[str, int]) -> tuple[list[int], list[int], list[float], list[dict[str, float]]]:
    rng = random.Random(seed)
    mean_a = MU0_HZ * (1.0 + COHERENCE_PERCENT / 100.0)
    mean_b = MU0_HZ * (1.0 - COHERENCE_PERCENT / 100.0)
    interval_steps = round(STIMULUS_INTERVAL_MS / DT_MS)
    stimulus_start = round(PRE_MS / DT_MS)
    interval_count = round(STIMULUS_MS / STIMULUS_INTERVAL_MS)
    times: list[int] = []
    neuron_ids: list[int] = []
    currents: list[float] = []
    rows: list[dict[str, float]] = []
    for interval in range(interval_count):
        step = stimulus_start + interval * interval_steps
        rate_a = max(0.0, rng.gauss(mean_a, STIMULUS_SIGMA_HZ))
        rate_b = max(0.0, rng.gauss(mean_b, STIMULUS_SIGMA_HZ))
        times.extend((step, step))
        neuron_ids.extend((ranges["current_A"], ranges["current_B"]))
        currents.extend((rate_a, rate_b))
        rows.append({"time_ms": step * DT_MS, "rate_a_hz": rate_a, "rate_b_hz": rate_b})

    stop_step = round((PRE_MS + STIMULUS_MS) / DT_MS)
    times.extend((stop_step, stop_step))
    neuron_ids.extend((ranges["current_A"], ranges["current_B"]))
    currents.extend((0.0, 0.0))
    rows.append({"time_ms": stop_step * DT_MS, "rate_a_hz": 0.0, "rate_b_hz": 0.0})
    return times, neuron_ids, currents, rows


def population_for(neuron_id: int) -> tuple[str, int]:
    if neuron_id < N_A:
        return "A", neuron_id
    if neuron_id < N_A + N_B:
        return "B", neuron_id - N_A
    if neuron_id < N_E:
        return "N", neuron_id - N_A - N_B
    if neuron_id < N_E + N_I:
        return "I", neuron_id - N_E
    return "non_output", -1


def export_spikes(path: Path, spikes: list[dict[str, int]]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "N": 0, "I": 0}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_step", "time_ms", "neuron_id", "population", "population_index"))
        for spike in spikes:
            step = int(spike["time"])
            neuron_id = int(spike["neuron_id"])
            population, local_id = population_for(neuron_id)
            if population == "non_output":
                continue
            counts[population] += 1
            writer.writerow((step, f"{step * DT_MS:.6f}", neuron_id, population, local_id))
    return counts


def export_stimulus(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("time_ms", "rate_a_hz", "rate_b_hz"))
        writer.writeheader()
        writer.writerows(rows)


def late_stimulus_rates(spikes: list[dict[str, int]]) -> dict[str, float]:
    start_ms = PRE_MS + 0.75 * STIMULUS_MS
    end_ms = PRE_MS + STIMULUS_MS
    duration_s = (end_ms - start_ms) / 1000.0
    counts = {"A": 0, "B": 0}
    for spike in spikes:
        time_ms = int(spike["time"]) * DT_MS
        if start_ms <= time_ms < end_ms:
            population, _ = population_for(int(spike["neuron_id"]))
            if population in counts:
                counts[population] += 1
    return {"A_hz": counts["A"] / (N_A * duration_s), "B_hz": counts["B"] / (N_B * duration_s)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("brainpy_decision_making"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--memory-budget-mb", type=int, default=128)
    parser.add_argument("--neuron-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    network_path = args.output_dir / "brainpy_decision_making.nbnet"

    total_started = time.perf_counter()
    build_started = time.perf_counter()
    result, projections, ranges = build_network(
        network_path, args.batch_size, args.neuron_backend
    )
    build_seconds = time.perf_counter() - build_started
    validation = nb.nbnet.validate(network_path, verify_checksum=True)
    if not validation.valid:
        raise RuntimeError(validation.errors)

    report: dict[str, object] = {
        "reference": "https://brainpy.readthedocs.io/quickstart/simulation.html#building-a-decision-making-network",
        "network_path": str(result.path),
        "dt_ms": DT_MS,
        "duration_ms": TOTAL_MS,
        "steps": round(TOTAL_MS / DT_MS),
        "neuron_backend": args.neuron_backend,
        "biological_neuron_model": (
            "TimeDrivenLIF_Exponential_triple"
            if args.neuron_backend == "cpu"
            else "TimeDrivenLIF_Exponential_triple_GPU"
        ),
        "population_sizes": {"A": N_A, "B": N_B, "N": N_N, "I": N_I},
        "total_neurons_including_inputs": result.neuron_count,
        "connection_count": result.connection_count,
        "projection_connections": projections,
        "w_pos": W_POS,
        "w_neg": W_NEG,
        "coherence_percent": COHERENCE_PERCENT,
        "network_file_bytes": result.file_bytes,
        "network_sha256": result.sha256,
        "checksum_valid": True,
        "nbnet_generation_seconds": build_seconds,
        "nmda_backend_note": (
            "CPU triple aligned to GPU triple: independent NMDA state, single-exponential tau=100 ms, and Mg block"
            if args.neuron_backend == "cpu"
            else "single-exponential tau=100 ms with Mg block; BrainPy reference uses rise=2 ms/decay=100 ms"
        ),
    }

    if not args.build_only:
        config = nb.SimulationConfig(
            steps=round(TOTAL_MS / DT_MS),
            timestep=DT_MS,
            event_queue="timing_wheel",
            timing_wheel_size=32768,
        )
        construct_started = time.perf_counter()
        sim = nb.Simulation(
            network_path,
            config,
            build_options=nb.StreamingBuildOptions(
                memory_budget_mb=args.memory_budget_mb,
                mmap=True,
                verify_checksum=True,
            ),
        )
        report["simulation_construct_seconds"] = time.perf_counter() - construct_started
        init_started = time.perf_counter()
        sim.init()
        report["simulation_init_seconds"] = time.perf_counter() - init_started

        times, neuron_ids, currents, stimulus_rows = stimulus_schedule(args.seed, ranges)
        sim.add_external_currents(times, neuron_ids, currents)
        run_started = time.perf_counter()
        sim.run()
        report["simulation_run_seconds"] = time.perf_counter() - run_started

        spikes = sim.output_spikes()
        spikes_path = args.output_dir / "spikes.csv"
        stimulus_path = args.output_dir / "stimulus_rates.csv"
        export_started = time.perf_counter()
        counts = export_spikes(spikes_path, spikes)
        export_stimulus(stimulus_path, stimulus_rows)
        report["export_seconds"] = time.perf_counter() - export_started
        report["spike_counts"] = counts
        report["total_output_spikes"] = sum(counts.values())
        report["late_stimulus_rates"] = late_stimulus_rates(spikes)
        report["winner"] = (
            "A"
            if report["late_stimulus_rates"]["A_hz"] > report["late_stimulus_rates"]["B_hz"]
            else "B"
        )
        report["build_stats"] = sim.build_stats
        report["spikes_path"] = str(spikes_path.resolve())
        report["stimulus_path"] = str(stimulus_path.resolve())

    report["total_seconds"] = time.perf_counter() - total_started
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
