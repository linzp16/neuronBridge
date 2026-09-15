"""Independently audit all exported samples and plot two complete event trajectories."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = args.trace_root
    scenarios, selected, event_count = {}, {}, 0
    pairs = ("weight", "pre", "post", "eligibility", "last_update")
    max_error = {name: 0.0 for name in pairs}
    with (root / "event_weight_trace.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scenario, event, connection = (int(row[key]) for key in ("scenario", "event_index", "connection"))
            entry = scenarios.setdefault(scenario, {"rows": 0, "events": {}, "max_weight_error": 0.0})
            seen = entry["events"].setdefault(event, set())
            if connection in seen:
                raise ValueError("duplicate connection snapshot")
            seen.add(connection)
            for key in pairs:
                a, b = float(row["reference_" + key]), float(row["generated_" + key])
                if not math.isfinite(a) or not math.isfinite(b) or a != b:
                    raise ValueError(f"CSV mismatch: scenario={scenario}, event={event}, connection={connection}, {key}")
                max_error[key] = max(max_error[key], abs(a - b))
            if not 0 <= float(row["generated_weight"]) <= 1:
                raise ValueError("weight out of bounds")
            if scenario in (0, 12) and connection == 0:
                selected.setdefault(scenario, []).append(row)
            entry["rows"] += 1
            event_count += 1
    if set(scenarios) != set(range(96)):
        raise ValueError("incomplete scenario coverage")
    for entry in scenarios.values():
        if set(entry["events"]) != set(range(275)) or any(ids != set(range(6)) for ids in entry["events"].values()):
            raise ValueError("incomplete event/connection trajectory")
        entry["events"] = len(entry["events"]) - 1
    simulation_count, network_cases = 0, {}
    with (root / "simulation_weight_trace.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            a, b = float(row["reference"]), float(row["generated"])
            if not math.isfinite(a) or not math.isfinite(b) or a != b:
                raise ValueError("Simulation CSV mismatch")
            key = (row["generated_rule"], int(row["clear"]), int(row["delay"]))
            samples = network_cases.setdefault(key, set())
            sample = (int(row["step"]), row["field"])
            if sample in samples:
                raise ValueError("duplicate Simulation sample")
            samples.add(sample)
            simulation_count += 1
    if set(network_cases) != {(rule, clear, delay)
                              for rule in ("CustomRStdpV1", "CustomRStdpPersistentV1")
                              for clear in (0, 1) for delay in (1, 3, 7)}:
        raise ValueError("incomplete Simulation cases")
    for samples in network_cases.values():
        fields = {field for step, field in samples}
        if len(fields) != 15 or samples != {(step, field) for step in range(1, 501) for field in fields}:
            raise ValueError("incomplete Simulation trajectory")
    summary = {"result": "PASS", "scenarios": scenarios, "event_rows": event_count,
               "events": sum(entry["events"] for entry in scenarios.values()),
               "max_abs_error": max_error, "simulation_cases": len(network_cases),
               "simulation_samples": simulation_count}
    (root / "trajectory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "scenarios"}, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for col, scenario in enumerate((0, 12)):
            rows = selected[scenario]
            for axis, key in zip(axes[:, col], ("weight", "eligibility")):
                x = [int(row["event_index"]) for row in rows]
                axis.plot(x, [float(row["reference_" + key]) for row in rows], color="#0072B2", lw=1.7, label="Handwritten R_STDP")
                axis.plot(x, [float(row["generated_" + key]) for row in rows], color="#D55E00", lw=1.0, ls="--", label="Generated CustomRStdpV1")
                axis.set(xlabel="Event index (same-tick order preserved)", ylabel=key.capitalize())
                axis.grid(alpha=0.2)
                axis.spines[["top", "right"]].set_visible(False)
            axes[0, col].set_title(f"Eligibility clear: {'off' if scenario == 0 else 'on'} | connection 0")
            axes[0, col].legend(fontsize=9)
        fig.suptitle(f"R-STDP full event trajectory | {len(scenarios)} scenarios, all exported errors = 0", fontsize=14)
        fig.savefig(root / "weight_trajectory_comparison.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
