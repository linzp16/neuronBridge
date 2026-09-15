"""Audit and plot the generated R-STDP CPU/GPU weight trajectory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader((args.trace_dir / "cpu_gpu_weight_trace.csv").open(newline="")))
    rules = sorted({row["rule"] for row in rows})
    max_error = max((float(row["abs_error"]) for row in rows), default=float("inf"))
    report = {
        "result": "PASS" if len(rows) == 22 and len(rules) == 2 and max_error <= 5e-6 else "FAIL",
        "rules": rules,
        "samples_per_backend": len(rows),
        "max_abs_weight_error": max_error,
        "tolerance": 5e-6,
    }
    (args.trace_dir / "cpu_gpu_audit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(rules), 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        for ax, rule in zip(axes, rules):
            selected = [row for row in rows if row["rule"] == rule]
            times = [int(row["time_step"]) for row in selected]
            ax.plot(times, [float(row["cpu_weight"]) for row in selected],
                    marker="o", label="Generated CPU")
            ax.plot(times, [float(row["gpu_weight"]) for row in selected],
                    linestyle="--", marker="x", label="Generated CUDA")
            ax.set_ylabel("Weight")
            ax.set_title(rule)
            ax.grid(alpha=0.25)
            ax.legend()
        axes[-1].set_xlabel("Time step")
        fig.suptitle(f"Generated R-STDP CPU/CUDA comparison | max error {max_error:.3g}")
        fig.savefig(args.trace_dir / "cpu_gpu_weight_trajectory.png", dpi=180)
        plt.close(fig)

    print(json.dumps(report, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
