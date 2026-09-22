"""Plot the 100-epoch OuterDynamic and synchronous REQ/REP comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def epoch_rows(rows: list[dict], epoch: int) -> list[dict]:
    return [row for row in rows if int(row["epoch"]) == epoch]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outerdynamic-dir", type=Path, default=Path("artifacts/full_cerebellum_training_100")
    )
    parser.add_argument(
        "--reqrep-dir", type=Path, default=Path("artifacts/full_cerebellum_reqrep_training_100")
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/full_cerebellum_reqrep_training_100/outerdynamic_vs_reqrep_comparison.png"),
    )
    args = parser.parse_args()

    outer_report = load_json(args.outerdynamic_dir / "training_report.json")
    reqrep_report = load_json(args.reqrep_dir / "training_report.json")
    outer_rows = load_rows(args.outerdynamic_dir / "trajectory.jsonl")
    reqrep_rows = load_rows(args.reqrep_dir / "trajectory.jsonl")

    outer_epochs = outer_report["epochs"]
    reqrep_epochs = reqrep_report["epochs"]
    epoch_numbers = [int(item["epoch"]) for item in outer_epochs]
    outer_error = [float(item["mean_error_m"]) for item in outer_epochs]
    reqrep_error = [float(item["mean_error_m"]) for item in reqrep_epochs]
    outer_time = [float(item["elapsed_seconds"]) for item in outer_epochs]
    reqrep_time = [float(item["elapsed_seconds"]) for item in reqrep_epochs]
    outer_best = int(outer_report["convergence"]["best_epoch"])
    reqrep_best = int(reqrep_report["convergence"]["best_epoch"])

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"outer": "#31688e", "reqrep": "#d1495b", "desired": "#202020"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Full cerebellum: internal OuterDynamic vs external synchronous REQ/REP", fontsize=16)

    ax = axes[0, 0]
    ax.plot(epoch_numbers, outer_error, color=colors["outer"], lw=1.8, label="OuterDynamic")
    ax.plot(epoch_numbers, reqrep_error, color=colors["reqrep"], lw=1.8, label="REQ/REP server")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean endpoint error (m, log scale)")
    ax.set_title("Training convergence")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epoch_numbers, outer_time, color=colors["outer"], lw=1.4, label="OuterDynamic")
    ax.plot(epoch_numbers, reqrep_time, color=colors["reqrep"], lw=1.4, label="REQ/REP server")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Wall time per epoch (s)")
    ax.set_title("Runtime cost after one-time construction")
    ax.legend()

    for ax, epoch, title in (
        (axes[1, 0], 100, "Final epoch trajectories"),
        (axes[1, 1], None, "Best epoch trajectories"),
    ):
        outer_selected = epoch_rows(outer_rows, outer_best if epoch is None else epoch)
        reqrep_selected = epoch_rows(reqrep_rows, reqrep_best if epoch is None else epoch)
        desired = reqrep_selected or outer_selected
        ax.plot(
            [row["desired_x"] for row in desired], [row["desired_y"] for row in desired],
            color=colors["desired"], lw=2.2, ls="--", label="Desired",
        )
        ax.plot(
            [row["x"] for row in outer_selected], [row["y"] for row in outer_selected],
            color=colors["outer"], lw=1.7,
            label=f"OuterDynamic (epoch {outer_best if epoch is None else epoch})",
        )
        ax.plot(
            [row["x"] for row in reqrep_selected], [row["y"] for row in reqrep_selected],
            color=colors["reqrep"], lw=1.7,
            label=f"REQ/REP (epoch {reqrep_best if epoch is None else epoch})",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
