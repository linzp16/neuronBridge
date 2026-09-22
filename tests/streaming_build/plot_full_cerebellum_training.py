"""Plot convergence, sampled weights, and trajectories for full cerebellum training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.trajectory.read_text(encoding="utf-8").splitlines() if line]
    epochs = report["epochs"]
    epoch_ids = [item["epoch"] for item in epochs]
    mean_error = [item["mean_error_m"] for item in epochs]
    rmse = [item["rmse_m"] for item in epochs]
    weight_mean = [item["sampled_weight_stats"]["mean"] for item in epochs]
    weight_min = [item["sampled_weight_stats"]["minimum"] for item in epochs]
    weight_max = [item["sampled_weight_stats"]["maximum"] for item in epochs]
    best_epoch = int(report["convergence"]["best_epoch"])

    fig, axes = plt.subplot_mosaic(
        [["error", "error"], ["weight", "trajectory"]],
        figsize=(13.2, 8.2),
        constrained_layout=True,
    )
    ax = axes["error"]
    ax.plot(epoch_ids, mean_error, color="#0072B2", linewidth=1.8, label="Mean position error")
    ax.plot(epoch_ids, rmse, color="#D55E00", linewidth=1.2, alpha=0.8, label="RMSE")
    ax.scatter([1, best_epoch, epoch_ids[-1]],
               [mean_error[0], mean_error[best_epoch - 1], mean_error[-1]],
               color=["#CC79A7", "#009E73", "#E69F00"], s=42, zorder=3)
    ax.annotate(f"epoch 1: {mean_error[0]:.4f} m", (1, mean_error[0]), xytext=(8, -5), textcoords="offset points")
    ax.annotate(f"best {best_epoch}: {mean_error[best_epoch - 1]:.4f} m",
                (best_epoch, mean_error[best_epoch - 1]), xytext=(-120, -34), textcoords="offset points")
    ax.annotate(f"epoch 100: {mean_error[-1]:.4f} m",
                (epoch_ids[-1], mean_error[-1]), xytext=(-138, 22), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("End-effector error (m, log scale)")
    ax.set_title("Full cerebellum training convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes["weight"]
    ax.plot(epoch_ids, weight_mean, color="#0072B2", linewidth=1.8, label="Sample mean")
    ax.fill_between(epoch_ids, weight_min, weight_max, color="#56B4E9", alpha=0.25, label="Sample min–max")
    ax.text(
        0.03,
        0.08,
        f"mean change: {weight_mean[-1] - weight_mean[0]:+.6f}",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GC→PC weight")
    ax.set_title("Sampled plastic-weight evolution")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes["trajectory"]
    desired = [row for row in rows if row["epoch"] == 1]
    ax.plot([row["desired_x"] for row in desired], [row["desired_y"] for row in desired],
            color="black", linestyle="--", linewidth=2.0, label="Desired")
    for epoch, color, label in (
        (1, "#CC79A7", "Epoch 1"),
        (best_epoch, "#009E73", f"Best epoch {best_epoch}"),
        (epoch_ids[-1], "#E69F00", "Epoch 100"),
    ):
        selected = [row for row in rows if row["epoch"] == epoch]
        ax.plot([row["x"] for row in selected], [row["y"] for row in selected],
                color=color, linewidth=1.5, label=label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("End-effector trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "NeuronBridge streamed 47,400-neuron / 36,001,600-synapse cerebellum",
        fontsize=14,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
