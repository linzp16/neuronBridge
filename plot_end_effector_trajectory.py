from pathlib import Path
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CoppeliaSim end-effector trajectory")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    csv_path = args.csv_path
    out_dir = args.output_dir or csv_path.parent
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"轨迹文件为空: {csv_path}")

    x = np.asarray([float(r["end_x"]) for r in rows])
    y = np.asarray([float(r["end_y"]) for r in rows])
    z = np.asarray([float(r["end_z"]) for r in rows])
    phase = np.asarray([int(r["phase"]) for r in rows])
    sample = np.arange(len(rows))

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=int(phase.min()), vmax=max(int(phase.max()), 1))

    fig, (ax_xy, ax_t) = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    # Draw each measured segment using its phase color.
    for i in range(len(x) - 1):
        ax_xy.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(phase[i])), linewidth=1.8)
    points = ax_xy.scatter(x, y, c=phase, cmap=cmap, norm=norm, s=20,
                           edgecolors="white", linewidths=0.35, zorder=3)
    ax_xy.scatter(x[0], y[0], s=90, c="#2ca02c", marker="o", zorder=5,
                  label="start")
    ax_xy.scatter(x[-1], y[-1], s=105, c="#d62728", marker="X", zorder=5,
                  label="end")

    # Label the first measured point of each phase without cluttering the plot.
    for p in np.unique(phase):
        idx = int(np.flatnonzero(phase == p)[0])
        ax_xy.annotate(str(p), (x[idx], y[idx]), xytext=(4, 4),
                       textcoords="offset points", fontsize=7, color="#333333")

    ax_xy.set_title("Measured end-effector trajectory")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.28)
    ax_xy.legend(loc="best", frameon=True)
    cbar = fig.colorbar(points, ax=ax_xy, pad=0.02)
    cbar.set_label("control phase")

    ax_t.plot(sample, x, color="#1f77b4", linewidth=1.6, label="x")
    ax_t.plot(sample, y, color="#ff7f0e", linewidth=1.6, label="y")
    ax_t.plot(sample, z, color="#2ca02c", linewidth=1.2, alpha=0.8, label="z")
    ax_t.set_title("End-effector position by sample")
    ax_t.set_xlabel("trajectory sample")
    ax_t.set_ylabel("position (m)")
    ax_t.grid(True, alpha=0.28)
    ax_t.legend(loc="best")

    fig.suptitle("NeuronBridge + CoppeliaSim closed-loop run", fontsize=14)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "end_effector_trajectory.png"
    svg_path = out_dir / "end_effector_trajectory.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")
    print(f"samples={len(rows)}, x_range={x.max() - x.min():.9f}, y_range={y.max() - y.min():.9f}")


if __name__ == "__main__":
    main()
