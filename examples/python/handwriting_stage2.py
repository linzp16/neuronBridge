"""Python migration of the C++ handwriting_stage2 reproduction example."""

from __future__ import annotations

import argparse
from pathlib import Path

from _handwriting_stage1 import ArmParameters
from _handwriting_stage1 import StrokeComputation
from _handwriting_stage1 import build_desired_joint_trajectory
from _handwriting_stage1 import load_stroke_path_from_directory
from _handwriting_stage1 import mean_squared_error
from _handwriting_stage1 import write_stroke_outputs
from _handwriting_stage2 import Stage2Config
from _handwriting_stage2 import decode_population_spikes
from _handwriting_stage2 import find_existing_stage2_input
from _handwriting_stage2 import load_binary_spike_series
from _handwriting_stage2 import load_binary_weight_windows
from _handwriting_stage2 import replay_stage2_path
from _handwriting_stage2 import simulate_cm_forward_exact
from _handwriting_stage2 import write_summary
from _handwriting_stage2 import write_torque_file
from _handwriting_stage2 import write_window_counts
from _example_paths import artifact_dir, data_dir


def default_input_dir() -> Path:
    return data_dir("handwriting_stage2")


def run_stage2(
    input_dir: Path,
    output_dir: Path,
    *,
    max_windows: int | None = None,
    stems: tuple[str, ...] = ("1", "2", "3"),
):
    cfg = Stage2Config()
    arm = ArmParameters()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext_spikes = load_binary_spike_series(
        find_existing_stage2_input(input_dir, "sspk_ext", ".bin"),
        find_existing_stage2_input(input_dir, "sspk_ext", ".shape.txt"),
    )
    summaries: list[tuple[str, int, float, float]] = []

    for stem in stems:
        desired_path = load_stroke_path_from_directory(input_dir, stem)
        weights = load_binary_weight_windows(
            find_existing_stage2_input(input_dir, f"ww_extCM{stem}", ".bin"),
            find_existing_stage2_input(input_dir, f"ww_extCM{stem}", ".shape.txt"),
            max_windows=max_windows,
        )
        cm_forward = simulate_cm_forward_exact(cfg, ext_spikes, weights, max_windows=max_windows)
        decoded_torque = decode_population_spikes(cm_forward.population_counts, cfg.n_pop)
        replay = replay_stage2_path(desired_path, decoded_torque, arm)
        desired_joints = build_desired_joint_trajectory(desired_path, arm, 0.01)

        if max_windows is not None:
            count = min(len(decoded_torque.q1), len(desired_path.x), len(replay.hand_path.x))
            desired_path = type(desired_path)(x=desired_path.x[:count], y=desired_path.y[:count])
            desired_joints = type(desired_joints)(
                theta1=desired_joints.theta1[:count],
                theta2=desired_joints.theta2[:count],
                dtheta1=desired_joints.dtheta1[:count],
                dtheta2=desired_joints.dtheta2[:count],
                ddtheta1=desired_joints.ddtheta1[:count],
                ddtheta2=desired_joints.ddtheta2[:count],
            )
            replay = type(replay)(
                theta1=replay.theta1[:count],
                theta2=replay.theta2[:count],
                dtheta1=replay.dtheta1[:count],
                dtheta2=replay.dtheta2[:count],
                ddtheta1=replay.ddtheta1[: max(0, count - 1)],
                ddtheta2=replay.ddtheta2[: max(0, count - 1)],
                hand_path=type(replay.hand_path)(x=replay.hand_path.x[:count], y=replay.hand_path.y[:count]),
            )

        stroke_dir = output_dir / f"stroke_{stem}"
        stroke_dir.mkdir(parents=True, exist_ok=True)
        write_window_counts(stroke_dir / f"cm_window_counts_{stem}.tsv", cm_forward.population_counts, cfg.n_pop)
        write_torque_file(stroke_dir / f"QQ{stem}.tsv", decoded_torque)
        write_summary(stroke_dir / f"summary_{stem}.tsv", desired_path, replay)
        write_stroke_outputs(stroke_dir, stem, StrokeComputation(desired_path, desired_joints, decoded_torque, replay))
        summaries.append(
            (
                stem,
                cm_forward.simulated_windows,
                mean_squared_error(desired_path.x, replay.hand_path.x),
                mean_squared_error(desired_path.y, replay.hand_path.y),
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=default_input_dir())
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=artifact_dir("handwriting_stage2"),
    )
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for stem, windows, mse_x, mse_y in run_stage2(args.input_dir, args.output_dir, max_windows=args.max_windows):
        print(f"stroke {stem}: windows={windows}, mse_x={mse_x}, mse_y={mse_y}")
    print("Stage-2 handwriting reproduction finished.")


if __name__ == "__main__":
    main()
