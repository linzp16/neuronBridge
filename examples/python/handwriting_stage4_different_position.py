"""Python migration of the C++ handwriting_stage4_different_position example."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from _handwriting_stage1 import ArmParameters
from _handwriting_stage1 import StrokePath
from _handwriting_stage1 import find_existing_series_path
from _handwriting_stage1 import inverse_kinematics
from _handwriting_stage1 import load_stroke_path_from_directory
from _handwriting_stage1 import load_torque_sequence
from _handwriting_stage1 import mean_squared_error
from _handwriting_stage1 import replay_shifted_stroke
from _handwriting_stage1 import write_two_column_file
from _example_paths import artifact_dir, data_dir


SHIFTS: tuple[tuple[float, float], ...] = (
    (-0.1, 0.1),
    (0.0, 0.1),
    (0.1, 0.1),
    (-0.1, 0.0),
    (0.0, 0.0),
    (0.1, 0.0),
    (-0.1, -0.1),
    (0.0, -0.1),
    (0.1, -0.1),
)


@dataclass(frozen=True)
class ShiftResult:
    dx: float
    dy: float
    target_path: StrokePath
    replay_path: StrokePath
    mse_x: float
    mse_y: float


def default_input_dir() -> Path:
    return data_dir("handwriting_stage4_different_position")


def shifted_path(base_path: StrokePath, dx: float, dy: float) -> StrokePath:
    return StrokePath(x=[value + dx for value in base_path.x], y=[value + dy for value in base_path.y])


def run_stage4(input_dir: Path, output_dir: Path, *, stem: str = "1", dt_seconds: float = 0.01) -> list[ShiftResult]:
    params = ArmParameters()
    base_path = load_stroke_path_from_directory(input_dir, stem)
    torques = load_torque_sequence(find_existing_series_path(input_dir, f"QQ{stem}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ShiftResult] = []
    for dx, dy in SHIFTS:
        target = shifted_path(base_path, dx, dy)
        replay = replay_shifted_stroke(inverse_kinematics(target.x[0], target.y[0], params), torques, params, dt_seconds)
        count = min(len(target.x), len(replay.hand_path.x))
        target = StrokePath(x=target.x[:count], y=target.y[:count])
        replay_path = StrokePath(x=replay.hand_path.x[:count], y=replay.hand_path.y[:count])
        results.append(
            ShiftResult(
                dx=dx,
                dy=dy,
                target_path=target,
                replay_path=replay_path,
                mse_x=mean_squared_error(target.x, replay_path.x),
                mse_y=mean_squared_error(target.y, replay_path.y),
            )
        )

    for index, result in enumerate(results, start=1):
        name = f"shift_{index:02d}"
        write_two_column_file(output_dir / f"{name}_target_path.tsv", result.target_path.x, result.target_path.y, "x", "y")
        write_two_column_file(output_dir / f"{name}_replay_path.tsv", result.replay_path.x, result.replay_path.y, "x", "y")
    write_combined_paths(output_dir / "all_target_paths.tsv", results, target=True)
    write_combined_paths(output_dir / "all_replay_paths.tsv", results, target=False)
    write_summary(output_dir / "summary.tsv", results)
    return results


def write_combined_paths(path: Path, results: list[ShiftResult], *, target: bool) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("shift_index\tdx\tdy\ttime_index\tx\ty\n")
        for index, result in enumerate(results, start=1):
            series = result.target_path if target else result.replay_path
            for time_index, (x, y) in enumerate(zip(series.x, series.y)):
                handle.write(f"{index}\t{result.dx:.17g}\t{result.dy:.17g}\t{time_index}\t{x:.17g}\t{y:.17g}\n")


def write_summary(path: Path, results: list[ShiftResult]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("shift_index\tdx\tdy\tpath_mse_x\tpath_mse_y\n")
        for index, result in enumerate(results, start=1):
            handle.write(
                f"{index}\t{result.dx:.17g}\t{result.dy:.17g}\t{result.mse_x:.17g}\t{result.mse_y:.17g}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=default_input_dir())
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=artifact_dir("handwriting_stage4_different_position"),
    )
    parser.add_argument("stem", nargs="?", default="1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_stage4(args.input_dir, args.output_dir, stem=args.stem)
    print(f"Stage-4 different-position reproduction finished for stroke {args.stem}.")
    for index, result in enumerate(results, start=1):
        print(
            f"shift {index}: dx={result.dx}, dy={result.dy}, "
            f"mse_x={result.mse_x}, mse_y={result.mse_y}"
        )


if __name__ == "__main__":
    main()
