"""Python migration of the C++ handwriting_stage1 reproduction example."""

from __future__ import annotations

import argparse
from pathlib import Path

from _handwriting_stage1 import ArmParameters, compute_stroke, load_stroke_path_from_directory, mean_squared_error
from _handwriting_stage1 import write_stroke_outputs
from _example_paths import artifact_dir, data_dir


def default_input_dir() -> Path:
    return data_dir("handwriting_stage1")


def run_stage1(input_dir: Path, output_dir: Path, *, dt_seconds: float = 0.01, stems: tuple[str, ...] = ("1", "2", "3")):
    params = ArmParameters()
    summaries: list[tuple[str, int, float, float]] = []
    for stem in stems:
        path = load_stroke_path_from_directory(input_dir, stem)
        result = compute_stroke(path, params, dt_seconds)
        write_stroke_outputs(output_dir / f"stroke_{stem}", stem, result)
        mse_x = mean_squared_error(result.desired_path.x, result.replay.hand_path.x)
        mse_y = mean_squared_error(result.desired_path.y, result.replay.hand_path.y)
        summaries.append((stem, len(result.desired_path.x), mse_x, mse_y))
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=default_input_dir())
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=artifact_dir("handwriting_stage1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    for stem, samples, mse_x, mse_y in run_stage1(args.input_dir, args.output_dir):
        print(f"stroke {stem}: samples={samples}, mse_x={mse_x}, mse_y={mse_y}")
    print("Stage-1 handwriting reproduction finished.")


if __name__ == "__main__":
    main()
