"""Python migration for the C++ handwriting_stage3_framework example.

This entry point preserves the framework example's command-line shape and
output contract while routing through the shared stage-3 Python runtime. It is
deliberately implemented as ordinary Python example code rather than a
per-example native binding channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _handwriting_stage3 import Stage3Config
from _handwriting_stage3 import Stage3Result
from _handwriting_stage3 import load_stage3_data
from _handwriting_stage3 import parse_stage3_mode
from _handwriting_stage3 import run_stage3_numpy
from _handwriting_stage3 import target_path_for_mode
from _handwriting_stage3 import write_stage3_outputs
from _example_paths import artifact_dir, data_dir


def default_input_dir() -> Path:
    return data_dir("handwriting_stage3_framework")


def default_output_dir() -> Path:
    return artifact_dir("handwriting_stage3_framework")


def run_framework(
    input_dir: Path,
    output_dir: Path,
    mode: str = "test",
    *,
    max_steps: int | None = None,
    skip_optional_weights: bool = False,
) -> Stage3Result:
    parsed_mode = parse_stage3_mode(mode)
    data = load_stage3_data(
        input_dir,
        require_test_weights=parsed_mode.value.startswith("test"),
        load_optional_weights=not skip_optional_weights,
    )
    result = run_stage3_numpy(Stage3Config(), data, parsed_mode, max_steps=max_steps)
    write_stage3_outputs(output_dir, result, target_path_for_mode(data, parsed_mode))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=default_input_dir())
    parser.add_argument("output_dir", nargs="?", type=Path, default=default_output_dir())
    parser.add_argument("mode", nargs="?", default="test")
    parser.add_argument("--max-steps", type=int, default=None, help="limit runtime steps for smoke/profiling runs")
    parser.add_argument("--skip-optional-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_framework(
        args.input_dir,
        args.output_dir,
        args.mode,
        max_steps=args.max_steps,
        skip_optional_weights=args.skip_optional_weights,
    )
    print(f"simulated_steps={result.simulated_steps}")
    print(f"window_count={len(result.pop_spk) // 8}")
    print(f"total_population_spikes={sum(result.pop_spk)}")
    print(f"output_dir={args.output_dir}")
    print(f"Stage-3 handwriting framework reproduction finished in mode {parse_stage3_mode(args.mode).value}.")


if __name__ == "__main__":
    main()
