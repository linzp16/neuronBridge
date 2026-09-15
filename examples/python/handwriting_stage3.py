"""Python migration scaffold for the C++ handwriting_stage3 example."""

from __future__ import annotations

import argparse
from pathlib import Path

from _handwriting_stage3 import Stage3Config
from _handwriting_stage3 import load_stage3_data
from _handwriting_stage3 import parse_stage3_mode
from _handwriting_stage3 import run_stage3_numpy
from _handwriting_stage3 import target_path_for_mode
from _handwriting_stage3 import write_stage3_outputs
from _example_paths import artifact_dir, data_dir


def default_input_dir() -> Path:
    return data_dir("handwriting_stage3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=default_input_dir())
    parser.add_argument("mode", nargs="?", default="test")
    parser.add_argument("--skip-optional-weights", action="store_true")
    parser.add_argument("--run", action="store_true", help="run the NumPy stage-3 runtime instead of only inspecting inputs")
    parser.add_argument("--max-steps", type=int, default=None, help="limit runtime steps for smoke/profiling runs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifact_dir("handwriting_stage3"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = parse_stage3_mode(args.mode)
    data = load_stage3_data(
        args.input_dir,
        require_test_weights=mode.value.startswith("test"),
        load_optional_weights=not args.skip_optional_weights,
    )
    target = target_path_for_mode(data, mode)
    print(f"mode={mode.value}")
    print(f"sspk_ext_shape={data.sspk_ext.shape.rows}x{data.sspk_ext.shape.cols}")
    print(f"sspk_extE_shape={data.sspk_ext_e.shape.rows}x{data.sspk_ext_e.shape.cols}")
    print(f"sspk_extI_shape={data.sspk_ext_i.shape.rows}x{data.sspk_ext_i.shape.cols}")
    print(f"ww_extCM1_shape={data.ww_extcm1.shape.dim0}x{data.ww_extcm1.shape.dim1}x{data.ww_extcm1.shape.dim2}")
    print(f"C_EMM1_shape={data.c_emm1.rows}x{data.c_emm1.cols}")
    print(f"w_MMbg1_shape={data.w_mmbg1.rows}x{data.w_mmbg1.cols}")
    print(f"target_samples={len(target.x)}")
    if args.run:
        result = run_stage3_numpy(Stage3Config(), data, mode, max_steps=args.max_steps)
        write_stage3_outputs(args.output_dir, result, target)
        print(f"simulated_steps={result.simulated_steps}")
        print(f"window_count={len(result.pop_spk) // 8}")
        print(f"total_population_spikes={sum(result.pop_spk)}")
        print(f"output_dir={args.output_dir}")
        print("Stage-3 handwriting NumPy runtime finished.")
    else:
        print("Stage-3 handwriting migration scaffold loaded inputs successfully.")


if __name__ == "__main__":
    main()
