"""Validate the NumPy/Matplotlib runtime before scientific tests run.

This check is intentionally separate from importing NeuronBridge so that an
ABI mismatch produces an actionable message instead of a long Matplotlib
traceback during an unrelated test.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys


def main() -> int:
    try:
        import numpy as np
    except Exception as exc:
        print(f"ERROR: NumPy cannot be imported: {exc}", file=sys.stderr)
        return 2

    if int(np.__version__.split(".", 1)[0]) < 2:
        print(
            f"ERROR: NumPy {np.__version__} is unsupported. "
            "NeuronBridge requires NumPy >=2.0,<3.0.",
            file=sys.stderr,
        )
        return 2

    try:
        matplotlib = importlib.import_module("matplotlib")
        importlib.import_module("matplotlib.pyplot")
        contourpy = importlib.import_module("contourpy")
    except Exception as exc:
        print(
            "ERROR: scientific plotting runtime is ABI-incompatible.\n"
            f"  NumPy: {np.__version__}\n"
            "  Required: Matplotlib >=3.9 and contourpy >=1.3 built for NumPy 2.x\n"
            f"  Import error: {exc}\n"
            "Fix with:\n"
            "  python -m pip install --upgrade \"numpy>=2,<3\" \"matplotlib>=3.9\" \"contourpy>=1.3\"",
            file=sys.stderr,
        )
        return 2

    print(
        "scientific runtime PASS: "
        f"numpy={np.__version__}, "
        f"matplotlib={matplotlib.__version__}, "
        f"contourpy={contourpy.__version__}"
    )

    # The native wheel currently has a Windows DLL load-order constraint: the
    # native extension must be imported before Matplotlib. Probe this in a
    # child process so a native access violation cannot terminate the test
    # runner itself.
    try:
        importlib.util.find_spec("neuronbridge")
    except Exception:
        return 0
    order_probe = subprocess.run(
        [sys.executable, "-c", "import matplotlib; import neuronbridge"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if order_probe.returncode != 0:
        print(
            "WARNING: importing Matplotlib before NeuronBridge can trigger a "
            "Windows native DLL access violation in this wheel. Import "
            "`neuronbridge` before `matplotlib`, or run tests in the isolated "
            "NeuronBridge environment. This probe ran in a child process and "
            "did not terminate the test runner.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
