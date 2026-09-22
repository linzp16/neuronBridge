"""Compare peak construction memory for Network and .nbnet input paths.

Run this script with PYTHONPATH pointing at an importable NeuronBridge build.
Each mode executes in a fresh process so Windows PeakWorkingSetSize values are
directly comparable.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

import neuronbridge as nb


def _lif_layer(count: int, *, dense: bool = False) -> nb.NeuronLayer:
    parameters = {
        "V_rest": nb.float32(-60.0),
        "V_reset": nb.float32(-60.0),
        "V_th": nb.float32(-50.0),
        "tau": nb.float32(20.0),
        "R": nb.float32(1.0),
        "t_ref": nb.int32(1),
        "gexc_tau": nb.float32(5.0),
        "ginh_tau": nb.float32(10.0),
        "Eexc": nb.float32(0.0),
        "Einh": nb.float32(-80.0),
    }
    if dense:
        parameters["dense_subnetwork_name"] = "memory_benchmark_dense"
        parameters["dense_steps_to_keep"] = nb.int32(8)
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double",
        count,
        parameters=parameters,
    )


def _windows_memory() -> dict[str, int]:
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
    }


def _create_nbnet(path: Path, connections: int, neurons_per_layer: int, dense: bool) -> None:
    options = nb.NbnetWriteOptions(overwrite=True)
    with nb.NbnetDescriptionBuilder(path, options=options) as builder:
        if dense:
            builder.add_layer(_lif_layer(neurons_per_layer, dense=True))
        else:
            builder.add_layer(nb.NeuronLayer.input_spike(neurons_per_layer))
            builder.add_layer(_lif_layer(neurons_per_layer))
        batch_size = 20_000
        for first in range(0, connections, batch_size):
            count = min(batch_size, connections - first)
            source = [(first + index) % neurons_per_layer for index in range(count)]
            target_offset = 0 if dense else neurons_per_layer
            target = [target_offset + ((first + index * 17) % neurons_per_layer) for index in range(count)]
            builder.append_connections(
                source,
                target,
                weight=0.25,
                max_weight=0.25,
                delay=1,
            )


def _worker(mode: str, path: Path, connections: int, neurons_per_layer: int, budget_mb: int, dense: bool) -> None:
    tracemalloc.start()
    started = time.perf_counter()
    config = nb.SimulationConfig(steps=1, timestep=0.1)
    if mode == "fast":
        source = [index % neurons_per_layer for index in range(connections)]
        target_offset = 0 if dense else neurons_per_layer
        target = [target_offset + ((index * 17) % neurons_per_layer) for index in range(connections)]
        network = nb.Network()
        if dense:
            network.add_layer(_lif_layer(neurons_per_layer, dense=True))
        else:
            network.add_layer(nb.NeuronLayer.input_spike(neurons_per_layer))
            network.add_layer(_lif_layer(neurons_per_layer))
        network.connect(
            nb.Connection(
                source=source,
                target=target,
                weight=0.25,
                max_weight=0.25,
                delay=1,
            )
        )
        simulation = nb.Simulation(network, config)
    else:
        simulation = nb.Simulation(
            path,
            config,
            build_options=nb.StreamingBuildOptions(
                memory_budget_mb=budget_mb,
                mmap=True,
                verify_checksum=True,
            ),
        )
    _, python_peak = tracemalloc.get_traced_memory()
    result = {
        "mode": mode,
        "connections": connections,
        "neurons": neurons_per_layer * (1 if dense else 2),
        "dense": dense,
        "elapsed_seconds": time.perf_counter() - started,
        "python_peak_bytes": python_peak,
        "build_stats": simulation.build_stats,
        **_windows_memory(),
    }
    print(json.dumps(result, sort_keys=True))


def _run_worker(mode: str, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--path",
        str(args.path),
        "--connections",
        str(args.connections),
        "--neurons-per-layer",
        str(args.neurons_per_layer),
        "--budget-mb",
        str(args.budget_mb),
    ]
    if args.dense:
        command.append("--dense")
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=os.environ.copy())
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("fast", "streaming"))
    parser.add_argument("--path", type=Path, default=Path("artifacts/streaming_build/memory_network.nbnet"))
    parser.add_argument("--connections", type=int, default=500_000)
    parser.add_argument("--neurons-per-layer", type=int, default=2048)
    parser.add_argument("--budget-mb", type=int, default=16)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/streaming_build/memory_comparison.json"))
    args = parser.parse_args()
    args.path = args.path.resolve()
    if args.worker:
        _worker(args.worker, args.path, args.connections, args.neurons_per_layer, args.budget_mb, args.dense)
        return

    args.path.parent.mkdir(parents=True, exist_ok=True)
    _create_nbnet(args.path, args.connections, args.neurons_per_layer, args.dense)
    fast = _run_worker("fast", args)
    streaming = _run_worker("streaming", args)
    peak_reduction = 1.0 - streaming["peak_working_set_bytes"] / fast["peak_working_set_bytes"]
    python_reduction = 1.0 - streaming["python_peak_bytes"] / fast["python_peak_bytes"]
    report = {
        "platform": sys.platform,
        "python": sys.version,
        "fast": fast,
        "streaming": streaming,
        "peak_working_set_reduction_fraction": peak_reduction,
        "python_peak_reduction_fraction": python_reduction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
