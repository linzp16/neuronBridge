# NeuronBridge Python API

This directory owns the Python 3.12 API and the pybind11 bridge to the native
runtime. The extension is built as `neuronbridge._core`; the CMake target is
`neuronbridge_python_core` and the exported in-tree alias is
`NeuronBridge::PythonCore`.

## Source layout

- `bindings/`: pybind11 bindings for `Simulation`, model discovery, runtime
  state, weight I/O, InputConv, Dense, and OuterDynamic access.
- `src/neuronbridge/`: stable Python-facing descriptions, communication
  adapters, frame protocol helpers, result readers, and plotting API.
- `tests/`: API tests that do not depend on migrated examples.
- `pyproject.toml`: Python 3.12 package metadata used by the Stage 7 wheel
  workflow.

The normal CMake build assembles an importable package under
`build/<tree>/python/<Config>/neuronbridge`. It never writes `_core.pyd`,
`_core.so`, or runtime libraries into the source package.

## Temporarily deferred APIs

The following experimental real-time interfaces remain in the source for
future development, but are temporarily deferred and are not part of the
current stable support or compatibility contract:

- `Simulation.enable_realtime()`
- `Simulation.run_realtime()`
- `Simulation.disable_realtime()`
- `Simulation.reset_bench_profiling()`
- `Simulation.bench_profiling_snapshot()`
- `Simulation.realtime_skip_counters()`
- `Simulation.reset_realtime_skip_counters()`
- `Simulation.realtime_restriction_counts()`
- `Simulation.reset_realtime_restriction_counts()`

Do not depend on these interfaces in production workflows. Their timing,
counter, behavior, and backward-compatibility guarantees will be defined only
after the real-time design and validation work resumes.

## Low-memory `.nbnet` construction

Keep using the in-memory API for small networks:

```python
sim = nb.Simulation(network, config)
```

For a large static topology, generate a description file incrementally and
load it through the path overload:

```python
with nb.NbnetDescriptionBuilder("large_network.nbnet") as builder:
    builder.add_layer(input_layer)
    builder.add_layer(output_layer)
    for batch in generate_connection_batches():
        builder.append_connections(
            batch.source,
            batch.target,
            weight=batch.weight,
            max_weight=batch.max_weight,
            delay=batch.delay,
        )

sim = nb.Simulation(
    "large_network.nbnet",
    config,
    build_options=nb.StreamingBuildOptions(
        memory_budget_mb=128,
        mmap=True,
        verify_checksum=True,
    ),
)
```

`NbnetDescriptionBuilder` generates the static network description; it does
not create or run a simulation. `nb.nbnet.from_network()` is convenient for
conversion and equivalence tests, but an already materialized `Network` still
incurs its original Python memory cost. Use the incremental builder or
`nb.nbnet.from_batches()` to avoid that cost.

## Scientific test environment

NeuronBridge scientific tests require a NumPy 2.x-compatible plotting stack:

```powershell
python -m pip install --upgrade "numpy>=2,<3" "matplotlib>=3.9" "contourpy>=1.3"
python scripts/check_scientific_runtime.py
```

The release wheel bundles MSVC/OpenMP runtime DLLs under NeuronBridge-private
names (for example `neuronbridge_vcomp140.dll`). The build patches every native
PE dependency to those private names, preventing collisions with runtimes
already loaded by Python, NumPy, or Matplotlib. This is the supported Windows
packaging mode; do not rename these DLLs manually.

Run the runtime check before testing:

```powershell
python scripts/check_scientific_runtime.py
python scripts/check_neuronbridge_runtime_deps.ps1 -PythonExe (Get-Command python).Source
```

## Developer build

Install build tools into the selected Python 3.12 environment, then point CMake
at that interpreter and pybind11 package:

```powershell
python -m pip install pybind11 pytest
cmake --preset windows-msvc-cuda `
  -DPython3_EXECUTABLE=C:\path\to\python.exe `
  -Dpybind11_DIR=C:\path\to\site-packages\pybind11\share\cmake\pybind11
cmake --build --preset windows-msvc-cuda-release --target neuronbridge_python_core
ctest --test-dir build\windows-msvc-cuda -C Release -R neuronbridge_python --output-on-failure
```

Examples and final self-contained wheel assembly remain Stage 6 and Stage 7
work respectively.

## Linux CUDA full-feature build

The Linux build is CUDA-enabled but retains the same ordinary CPU neurons,
OpenMP queues, Dense CPU/GPU subnetworks, communication, learning rules,
OuterDynamic, and DebugMonitor functionality as Windows. It is not a GPU-only
feature subset and it is not a CPU-only wheel.

With the validated Linux dependency bundle installed, configure and build with:

```bash
export NR_DEPENDENCY_ROOT=/opt/neuronbridge/dependencies/linux-x86_64-gcc-cuda12/r1
cmake --preset linux-gcc-cuda-release
cmake --build --preset linux-gcc-cuda-release --target neuronbridge_python_core
ctest --preset linux-gcc-cuda-release
```

Build the repaired manylinux wheel with:

```bash
PYTHON_EXE=python3.12 \
NR_DEPENDENCY_ROOT="$NR_DEPENDENCY_ROOT" \
bash scripts/build_neuronbridge_wheel.sh
```

The Linux wheel uses `auditwheel`; `libcuda.so.1` remains owned by the host
NVIDIA driver and must not be bundled.
