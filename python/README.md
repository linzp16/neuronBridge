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
`build/<tree>/python/<Config>/neuronbridge`. It never writes `_core.pyd` or
runtime DLLs into the source package.

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
