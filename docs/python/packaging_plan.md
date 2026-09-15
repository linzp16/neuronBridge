# neuronbridge packaging strategy

## Local validation environment

The current native-extension validation uses a project-local virtual environment:

```text
D:\code_optimized\network_release\.venv-neuronbridge
```

Installed Python build tools are under:

```text
D:\code_optimized\network_release\.venv-neuronbridge\Lib\site-packages
```

Python 3.12 is the release baseline. The current Python 3.12 validation environment is:

```text
D:\code_optimized\network_release\.conda-neuronbridge-py312
```

Installed Python build tools for the Python 3.12 baseline are under:

```text
D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages
```

The older `.venv-neuronbridge` Python 3.14 environment was used only for the first pybind11 smoke test and should not be treated as the release target.

## Wheel distribution

The normal Python distribution should be a Windows wheel containing:

- `neuronbridge` Python package files.
- `neuronbridge._core` native extension.
- Required non-system runtime DLLs copied next to the extension module.

The wheel must not require users to install:

- Visual Studio or MSVC.
- CMake.
- pybind11.
- scikit-build-core.
- Pinocchio or ZeroMQ separately.
- Project-private native runtime DLLs separately.

The wheel may still require:

- A compatible Python 3.12 interpreter.
- NVIDIA GPU driver when GPU simulation is used.

`python312.dll` is normally provided by the user's Python 3.12 installation and should not be treated as a wheel-bundled project dependency.

## Portable offline distribution

For users who should not install Python manually, provide a separate portable package that includes:

- Embedded or packaged Python runtime.
- The repaired `neuronbridge` wheel.
- Python dependencies for visualization.
- Native runtime DLLs.
- Example scripts and small validation datasets.
- Launch scripts that set `PYTHONPATH`/`PATH` internally.

This is the correct format for "no extra environment installation" delivery. It is broader than a normal Python wheel because it owns the Python runtime too.

## CMake default

`NR_ENABLE_PYTHON` defaults to `ON` because `neuronbridge` is the primary release path. This makes Python 3.12, pybind11, Python include/lib, and native dependency packaging problems visible during normal development and CI instead of appearing only at final release time.

Traditional C++-only validation can still opt out explicitly:

```powershell
cmake -S . -B build_cpp_only -DNR_ENABLE_PYTHON=OFF
```

## Native dependency handling

The current CMake build copies many runtime DLLs next to `_core*.pyd` during local builds. A release pipeline should still validate the final artifact in a clean environment, because local PATH entries can hide missing packaged DLLs.

Recommended release checks:

- Build `neuronbridge_core` in Release mode through the Visual Studio Developer environment.
- Repair or collect DLL dependencies for the wheel.
- Install the produced wheel into a fresh virtual environment.
- Run `import neuronbridge` and `neuronbridge.backend_info()`.
- Open an existing DebugMonitor output directory.
- Run a small visualization smoke test without requiring a compiler on the target machine.

## Current validation result

Validated locally on this machine with the Python 3.12 release baseline:

- Visual Studio 2022 Community, MSVC 19.41.
- CUDA Toolkit 12.6.
- Python 3.12.14 in `.conda-neuronbridge-py312`.
- pybind11 3.1.0.
- pytest 9.1.1.
- scikit-build-core 1.0.3.
- numpy 2.5.2.
- pyzmq 27.2.0 for optional ZMQ peer validation.

The native extension built successfully:

```text
D:\code_optimized\network_release\python\src\neuronbridge\_core.cp312-win_amd64.pyd
```

The native import smoke test passed and reported `binding=pybind11`.

## Current wheel validation

The first Windows wheel has been generated:

```text
D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl
```

Build script:

```powershell
D:\code_optimized\network_release\scripts\build_neuronbridge_wheel.ps1
```

Validation script:

```powershell
D:\code_optimized\network_release\scripts\validate_neuronbridge_wheel.ps1
```

Runtime dependency inspection script:

```powershell
D:\code_optimized\network_release\scripts\check_neuronbridge_runtime_deps.ps1
```

The wheel excludes stale `neuronbridge/Release/**` build outputs and contains the Python 3.12 extension plus same-directory runtime DLLs. Clean venv validation was run with `--no-deps` semantics and passed:

```text
native_extension_loaded=True
binding=pybind11
cuda_enabled=True
dense_runtime_enabled=True
```

## Current offline package validation

The first Python 3.12 offline package directory has been generated:

```text
D:\code_optimized\network_release\release\neuronbridge_offline_py312
```

The matching archive for handoff has also been generated:

```text
D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip
```

Bundle script:

```powershell
D:\code_optimized\network_release\scripts\build_neuronbridge_offline_bundle.ps1
```

The package contains:

- `wheels\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- Dependency wheels for `numpy 2.5.2`, `matplotlib 3.11.1`, `pandas 3.0.5`, and `pyzmq 27.2.0`.
- Transitive visualization dependencies: `contourpy`, `cycler`, `fonttools`, `kiwisolver`, `packaging`, `pillow`, `pyparsing`, `python-dateutil`, `six`, and `tzdata`.
- Migrated Python examples.
- Packaging/migration notes.
- Offline install and smoke-test scripts.

Offline install validation was run from local wheels only:

```powershell
D:\code_optimized\network_release\release\neuronbridge_offline_py312\install_neuronbridge_offline.ps1
```

The validation venv is:

```text
D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-offline-validate
```

The native import smoke test passed:

```text
native_extension_loaded=True
binding=pybind11
cuda_enabled=True
dense_runtime_enabled=True
```

The package smoke test also ran `examples\dense_run_no_debug.py --help` successfully.

Remaining release hardening:

- Add or reference an embedded Python 3.12 runtime for users without Python.
- Validate the package on a clean target machine without Visual Studio, CMake, pybind11, or project build artifacts on PATH.
