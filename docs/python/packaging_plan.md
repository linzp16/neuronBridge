# NeuronBridge packaging strategy

## Release baseline

NeuronBridge targets CPython 3.12 on Windows x64. The maintained release build
uses MSVC, CUDA, the bundled native dependency revision from
`dependencies/manifest.json`, scikit-build-core, pybind11, and delvewheel.
Build tools are maintainer dependencies and are not installed for users.

`NR_ENABLE_PYTHON` and `NR_ENABLE_CUDA` remain enabled for release builds so
Python and CUDA integration failures surface during normal validation.

## Python wheel

The Windows wheel contains:

- The public `neuronbridge` Python package.
- `neuronbridge._core` built for CPython 3.12.
- CUDA runtime, Pinocchio, ZeroMQ, OpenMP, and required MSVC runtime DLLs.

The CMake install step places direct runtime DLLs beside `_core`. Delvewheel
then analyzes those binaries, adds the transitive non-system DLL closure, and
patches package startup. Existing direct DLLs are analyzed instead of copied a
second time.

Build and validate with:

```powershell
$env:NEURONBRIDGE_PYTHON = "C:\Python312\python.exe"
.\scripts\bootstrap_dependencies.ps1
.\scripts\build_neuronbridge_wheel.ps1
.\scripts\check_neuronbridge_runtime_deps.ps1
.\scripts\validate_neuronbridge_wheel.ps1
```

Static inspection rejects debug/import-library products and requires the native
extension, CUDA runtime, Pinocchio, and ZeroMQ. Dynamic inspection installs the
wheel with `--no-deps` into a new venv and runs from an isolated directory. The
CUDA acceptance exercises `CustomLifConductanceV1`, `CustomRStdpV1`,
`CustomRStdpPersistentV1`, and `CustomPairStdpV1` through the public API.

Wheel users provide only CPython 3.12 and a compatible NVIDIA driver. They do
not install Visual Studio, CMake, CUDA Toolkit, pybind11, Pinocchio, or ZeroMQ.

## Portable offline package

Users who should not install Python receive the portable ZIP built by
`build_neuronbridge_offline_bundle.ps1`. It contains:

- A reviewed Python 3.12 runtime under `runtime/`.
- NeuronBridge and all Python dependencies preinstalled in
  `runtime/Lib/site-packages`.
- NumPy, Matplotlib, Pandas, and pyzmq with their transitive dependencies.
- Native runtime DLLs already carried by the repaired wheel.
- Migrated examples, third-party notices, launch scripts, and SHA-256 hashes.

The runtime can come from a reviewed embeddable archive or an approved Python
runtime root. Dependency resolution is performed with `--no-index` against a
provided wheelhouse unless the maintainer explicitly requests downloads.

```powershell
.\scripts\build_neuronbridge_offline_bundle.ps1 `
  -PythonRuntimeRoot C:\approved\python312-runtime `
  -DependencyWheelhouse C:\approved\cp312-wheelhouse
```

After extraction, `neuronbridge-python.ps1` is the Python entry point and
`run_smoke_test.ps1` verifies the packaged interpreter, native extension,
visualization stack, communication stack, and an example CLI. No environment
installation is required. CUDA execution still depends on the machine's NVIDIA
driver.

## Source and data archives

`build_source_package.ps1` creates a pure source ZIP by default. Passing
`-IncludeDependencyBundle` creates the larger source-with-Windows-dependencies
variant and rejects unresolved Git LFS pointers.

Large handwriting and EI datasets never enter Git or the wheel.
`package_example_data.ps1` creates their separately versioned archive using the
layout recorded in `examples/data/manifest.json` and writes per-file hashes.

`write_release_checksums.ps1` creates one `artifacts/SHA256SUMS.txt` for final
wheel and ZIP artifacts.

## Automation boundary

GitHub-hosted Windows CI validates metadata, the model generator, source-only
Python APIs, PowerShell syntax, and the dependency archive. A self-hosted runner
labelled `neuronbridge-cuda` owns full CUDA compilation, CTest, wheel repair,
and installed-wheel execution.

Public release remains blocked until a project-level `LICENSE` is approved and
the bundled dependency redistribution review is complete.
