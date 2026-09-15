# NeuronBridge

NeuronBridge is a Windows-first C++/CUDA neural simulation runtime with a
Python 3.12 interface.

This repository is being migrated from the historical `network_release`
workspace. The Windows dependency bundle and the behavior-preserving native
C++/CUDA runtime migration are complete. Maintainer-owned model code generation
is integrated and validated for Legacy CPU, Legacy CUDA, and Dense CUDA. The
Python 3.12 API and pybind11 bindings are integrated; user-facing examples and
release packaging remain staged work.

## Supported platform

- Windows 10/11 x64
- Visual Studio 2022 with the Desktop development with C++ workload
- CMake 3.24 or newer
- CUDA Toolkit 12.x
- Python 3.12
- Git LFS

## Repository status

See `docs/migration/migration_plan.md` for the staged migration status. Do not
publish a public release until a project-level `LICENSE` has been approved.

## Native developer workflow

```powershell
.\scripts\doctor.ps1
.\scripts\bootstrap_dependencies.ps1
cmake --preset windows-msvc-cuda
cmake --build --preset windows-msvc-cuda-release
ctest --test-dir build\windows-msvc-cuda -C Release --output-on-failure
```

Run these commands from a Visual Studio 2022 developer PowerShell. The native
build exports the CMake targets `NeuronBridge::CoreCpp`,
`NeuronBridge::CoreCuda`, `NeuronBridge::DenseRuntime`, and the complete
application-facing `NeuronBridge::Runtime` target.

## Python developer workflow

The default build also creates `neuronbridge._core` because
`NR_ENABLE_PYTHON=ON`. Configure with a Python 3.12 interpreter and its
pybind11 CMake package, then build `neuronbridge_python_core`. The importable
build-tree package is written under
`build\windows-msvc-cuda\python\Release\neuronbridge`; see
`python/README.md` for commands and API ownership.
