# Windows development baseline

The initial NeuronBridge release targets Windows x64 with Visual Studio 2022,
CUDA 12.x, CMake 3.24 or newer, and Python 3.12.

Run the environment check and restore the versioned dependency bundle from the
repository root:

```powershell
.\scripts\doctor.ps1
.\scripts\bootstrap_dependencies.ps1
```

Open a Visual Studio 2022 developer PowerShell, then configure, build, and test
the native runtime:

```powershell
cmake --preset windows-msvc-cuda
cmake --build --preset windows-msvc-cuda-release
ctest --test-dir build\windows-msvc-cuda -C Release --output-on-failure
```

`NR_ENABLE_MODEL_CODEGEN` defaults to `ON`. Configure must resolve a Python
3.12.x interpreter, then CMake generates maintained models into the build tree
before compiling the native targets. The generator itself uses only the Python
standard library.

The validation suite covers dependency loading, the handwritten LIF and Dense
CUDA smoke tests, generator unit tests, Catalog/Factory registration, CPU
trajectories, Legacy GPU and Dense GPU neurons, heterogeneous learning rules,
and CPU/GPU error reports. Set `-DNR_ENABLE_MODEL_CODEGEN=OFF` only when testing
the empty Catalog fallback and handwritten runtime independently.

No script may install files into Windows system directories or permanently
modify the user or system `PATH`.
