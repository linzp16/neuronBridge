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

The Stage 3 native tests cover dependency loading, a deterministic handwritten
LIF trajectory, and a two-step Dense CUDA firing baseline. Python and model
code-generation tests are added by their later migration stages.

No script may install files into Windows system directories or permanently
modify the user or system `PATH`.
