# Release process

A NeuronBridge release is built from a clean annotated Git tag. Release
artifacts are uploaded to GitHub Releases and are not committed to the normal
Git history.

Planned Windows artifacts:

- Pure source archive.
- Source archive with the Windows x64 dependency bundle.
- Standalone dependency bundle.
- Python 3.12 Windows wheel.
- Self-contained Python 3.12 offline bundle.
- SHA-256 checksums and validation summary.

The release workflow verifies version consistency, dependency hashes, native
builds, Python tests, CUDA baselines, repaired-wheel imports, and third-party
notices before creating release staging. Generated files are written only under
`artifacts/` and are uploaded to a release rather than committed.

## Local artifact workflow

Use Python 3.12 from a Visual Studio 2022 developer-capable machine with CUDA:

```powershell
$env:NEURONBRIDGE_PYTHON = "C:\Python312\python.exe"
.\scripts\bootstrap_dependencies.ps1
.\scripts\build_neuronbridge_wheel.ps1
.\scripts\check_neuronbridge_runtime_deps.ps1
.\scripts\validate_neuronbridge_wheel.ps1
.\scripts\build_source_package.ps1
.\scripts\write_release_checksums.ps1
```

The installed-wheel validator runs from a fresh venv and an isolated working
directory. It rejects source/build paths in `sys.path` and executes generated
Dense CUDA neurons plus three generated learning rules.

For a portable package, provide a reviewed Python 3.12 runtime and a complete
CPython 3.12 wheelhouse:

```powershell
.\scripts\build_neuronbridge_offline_bundle.ps1 `
  -PythonRuntimeRoot C:\approved\python312-runtime `
  -DependencyWheelhouse C:\approved\wheelhouse
```

The result contains `runtime/python.exe`, preinstalled Python dependencies,
native DLLs, examples, third-party notices, and per-file SHA-256 hashes. It does
not require users to install Python, MSVC, CMake, pybind11, ZeroMQ, or the CUDA
Toolkit. GPU use still requires a compatible NVIDIA driver.

The optional research datasets are distributed separately:

```powershell
.\scripts\package_example_data.ps1 -DataRoot D:\approved\neuronbridge-data
```

## GitHub automation

`.github/workflows/ci.yml` runs the CPU-compatible C++/Python matrix on the
GitHub-hosted Windows 2022 runner. `.github/workflows/windows-cuda-release.yml`
uses a runner labelled `self-hosted`, `Windows`, `X64`, and
`neuronbridge-cuda` for CUDA builds and repaired-wheel validation.

Public release is blocked until a project-level `LICENSE` is approved.
