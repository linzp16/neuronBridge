# NeuronBridge

NeuronBridge is a Windows-first C++/CUDA neural simulation runtime with a
Python 3.12 interface.

This repository is being migrated from the historical `network_release`
workspace. The repository skeleton is complete, but the simulation core and
bundled dependency archive have not yet been imported.

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

## Planned developer workflow

```powershell
.\scripts\doctor.ps1
.\scripts\bootstrap_dependencies.ps1
.\scripts\configure.ps1 -Preset windows-msvc-cuda
.\scripts\build.ps1 -Configuration Release
.\scripts\test.ps1 -Suite all
```

Dependency packaging, bootstrap, and verification commands are available in
Stage 2. Configure, build, test, and release commands will be added with their
corresponding implementation stages.
