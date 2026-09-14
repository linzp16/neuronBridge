# Windows development baseline

The initial NeuronBridge release targets Windows x64 with Visual Studio 2022,
CUDA 12.x, CMake 3.24 or newer, and Python 3.12.

Run the environment check from the repository root:

```powershell
.\scripts\doctor.ps1
```

The repository-bootstrap stage does not yet contain the simulation core or a
bundled dependency archive. Configure, build, test, and package commands are
added only when their implementation stages are complete.

No script may install files into Windows system directories or permanently
modify the user or system `PATH`.
