# Release scripts

All release scripts are Windows-first PowerShell entry points and resolve paths
from the repository root. Generated files stay under `build/` or `artifacts/`.

- `build_neuronbridge_wheel.ps1`: builds and repairs the Python 3.12 CUDA wheel.
- `check_neuronbridge_runtime_deps.ps1`: inspects wheel contents and DLL closure.
- `validate_neuronbridge_wheel.ps1`: installs into an isolated venv and runs CUDA.
- `build_neuronbridge_offline_bundle.ps1`: creates a Python-runtime-inclusive package.
- `build_source_package.ps1`: creates pure-source or source-with-dependencies ZIPs.
- `package_example_data.ps1`: creates the separately versioned optional data ZIP.
- `write_release_checksums.ps1`: hashes final wheel and ZIP artifacts.
- `build_release_artifacts.ps1`: orchestrates the complete local release gate.

Set `NEURONBRIDGE_PYTHON` or pass `-PythonExe` to select Python 3.12. Public
release commands must use the metadata check with `--require-license`.
