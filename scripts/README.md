# Release scripts

Release scripts resolve paths from the repository root. Generated files stay
under `build/` or `artifacts/`. Windows release entry points use PowerShell;
the Linux CUDA wheel entry point uses Bash.

- `build_neuronbridge_wheel.ps1`: builds and repairs the Python 3.12 CUDA wheel.
- `check_neuronbridge_runtime_deps.ps1`: inspects wheel contents and DLL closure.
- `validate_neuronbridge_wheel.ps1`: installs into an isolated venv and runs CUDA.
- `build_neuronbridge_offline_bundle.ps1`: creates a Python-runtime-inclusive package.
- `build_source_package.ps1`: creates pure-source or source-with-dependencies ZIPs.
- `package_example_data.ps1`: creates the separately versioned optional data ZIP.
- `write_release_checksums.ps1`: hashes final wheel and ZIP artifacts.
- `build_release_artifacts.ps1`: orchestrates the complete local release gate.
- `build_neuronbridge_wheel.sh`: builds and repairs the full-feature Linux CUDA wheel.
- `check_linux_wheel_runtime.py`: validates ELF dependencies, RPATH, and auditwheel policy.
- `check_include_case.py`: checks quoted repository includes for Linux case correctness.
- `inspect_neuronbridge_wheel.py`: validates both Windows and Linux wheel layouts.
- `private_runtime_directory.py`: rewrites build-tree MSVC/OpenMP dependencies
  to package-private DLL names and fails if public runtime references remain.
- `private_runtime_wheel.py`: applies the same runtime isolation to the final
  Windows wheel and refreshes its `RECORD` metadata.

Set `NEURONBRIDGE_PYTHON` or pass `-PythonExe` to select Python 3.12. Public
release commands must use the metadata check with `--require-license`.
