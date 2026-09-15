# Stage 7 release packaging audit

## Scope

Stage 7 establishes reproducible Windows release entry points for the repaired
CPython 3.12 wheel, portable Python package, source archives, optional research
data, checksums, and GitHub CI. It does not publish `0.1.0a0`; clean-tag release
rehearsal remains Stage 8.

## Implemented release surfaces

- CMake installs `_core.pyd`, the matching MSVC/OpenMP redistributables, and its
  direct CUDA, Pinocchio, Boost, and ZeroMQ runtime DLLs into both the build-tree
  import package and wheel staging tree.
- `build_neuronbridge_wheel.ps1` uses the standard `build` frontend and repairs
  the Windows wheel with delvewheel. Existing direct DLLs are analyzed without
  being duplicated.
- Static wheel inspection rejects build/debug files and requires the native,
  CUDA, Pinocchio, and ZeroMQ runtime entries.
- Installed-wheel validation uses a fresh venv, forbids source/build package
  paths, and executes three generated learning rules on Dense CUDA.
- The portable package owns Python 3.12, Python dependencies, native DLLs,
  examples, notices, launch scripts, and internal SHA-256 hashes.
- Source packaging supports both pure source and source with the Git LFS Windows
  dependency bundle. Optional large datasets have an independent archive tool.
- GitHub-hosted Windows CI owns metadata, generator, source API, dependency
  archive, and script syntax checks. Full CUDA and wheel validation use a
  self-hosted runner labelled `neuronbridge-cuda`.

## Executed evidence

The clean Release build used Visual Studio 2022, MSVC 19.41, CUDA 12.6, Python
3.12.14, pybind11 3.1.0, scikit-build-core 1.0.3, and pytest 9.1.1. The complete
CTest matrix passed 18 of 18 tests. The new release-tool suite passed 5 of 5.

The repaired wheel passed static and isolated runtime validation:

- File: `neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- Size: 4,319,722 bytes
- SHA-256: `fc223e412529821bbe29e890129fcb486aebbceec77013485846217aa7c150ae`
- Native DLL entries: 13, including MSVC/OpenMP redistributables and no
  forbidden build products
- Backend: pybind11, CUDA enabled, Dense runtime enabled
- `CustomRStdpV1`: weight `8.0 -> 13.345841407775879`
- `CustomRStdpPersistentV1`: weight `8.0 -> 15.503661155700684`
- `CustomPairStdpV1`: weight `8.0 -> 15.195880889892578`

Portable and source artifacts were also created and inspected:

- Portable Python package: 140,695,152 bytes; package-owned Python imported
  NeuronBridge, NumPy, Matplotlib, Pandas, and pyzmq and ran the 1000-step Dense
  smoke example.
- Pure source ZIP: 1,021,256 bytes.
- Source-with-Windows-dependencies ZIP: 282,144,643 bytes; the dependency LFS
  payload passed its registered size and SHA-256 checks before inclusion.
- Optional-data packaging was exercised with a partial fixture and produced the
  expected `<dataset>/input_data` layout and file manifest.

Stage 7 installed release tools only into:

`D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages`

The additions were `build 1.6.1`, `pyproject-hooks 1.2.0`, `delvewheel 1.13.1`,
and `pefile 2024.8.26`. No system Python installation was modified.

## Remaining gates

- Public redistribution remains blocked until the project owner approves a
  project-level `LICENSE` and completes the dependency license review.
- Real handwriting/EI datasets were not packaged; only the archive machinery
  was tested with a fixture.
- GitHub Actions files were parsed and their commands reproduced locally, but
  cannot run on GitHub until the repository and `neuronbridge-cuda` runner
  exist.
- The historical core still includes CUDA headers in nominal CUDA-off builds,
  so hosted CI intentionally performs source/generator checks rather than
  claiming a CUDA-free native build.

These items do not block completion of the Stage 7 packaging implementation.
They are explicit inputs to the Stage 8 clean-tag release rehearsal.
