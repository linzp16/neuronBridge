# Stage 5 Python bridge audit

## Scope

Stage 5 migrates the pybind11 extension and stable `neuronbridge` Python API
onto the Stage 3/4 target graph. It deliberately excludes example migration,
final wheel repair, offline installation, and GitHub CI, which remain Stage 6
and Stage 7 work.

## Implemented boundary

- Python is fixed to CPython 3.12.x and pybind11 is the maintained binding
  mechanism.
- `neuronbridge_python_core` links only `NeuronBridge::Runtime`; it does not
  compile private copies of ZMQ, OuterDynamic, or simulation sources.
- The user-facing binary remains `neuronbridge._core`.
- CMake assembles pure Python files, `_core.pyd`, Pinocchio, Boost
  serialization, ZeroMQ, and CUDA runtime DLLs in the build tree.
- The source package remains free of generated binaries.
- `NR_ENABLE_PYTHON` remains ON by default so Python dependency and binding
  failures appear in normal development builds.
- The Python API includes model/network descriptions, `Simulation`, weight
  file I/O, InputConv frame protocols and sync/async adapters, Dense runtime
  snapshots, OuterDynamic state access, debug-monitor readers, weight plots,
  and OuterDynamic plots.
- Maintainer-generated models are discoverable through
  `catalog.list_models(public_only=False)` and use the same parameter query as
  handwritten models. No end-user code generator is exposed.

## Validation matrix

The Stage 5 gate requires all of the following from a clean Windows MSVC 2022
Release build with CUDA 12.6 and Python 3.12:

| Check | Expected |
| --- | --- |
| Configure with Python ON | Python 3.12 and pybind11 found |
| Build `neuronbridge_python_core` | `_core.cp312-win_amd64.pyd` produced |
| Python API tests | frame protocol and visualization pass |
| Catalog tests | handwritten and generated models pass |
| Native Python smoke | minimal LIF simulation runs through pybind11 |
| Full CTest regression | native, codegen, dependency, and Python tests pass |
| Configure with Python OFF | no pybind11 dependency is required |

## Runtime and release boundary

Copying runtime DLLs beside `_core.pyd` makes the build-tree package directly
loadable and records the dependency set needed by packaging. Stage 5 does not
claim wheel portability. Stage 7 must install these DLLs into the wheel, run a
clean-environment wheel test, inspect imported DLL dependencies, and produce
the offline dependency package promised to users.

## Recorded result

The clean formal-repository build at `build/stage5-release` produced
`_core.cp312-win_amd64.pyd` and deployed its four non-system runtime DLLs next
to the extension. CTest passed 16 of 16 tests: three Python bridge tests, one
dependency probe, two native smoke tests, and ten model-codegen/catalog CPU and
GPU tests. A separate configure with Python, CUDA, model codegen, and tests
disabled also completed without discovering Python or pybind11.

## Tool environment

Validation reuses, without modifying, the Python 3.12 environment at
`D:\code_optimized\network_release\.conda-neuronbridge-py312`. The relevant
tools are pybind11 3.1.0 and pytest 9.1.1; runtime validation uses NumPy 2.5.2,
Matplotlib 3.11.1, and pyzmq 27.2.0. No package was installed during Stage 5.
