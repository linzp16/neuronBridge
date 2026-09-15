# Stage 3 native core audit

## Scope

Stage 3 migrates the current C++/CUDA runtime from the historical
`network_release` workspace without changing model equations, event flow,
state layout, class names, or numerical algorithms. Examples, Python bindings,
model specifications, and the code generator remain assigned to later stages.

The migration contains 369 source and header files. A SHA-256 comparison of
every migrated file against the current historical workspace reported zero
mismatches.

## Source layout

- `src/native/legacy/source_file_realtime_v1_async` preserves the historical
  include anchor and contains the event, network, simulation, handwritten
  model, communication, InputConv, and OuterDynamic implementation.
- `src/native/gpu` contains production GPU headers and only the bridge, debug
  monitor, Dense runtime, GPU propagation, InputConv, and simulation support
  implementation directories used by the active build.
- `src/native/shared` contains the model catalogs and empty generated-catalog
  implementations used before Stage 4 is migrated.
- Historical executable entry points and GPU test/example sources are not part
  of Stage 3. `mainprogram/inc/BenchProfiling.h` is retained because the core
  event scheduler includes it directly.

## CMake targets

`cmake/NeuronBridgeNativeCore.cmake` defines these stable ownership boundaries:

| Target | Responsibility |
|---|---|
| `NeuronBridge::CoreCpp` | Legacy C++ engine and shared catalogs |
| `NeuronBridge::CoreCuda` | Main-network CUDA backend and GPU neuron interfaces |
| `NeuronBridge::DenseRuntime` | Dense CUDA runtime, InputConv, bridge, and monitors |
| `NeuronBridge::RuntimeSupport` | ZMQ communication drivers and OuterDynamic models |
| `NeuronBridge::Runtime` | Complete application-facing native runtime |

The lower-level targets remain useful for compile isolation. Applications must
link `NeuronBridge::Runtime`, matching the historical rule that CORE apps link
the complete Dense GPU-aware runtime and compile communication/OuterDynamic
support with the executable.

## Validation environment

- Windows 10/11 x64 host
- Visual Studio 2022 17.11.4
- MSVC 19.41.34120
- CMake 3.24 or newer
- CUDA Toolkit 12.6.20
- dependency bundle `windows-x64-msvc/r1`

Configuration used the Visual Studio 17 2022 x64 generator with
`NR_DEPENDENCY_MODE=BUNDLED`, `NR_ENABLE_CUDA=ON`, and `NR_BUILD_TESTS=ON`.

## Executed evidence

The migrated targets compiled and linked successfully:

- `neuronbridge_core_cpp.lib`
- `neuronbridge_core_cuda.lib`
- `neuronbridge_dense_runtime.lib`
- `neuronbridge_runtime_support.lib`
- `neuronbridge_native_lif_smoke.exe`
- `neuronbridge_dense_cuda_smoke.exe`

CTest completed three of three tests successfully:

| Test | Result | Key output |
|---|---|---|
| `neuronbridge_dependency_probe` | PASS | Pinocchio construction OK; ZeroMQ 4.3.5 |
| `neuronbridge_native_lif_smoke` | PASS | 2,000 steps; 36 spikes; voltage sum -118974 |
| `neuronbridge_dense_cuda_smoke` | PASS | two steps; one t0 firing event; zero output spikes |

The historical `dense_subnetwork_smoke` was rebuilt from the current old
workspace and passed. Its reference prefix reported `firing_t0: 0`,
`outputs_t0=0`, and `outputs_t1=0`; the migrated two-step fixture reported the
same firing and output behavior.

## Known warnings and deferred work

Several historical headers produce MSVC C4819 code-page warnings, and one CUDA
header reports legacy Mac line endings. Stage 3 leaves their bytes unchanged so
the migration remains auditable. Encoding and line-ending normalization should
be an isolated maintenance commit with a full regression run.

Stage 3 uses dependency compatibility bundle r1. Runtime tracing and bundle
reduction can now begin, but redistribution licensing, project-level `LICENSE`,
Python 3.12 bindings, code generation, wheel assembly, and public release
acceptance are not yet complete.
