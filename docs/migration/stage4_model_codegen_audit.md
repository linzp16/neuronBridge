# Stage 4 model code-generation audit

Date: 2026-09-15

## Scope

Stage 4 migrates four maintainer-owned `.nbmodel.json` descriptors, the generic
Python 3.12 descriptor-to-C++/CUDA generator, CMake integration, generated
Catalog/Factory registration, and numerical baselines. It does not migrate the
pybind11 package, wheel build, or user-facing examples.

Generated files remain in the build tree. Specifications, emitters, and tests
are source-controlled. The workflow does not provide an end-user custom-model
compiler.

## Build integration

- `NR_ENABLE_MODEL_CODEGEN` defaults to `ON`.
- A Python 3.12.x interpreter is required when the option is enabled.
- One custom target emits aggregate CPU and CUDA translation units, headers,
  registry fragments, Catalog entries, and `manifest.json`.
- `NeuronBridge::CoreCpp`, `NeuronBridge::CoreCuda`, and
  `NeuronBridge::DenseRuntime` consume the same generated output.
- With code generation disabled, the handwritten runtime compiles the empty
  generated Catalog providers.
- MSVC and the CUDA host compiler use UTF-8 source decoding so clean Git clones
  do not depend on CRLF to parse UTF-8 comments correctly.

## Toolchain

- Visual Studio 2022 Community 17.11.4
- MSVC 19.41.34120
- CUDA Toolkit 12.6.20
- Python 3.12.14
- bundled dependency set `windows-x64-msvc/r1`

## Enabled-path evidence

The generator manifest records four specifications and their SHA-256 hashes:
`CustomLifConductanceV1`, `CustomPairStdpV1`,
`CustomRStdpPersistentV1`, and `CustomRStdpV1`.

CTest result: 10/10 passed.

| Validation | Result |
| --- | --- |
| Generator unit tests | 33/33 passed |
| CPU LIF state comparison | 288,000 comparisons, max error 0 |
| CPU LIF simulation | 6 cases, 1,116 matched spikes, max state error 0 |
| Generated CPU R-STDP | 723,600 comparisons, max error 0 |
| R-STDP CPU/GPU | 2 rules, 22 samples, max weight error 1.90735e-06 |
| Dense CUDA LIF | 160 samples, 20 spikes, max error 0 |
| Legacy CUDA LIF | 1,000 samples, 115 spikes, max error 3.8147e-06 |
| Pair STDP CPU/GPU | 17 samples, max error 0 |
| CUDA math functions | CPU 7.6714, GPU 7.6714 |
| Catalog/Factory checks | passed |

CPU R-STDP traces are written under `model_codegen/rstdp_trace`; independent
CPU/GPU traces are written under `model_codegen/rstdp_gpu_trace` in the build
tree. No real-time Dense weight accessor is introduced.

## Disabled-path evidence

A separate build configured with `NR_ENABLE_MODEL_CODEGEN=OFF` compiled the
empty Catalog providers and passed 3/3 tests: bundled dependencies, handwritten
Legacy LIF, and Dense CUDA smoke. No generated model output is required by this
path.

## Stage gate

Stage 4 is accepted when the same checks pass from the canonical repository.
Stage 5 may then migrate pybind11 and the Python API against these registered
native models. Wheel and offline-package acceptance remain later release work.
