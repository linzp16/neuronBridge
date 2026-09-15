# Stage 6 examples and baselines audit

## Scope

Stage 6 migrates the previously aligned Python examples and their regression
tests into the formal repository. It does not change the pybind11 surface or
native simulation behavior. Wheel construction, GitHub CI, and distribution of
large optional datasets remain Stage 7 work.

## Migrated examples

The 26 historical scripts are preserved under `examples/python`: seven shared
helpers and nineteen runnable entry points. They cover basic network
construction, native lifecycle and weight I/O, Dense CUDA, InputConv,
sync/async ZMQ, OuterDynamic/cerebellum, EI networks, pattern motion, and
handwriting stages 1 through 4.

All scripts use the public `neuronbridge` API. A source scan rejects direct
`neuronbridge._core` use. Shared path handling removes historical absolute
paths, writes output under `artifacts/examples`, and resolves optional data
from `NEURONBRIDGE_DATA_ROOT` or `examples/data`.

## Test classification

- Unit/API tests remain in `python/tests`.
- Native C++ smoke tests remain in `tests/native`.
- Model generation and CPU/GPU numerical baselines remain in
  `tests/model_codegen`.
- Migrated end-to-end Python examples and communication protocol tests live in
  `tests/python/test_migrated_examples.py`.
- Large-data reproduction checks use explicit optional-data skips. Synthetic
  fixtures still execute the handwriting stage-3 NumPy runtime by default.

The example integration suite is controlled by `NR_BUILD_EXAMPLES` and carries
the CTest labels `python`, `examples`, and `integration`.

## Data policy

The historical handwriting inputs are approximately 0.58 to 0.81 GB per
stage and include duplicate MAT/binary forms. They are excluded from Git and
from the wheel. `examples/data/manifest.json` records the logical layout and
size estimates. Stage 7 must create a versioned optional data archive with
checksums before those data-dependent baselines can become release gates.

## Recorded validation

On Windows with MSVC 2022, CUDA 12.6, and Python 3.12.14, the migrated example
suite completed with 64 passed and 5 skipped tests. This includes a
parameterized import check for every migrated script. The skips were only the
four handwriting external-data checks and the EI connectivity check. All
data-free native, Dense, InputConv, ZMQ, pattern-motion, OuterDynamic,
visualization, and synthetic handwriting checks passed.

The complete Release CTest matrix passed 17 of 17 tests, including every
dependency, native, model-codegen, Python API, and migrated-example gate from
Stages 2 through 6.
