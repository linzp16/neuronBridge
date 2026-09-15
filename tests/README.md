# Validation

Tests will be classified as unit, integration, communication, or numerical
baseline validation. Generated runtime output is written under `artifacts` and
is never committed.

`dependencies/` contains the Stage 2 native dependency configure, link, and
runtime probe.

`native/` contains C++ runtime smoke tests. `model_codegen/` owns generated
CPU/GPU numerical baselines and Catalog validation. `python/` contains the
migrated example integration suite, including communication negative tests,
Dense/InputConv quick baselines, synthetic handwriting stage-3 execution, and
optional external-data checks.

When `NR_BUILD_EXAMPLES=ON`, CTest registers
`neuronbridge_python_migrated_examples`. Missing optional data is reported as a
skip; missing native capabilities or behavioral regressions are failures.
