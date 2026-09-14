# Repository layout

The repository is a monorepo because the native runtime, Python bindings,
generated models, and baseline tests evolve under one compatibility contract.

- `include/neuronbridge`: public C++ headers.
- `src/core`: scheduling, network, events, and simulation implementation.
- `src/backends`: CPU and CUDA backend implementation.
- `src/models`: handwritten neuron, learning-rule, and outer-dynamic models.
- `src/generated`: maintainer-generated C++ and CUDA model sources.
- `python/bindings`: pybind11 translation layer.
- `python/src/neuronbridge`: stable Python user API.
- `models/specs`: equation and event model descriptors.
- `tools/model_codegen`: generic descriptor-to-C++/CUDA generator.
- `examples`: user-facing C++ and Python examples.
- `tests`: unit, integration, protocol, and numerical baseline validation.
- `dependencies`: manifests, licenses, bundles, and local extraction area.
- `scripts`: Windows environment, build, validation, and release orchestration.

Build output and runtime data never share a directory with source files.
