# Repository layout

The repository is a monorepo because the native runtime, Python bindings,
generated models, and baseline tests evolve under one compatibility contract.

- `include/neuronbridge`: public C++ headers.
- `src/native/legacy`: behavior-preserving event, network, simulation, and
  handwritten model sources migrated in Stage 3.
- `src/native/gpu`: CUDA backend, Dense runtime, InputConv, and debug monitor
  sources migrated in Stage 3.
- `src/native/shared`: cross-runtime model catalog implementation.
- `build/<preset>/generated/model_codegen`: ephemeral C++ and CUDA output from
  maintainer-owned specifications; generated build products are not committed.
- `python/bindings`: pybind11 translation layer.
- `python/src/neuronbridge`: stable Python user API.
- `models/specs`: equation and event model descriptors.
- `tools/model_codegen`: generic descriptor-to-C++/CUDA generator.
- `examples`: user-facing C++ and Python examples.
- `tests`: unit, integration, protocol, and numerical baseline validation.
- `dependencies`: manifests, licenses, bundles, and local extraction area.
- `scripts`: Windows environment, build, validation, and release orchestration.

Build output and runtime data never share a directory with source files.

The `src/native` layout is intentionally transitional. The historical
`source_file_realtime_v1_async` include anchor remains intact so Stage 3 can be
verified as a source-identical move. Renaming includes or splitting handwritten
models into final ownership directories requires a later, separately tested
maintenance change.
