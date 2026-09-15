# Model code-generation documentation

Stage 4 migrated the original design records in this directory so their
decisions and implementation history remain available. Paths beginning with
`source_file/` or `model_specs/`, and dated statements such as "not yet
implemented", describe the historical `network_release` workspace at that
point in time.

The current source locations are:

- specifications: `models/specs`
- generator: `tools/model_codegen`
- CMake integration: `cmake/NeuronBridgeModelCodegen.cmake`
- generated build output: `build/<preset>/generated/model_codegen`
- runtime baselines: `tests/model_codegen`

The current implementation and validation authority is
`../migration/stage4_model_codegen_audit.md`. Code generation is a maintainer
workflow. Official wheels contain precompiled implementations; users do not
compile descriptors or install a compiler toolchain.
