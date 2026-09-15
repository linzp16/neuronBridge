# Model code generator

This standard-library-only Python package validates maintainer-owned model
descriptors and emits Legacy CPU, Legacy CUDA, Dense CUDA, Factory registry,
and Catalog integration sources. Generated files live in the CMake build tree;
the descriptor, generator, and baseline tests are the maintained source of
truth.

Run a validation-only pass with Python 3.12:

```powershell
python -m tools.model_codegen.cli --spec-root models/specs `
  --output-root build/generated/model_codegen `
  --manifest build/generated/model_codegen/manifest.json --check
```

Normal CMake builds run the emitter automatically when
`NR_ENABLE_MODEL_CODEGEN=ON`.
