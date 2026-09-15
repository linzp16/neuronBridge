# Staged migration plan

## Status

- [x] Stage 0: record the source repository and migration boundary.
- [x] Stage 1: create the clean repository skeleton and Git policy.
- [x] Stage 2: audit and package Windows bundled dependencies.
- [x] Stage 3: migrate the C++/CUDA core without behavior changes.
- [x] Stage 4: migrate model specifications and the code generator.
- [ ] Stage 5: migrate pybind11 bindings and the Python API.
- [ ] Stage 6: classify and migrate examples, tests, and baselines.
- [ ] Stage 7: establish wheel, offline package, and GitHub CI workflows.
- [ ] Stage 8: perform the `0.1.0a0` release rehearsal from a clean tag.

Each stage must produce an isolated commit and validation record. Directory
movement, behavior changes, and optimization must not be combined in one
migration commit.

## Stage gates

Stage 3 may start because dependency bundle r1 can be restored, configured,
linked, and executed without using the historical workspace paths. Public
release remains blocked until the project owner approves a project-level
license and copyright holder, and the dependency license review is complete.

Stage 4 is complete. Four maintained descriptors, the generic generator,
Catalog/Factory integration, and CPU/Legacy GPU/Dense GPU baselines pass with
code generation enabled. The handwritten runtime also passes with code
generation disabled. See `stage4_model_codegen_audit.md` for evidence.

Stage 5 may start from this boundary. It must expose the already registered
models through pybind11 and the stable Python package without adding a
wheel-time compiler or end-user descriptor build path.
