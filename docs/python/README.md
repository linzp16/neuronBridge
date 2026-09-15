# Python design records

`python/README.md` and `docs/migration/stage5_python_bridge_audit.md` describe
the current source and build contract. The files in this directory preserve
the detailed design history developed before repository migration:

- `bridge_review.md`: original binding surface review.
- `implementation_memory.md`: long-running implementation decisions and
  historical validation notes.
- `visualization_api_plan.md`: plotting API design and phased delivery record.
- `packaging_plan.md`: input to Stage 7; it is not yet a release guarantee.

Historical absolute paths and phase labels in those records are evidence, not
current build instructions. The repository CMake files and migration audits
are authoritative when they differ.
