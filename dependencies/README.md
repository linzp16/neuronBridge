# Bundled dependencies

NeuronBridge will support `AUTO`, `BUNDLED`, and `SYSTEM` dependency modes.
The initial supported bundle target is `windows-x64-msvc`.

During the dependency migration stage, the validated Pinocchio and ZeroMQ
prefixes from the historical workspace will be audited, minimized, packaged,
hashed, and recorded in `manifest.json`.

The versioned archive belongs under:

```text
dependencies/bundles/windows-x64-msvc/
```

The archive is tracked through Git LFS. Its expanded contents belong under
`dependencies/vendor/` and are never committed as individual files.

The Stage 2 repository contains compatibility bundle r1. Its expanded vendor
directory remains local and ignored by Git.

## Maintainer packaging

```powershell
.\scripts\package_dependencies.ps1 `
  -SourceVendorRoot <historical-dependency-vendor> `
  -CondaMetaDir <pinocchio-conda-meta> `
  -Revision 1
```

Revision 1 is intentionally a compatibility bundle and excludes PDB files.
See `docs/migration/stage2_dependency_audit.md` for the reduction policy.

## User bootstrap

```powershell
git lfs pull --include="dependencies/bundles/**"
.\scripts\bootstrap_dependencies.ps1
.\scripts\verify_dependencies.ps1
```

Bootstrap verifies the archive hash, extracts into a temporary directory,
checks required files, and atomically installs the bundle under
`dependencies/vendor/windows-x64-msvc/r1`.
