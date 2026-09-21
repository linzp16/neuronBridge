# Bundled dependencies

NeuronBridge will support `AUTO`, `BUNDLED`, and `SYSTEM` dependency modes.
The validated bundle target is currently `windows-x64-msvc`. The Linux port
adds the reserved platform key `linux-x86_64-gcc-cuda12`; its archive must not
be added to `manifest.json` until the payload, license inventory, and SHA256
have been produced and verified on the Linux CUDA build host.

During the dependency migration stage, the validated Pinocchio and ZeroMQ
prefixes from the historical workspace will be audited, minimized, packaged,
hashed, and recorded in `manifest.json`.

The versioned archive belongs under:

```text
dependencies/bundles/windows-x64-msvc/
dependencies/bundles/linux-x86_64-gcc-cuda12/
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

## Linux CUDA bundle contract

The Linux bundle must install under:

```text
dependencies/vendor/linux-x86_64-gcc-cuda12/r1/
  pinocchio/
    include/
    lib/
    share/
  zeromq/
    include/zmq.h
    include/zmq.hpp
    lib/
```

It must use one GCC/libstdc++ ABI for Pinocchio, Boost, coal/hpp-fcl,
urdfdom, sdformat/gz, ZeroMQ, and cppzmq. CUDA driver libraries such as
`libcuda.so.1` are host-owned and must never be included in the archive or
wheel. The Linux wheel build may temporarily use `NR_DEPENDENCY_MODE=SYSTEM`
for bring-up, but release artifacts must use the hashed bundle recorded in
`manifest.json`.
