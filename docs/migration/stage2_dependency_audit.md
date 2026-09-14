# Stage 2 dependency audit

## Source baseline

The historical dependency tree contains 26,627 files and approximately 0.979
GiB. It consists of a broad Pinocchio Conda `Library` prefix and a separately
packaged ZeroMQ prefix.

The largest file groups before packaging are approximately:

| Type | Count | Size |
|---|---:|---:|
| DLL | 250 | 564.4 MiB |
| C/C++ headers | 18,201 | 176.0 MiB |
| PDB | 4 | 88.9 MiB |
| LIB | 165 | 85.9 MiB |
| EXE | 63 | 49.2 MiB |

## Stage 2 decision

Revision 1 is a compatibility bundle. It preserves the validated Pinocchio and
ZeroMQ prefixes but excludes PDB debug symbols. The broad prefix is retained
because upstream exported CMake targets may reference transitive libraries and
tools that cannot be proven unnecessary before the native core is compiled
from the new repository.

The source Conda environment contains Python 3.14-related files even though
NeuronBridge targets Python 3.12. Those files are recorded as a reduction
candidate and must not be treated as the Python runtime used by NeuronBridge.

## Validation contract

- Archive size and SHA-256 must match `dependencies/manifest.json`.
- The ZIP must be readable and contain the declared root directory.
- Pinocchio, Boost, Eigen, and ZeroMQ headers must be present.
- Pinocchio and ZeroMQ link libraries and runtime DLLs must pass minimum-size
  checks so Git LFS pointers or truncated files cannot be accepted.
- Installation occurs only under `dependencies/vendor`.
- Installation never modifies the system `PATH` or system directories.
- Known Conda build-machine paths in Assimp and Boost CMake exports are repaired
  to use their computed package-relative prefixes.
- Tests prepend bundle runtime directories only to the child test process;
  Conda Boost runtime metadata is not used as an authoritative DLL collector.

## Deferred reduction

After Stage 3 can compile and run the native core against the extracted bundle,
runtime DLL tracing and imported-target analysis will produce a smaller
revision. Python 3.14 bindings, PDB files, unused tools, documentation, tests,
and unrelated numerical packages are candidates, but no item is removed solely
by filename.

## Licensing status

Primary versions and declared licenses have been identified. Complete license
text collection and redistribution review are still required before a public
GitHub release.

## Revision 1 result

- Archive: `neuronbridge-deps-windows-x64-msvc-r1.zip`
- Archive size: 292,178,558 bytes (278.64 MiB)
- Expanded payload: 26,624 files and 913.87 MiB including install metadata
- Excluded debug symbols: 4 PDB files
- SHA-256: `a9565ff3c8c57ffe1321860e1025bdc30c15bd976380a62ac8d4a9dcc07a18c4`
- Conda inventory: 84 package records and 26 declared license identifiers
- Forbidden historical/build-machine CMake paths after repair: 0

The dependency probe configured with MSVC 19.41, compiled and linked against
the extracted prefix, constructed an empty Pinocchio model, and reported
ZeroMQ 4.3.5 at runtime. Initial bootstrap, forced replacement, and repeated
idempotent bootstrap all passed.
