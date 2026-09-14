# Release process

A NeuronBridge release is built from a clean annotated Git tag. Release
artifacts are uploaded to GitHub Releases and are not committed to the normal
Git history.

Planned Windows artifacts:

- Pure source archive.
- Source archive with the Windows x64 dependency bundle.
- Standalone dependency bundle.
- Python 3.12 Windows wheel.
- Self-contained Python 3.12 offline bundle.
- SHA-256 checksums and validation summary.

The release workflow must verify version consistency, Git cleanliness,
dependency hashes, native builds, Python tests, CUDA baselines, wheel imports,
and third-party notices before creating release staging.

Public release is blocked until a project-level `LICENSE` is approved.
