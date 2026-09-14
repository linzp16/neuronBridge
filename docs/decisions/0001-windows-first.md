# ADR 0001: Windows-first initial release

## Status

Accepted.

## Decision

The initial NeuronBridge repository and release process target Windows x64,
Visual Studio 2022, CUDA 12.x, and Python 3.12. PowerShell is the official
orchestration environment for this phase.

Platform-specific logic must remain in scripts and CMake platform modules so a
future Linux implementation does not require changes to simulation semantics.
