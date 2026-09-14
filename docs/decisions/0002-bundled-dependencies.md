# ADR 0002: Versioned bundled dependencies

## Status

Accepted.

## Decision

NeuronBridge supports a Windows bundled dependency distribution so users do
not need to install Pinocchio, Boost, Eigen, or ZeroMQ separately.

The compressed dependency archive is versioned through Git LFS. Its manifest,
hashes, and license notices are committed normally. Expanded vendor files are
local build state and are ignored by Git.

System dependencies remain an optional maintainer mode. CUDA Toolkit and the
NVIDIA driver are not part of the source dependency bundle.
