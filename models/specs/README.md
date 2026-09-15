# Model specifications

This directory contains maintainer-owned `.nbmodel.json` descriptors. CMake
validates them and generates the matching C++ and CUDA implementation fragments
under the build directory when `NR_ENABLE_MODEL_CODEGEN=ON`.

Descriptors are source inputs, not a wheel-time extension mechanism. Released
wheel users receive precompiled native implementations and do not need Python
code-generation tools, a C++ compiler, or a CUDA toolkit.
