set(NR_VERSION_FULL "0.1.0a0" CACHE STRING "Full NeuronBridge release version")

option(NR_ENABLE_PYTHON "Build the Python 3.12 bindings" ON)
option(NR_ENABLE_CUDA "Build the CUDA backend" ON)
option(NR_BUILD_TESTS "Build validation targets" ON)
option(NR_BUILD_EXAMPLES "Build C++ examples" ON)

set(NR_DEPENDENCY_MODE "AUTO" CACHE STRING
    "Dependency mode: AUTO, BUNDLED, or SYSTEM")
set_property(CACHE NR_DEPENDENCY_MODE PROPERTY STRINGS AUTO BUNDLED SYSTEM)

string(TOUPPER "${NR_DEPENDENCY_MODE}" NR_DEPENDENCY_MODE)
if(NOT NR_DEPENDENCY_MODE MATCHES "^(AUTO|BUNDLED|SYSTEM)$")
  message(FATAL_ERROR
      "NR_DEPENDENCY_MODE must be AUTO, BUNDLED, or SYSTEM; got: "
      "${NR_DEPENDENCY_MODE}")
endif()
