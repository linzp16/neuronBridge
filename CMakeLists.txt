cmake_minimum_required(VERSION 3.24)

project(NeuronBridge VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(NeuronBridgeOptions)

set(_nr_bundle_msvc_runtime_default OFF)
if(WIN32 AND NR_ENABLE_PYTHON)
  set(_nr_bundle_msvc_runtime_default ON)
endif()
option(NR_BUNDLE_MSVC_RUNTIME
       "Bundle and privatize MSVC/OpenMP runtime DLLs"
       ${_nr_bundle_msvc_runtime_default})
if(MSVC AND NR_ENABLE_PYTHON AND NOT NR_BUNDLE_MSVC_RUNTIME)
  message(FATAL_ERROR
      "NR_BUNDLE_MSVC_RUNTIME=OFF would create an unsafe build-tree Python "
      "package that can bind to incompatible process-global MSVC/OpenMP DLLs. "
      "Reconfigure with -DNR_BUNDLE_MSVC_RUNTIME=ON.")
endif()
if(WIN32)
  include(platforms/WindowsMSVC)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(platforms/LinuxGNU)
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()
include(NeuronBridgeDependencies)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NR_ENABLE_CUDA)
  if(NR_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${NR_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures included in the build" FORCE)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "NR_ENABLE_CUDA=ON requires a working CUDA compiler")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(CUDAToolkit REQUIRED)
endif()

if(NR_ENABLE_MODEL_CODEGEN)
  find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
  if(NOT Python3_VERSION MATCHES "^3\\.12(\\.|$)")
    message(FATAL_ERROR
        "Model code generation requires Python 3.12.x; found ${Python3_VERSION}. "
        "Set Python3_EXECUTABLE to a Python 3.12 interpreter.")
  endif()
  include(NeuronBridgeModelCodegen)
  nr_configure_model_codegen()
endif()

include(NeuronBridgeNativeCore)

if(NR_BUILD_TESTS)
  enable_testing()
endif()

if(NR_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

if(NR_BUILD_TESTS)
  add_subdirectory(tests/dependencies)
  add_subdirectory(tests/native)
  if(NR_ENABLE_MODEL_CODEGEN)
    add_subdirectory(tests/model_codegen)
  endif()
  if(NR_ENABLE_PYTHON)
    add_subdirectory(tests/python)
    add_subdirectory(tests/release)
  endif()
endif()

message(STATUS "NeuronBridge native core migration stage")
message(STATUS "Full version: ${NR_VERSION_FULL}")
message(STATUS "Target platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Core source migration: enabled")
message(STATUS "CUDA backend: ${NR_ENABLE_CUDA}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA runtime linkage: ${NR_CUDA_RUNTIME_LINKAGE}")
message(STATUS "Maintainer model codegen: ${NR_ENABLE_MODEL_CODEGEN}")
message(STATUS "Python 3.12 bindings: ${NR_ENABLE_PYTHON}")
message(STATUS "Bundled dependency migration: compatibility bundle r1")
