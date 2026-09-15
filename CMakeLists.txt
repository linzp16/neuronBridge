cmake_minimum_required(VERSION 3.24)

project(NeuronBridge VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(NeuronBridgeOptions)
include(platforms/WindowsMSVC)
include(NeuronBridgeDependencies)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "NR_ENABLE_CUDA=ON requires a working CUDA compiler. "
        "Run CMake from a Visual Studio 2022 developer environment.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  find_package(CUDAToolkit REQUIRED)
endif()

include(NeuronBridgeNativeCore)

if(NR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/dependencies)
  add_subdirectory(tests/native)
endif()

message(STATUS "NeuronBridge native core migration stage")
message(STATUS "Full version: ${NR_VERSION_FULL}")
message(STATUS "Core source migration: enabled")
message(STATUS "CUDA backend: ${NR_ENABLE_CUDA}")
message(STATUS "Bundled dependency migration: compatibility bundle r1")
