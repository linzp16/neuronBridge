include_guard(GLOBAL)

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(NR_DEPENDENCY_PLATFORM_KEY "windows-x64-msvc" CACHE STRING
    "Bundled dependency platform key")
set(NR_DEPENDENCY_ROOT "" CACHE PATH
    "Explicit expanded dependency root")

if(NR_DEPENDENCY_MODE STREQUAL "AUTO" OR NR_DEPENDENCY_MODE STREQUAL "BUNDLED")
  if(NOT NR_DEPENDENCY_ROOT)
    set(_nr_manifest_path
        "${CMAKE_SOURCE_DIR}/dependencies/manifest.json")
    if(EXISTS "${_nr_manifest_path}")
      file(READ "${_nr_manifest_path}" _nr_manifest_json)
      string(JSON _nr_bundle_count LENGTH "${_nr_manifest_json}" bundles)
      if(_nr_bundle_count GREATER 0)
        math(EXPR _nr_bundle_last "${_nr_bundle_count} - 1")
        foreach(_nr_index RANGE 0 ${_nr_bundle_last})
          string(JSON _nr_key GET "${_nr_manifest_json}"
                 bundles ${_nr_index} platform_key)
          if(_nr_key STREQUAL NR_DEPENDENCY_PLATFORM_KEY)
            string(JSON _nr_install_subdir GET "${_nr_manifest_json}"
                   bundles ${_nr_index} install_subdir)
            set(NR_DEPENDENCY_ROOT
                "${CMAKE_SOURCE_DIR}/dependencies/vendor/${_nr_install_subdir}")
            break()
          endif()
        endforeach()
      endif()
    endif()
  endif()
endif()

if(NR_DEPENDENCY_MODE STREQUAL "BUNDLED" AND
   NOT IS_DIRECTORY "${NR_DEPENDENCY_ROOT}")
  message(FATAL_ERROR
      "Bundled dependencies are not installed. Run "
      ".\\scripts\\bootstrap_dependencies.ps1 before configuring.")
endif()

set(_nr_use_bundled OFF)
if(IS_DIRECTORY "${NR_DEPENDENCY_ROOT}")
  set(_nr_use_bundled ON)
elseif(NR_DEPENDENCY_MODE STREQUAL "AUTO")
  message(STATUS "Bundled dependencies are unavailable; trying system packages")
endif()

if(_nr_use_bundled)
  set(NR_PINOCCHIO_PREFIX
      "${NR_DEPENDENCY_ROOT}/pinocchio-cpp/Library" CACHE PATH "" FORCE)
  set(NR_ZEROMQ_PREFIX
      "${NR_DEPENDENCY_ROOT}/zeromq" CACHE PATH "" FORCE)

  list(PREPEND CMAKE_PREFIX_PATH
      "${NR_PINOCCHIO_PREFIX}"
      "${NR_PINOCCHIO_PREFIX}/lib/cmake"
      "${NR_PINOCCHIO_PREFIX}/share"
      "${NR_ZEROMQ_PREFIX}")
  set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

  find_package(pinocchio CONFIG REQUIRED
      PATHS "${NR_PINOCCHIO_PREFIX}/lib/cmake/pinocchio"
      NO_DEFAULT_PATH)

  find_library(NR_ZEROMQ_LIBRARY
      NAMES libzmq-mt-4_3_5 libzmq
      PATHS "${NR_ZEROMQ_PREFIX}/lib"
      NO_DEFAULT_PATH REQUIRED)
  find_path(NR_ZEROMQ_INCLUDE_DIR
      NAMES zmq.h
      PATHS "${NR_ZEROMQ_PREFIX}/include"
      NO_DEFAULT_PATH REQUIRED)
else()
  find_package(pinocchio CONFIG REQUIRED)
  find_library(NR_ZEROMQ_LIBRARY NAMES libzmq-mt-4_3_5 libzmq REQUIRED)
  find_path(NR_ZEROMQ_INCLUDE_DIR NAMES zmq.h REQUIRED)
endif()

if(NOT TARGET NeuronBridgeZeroMQ)
  add_library(NeuronBridgeZeroMQ SHARED IMPORTED)
  set_target_properties(NeuronBridgeZeroMQ PROPERTIES
      IMPORTED_IMPLIB "${NR_ZEROMQ_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NR_ZEROMQ_INCLUDE_DIR}")
  if(_nr_use_bundled)
    set_target_properties(NeuronBridgeZeroMQ PROPERTIES
        IMPORTED_LOCATION
        "${NR_ZEROMQ_PREFIX}/bin/libzmq-mt-4_3_5.dll")
  endif()
  add_library(NeuronBridge::ZeroMQ ALIAS NeuronBridgeZeroMQ)
endif()

add_library(neuronbridge_dependencies INTERFACE)
add_library(NeuronBridge::Dependencies ALIAS neuronbridge_dependencies)
target_link_libraries(neuronbridge_dependencies INTERFACE
    OpenMP::OpenMP_CXX
    pinocchio::pinocchio
    NeuronBridge::ZeroMQ)
target_compile_definitions(neuronbridge_dependencies INTERFACE
    _WIN32_WINNT=0x0A00
    WINVER=0x0A00)

set(NR_USING_BUNDLED_DEPENDENCIES "${_nr_use_bundled}" CACHE INTERNAL "")
message(STATUS "NeuronBridge dependency mode: ${NR_DEPENDENCY_MODE}")
message(STATUS "NeuronBridge dependency root: ${NR_DEPENDENCY_ROOT}")
