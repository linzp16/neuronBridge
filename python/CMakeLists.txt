find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(MSVC AND NR_BUNDLE_MSVC_RUNTIME)
  # The Windows wheel post-processes these app-local runtimes to private DLL
  # names so they cannot collide with other native Python packages.
  set(CMAKE_INSTALL_OPENMP_LIBRARIES TRUE)
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
  include(InstallRequiredSystemLibraries)
  if(NOT CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
    message(FATAL_ERROR "MSVC runtime libraries could not be located")
  endif()
endif()

pybind11_add_module(neuronbridge_python_core MODULE
    bindings/neuronbridge_core.cpp)
add_library(NeuronBridge::PythonCore ALIAS neuronbridge_python_core)

target_link_libraries(neuronbridge_python_core PRIVATE
    NeuronBridge::Runtime)
target_include_directories(neuronbridge_python_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings")
set_target_properties(neuronbridge_python_core PROPERTIES
    OUTPUT_NAME "_core"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/neuronbridge"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/neuronbridge")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(neuronbridge_python_core PROPERTIES
      BUILD_RPATH_USE_ORIGIN ON
      INSTALL_RPATH "$ORIGIN"
      INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

# Assemble an importable package in the build tree without writing binaries
# into python/src/neuronbridge.
add_custom_command(TARGET neuronbridge_python_core POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neuronbridge"
            "$<TARGET_FILE_DIR:neuronbridge_python_core>"
    VERBATIM)

if(WIN32 AND NR_USING_BUNDLED_DEPENDENCIES)
  # URDF-backed OuterDynamic models call pinocchio::urdf::buildModel.  On
  # Windows this pulls in the parser DLL and its app-local dependency closure;
  # keeping the list explicit prevents accidentally bundling the entire Conda
  # distribution shipped with the Pinocchio compatibility package.
  set(_nr_pinocchio_runtime_dll_names
      pinocchio_default.dll
      pinocchio_parsers.dll
      boost_serialization.dll
      boost_filesystem.dll
      coal.dll
      urdfdom_model.dll
      console_bridge.dll
      tinyxml2.dll
      sdformat.dll
      gz-math.dll
      gz-utils.dll
      dl.dll
      assimp.dll
      qhull_r.dll
      octomap.dll
      octomath.dll
      zlib.dll)
  set(_nr_pinocchio_runtime_dlls)
  foreach(_nr_runtime_name IN LISTS _nr_pinocchio_runtime_dll_names)
    list(APPEND _nr_pinocchio_runtime_dlls
        "${NR_PINOCCHIO_PREFIX}/bin/${_nr_runtime_name}")
  endforeach()
  add_custom_command(TARGET neuronbridge_python_core POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              ${_nr_pinocchio_runtime_dlls}
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${NR_ZEROMQ_PREFIX}/bin/libzmq-mt-4_3_5.dll"
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      VERBATIM)
endif()
if(WIN32 AND NR_ENABLE_CUDA AND
   NOT NR_CUDA_RUNTIME_LINKAGE STREQUAL "STATIC")
  file(GLOB _nr_cudart_runtime_dlls
      "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
  if(NOT _nr_cudart_runtime_dlls)
    message(FATAL_ERROR "CUDA runtime DLL was not found in ${CUDAToolkit_BIN_DIR}")
  endif()
  add_custom_command(TARGET neuronbridge_python_core POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              ${_nr_cudart_runtime_dlls}
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      COMMAND_EXPAND_LISTS
      VERBATIM)
endif()
if(MSVC AND NR_BUNDLE_MSVC_RUNTIME)
  add_custom_command(TARGET neuronbridge_python_core POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      COMMAND_EXPAND_LISTS
      VERBATIM)

  # Make the build-tree package safe to import before any wheel is assembled.
  # This must run after every native dependency and runtime DLL has been copied.
  add_custom_command(TARGET neuronbridge_python_core POST_BUILD
      COMMAND "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/scripts/private_runtime_directory.py"
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
              --report
              "$<TARGET_FILE_DIR:neuronbridge_python_core>/private-runtime-report.json"
      COMMENT "Privatizing NeuronBridge MSVC/OpenMP runtime dependencies"
      VERBATIM)
endif()

install(TARGETS neuronbridge_python_core
    LIBRARY DESTINATION neuronbridge
    RUNTIME DESTINATION neuronbridge)

if(WIN32 AND NR_USING_BUNDLED_DEPENDENCIES)
  install(FILES
      ${_nr_pinocchio_runtime_dlls}
      "${NR_ZEROMQ_PREFIX}/bin/libzmq-mt-4_3_5.dll"
      DESTINATION neuronbridge)
endif()
if(WIN32 AND NR_ENABLE_CUDA AND
   NOT NR_CUDA_RUNTIME_LINKAGE STREQUAL "STATIC")
  install(FILES ${_nr_cudart_runtime_dlls} DESTINATION neuronbridge)
endif()
if(MSVC AND NR_BUNDLE_MSVC_RUNTIME)
  install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION neuronbridge)

  # install(TARGETS) copies the already-rewritten extension, while the runtime
  # install rules above start from Microsoft's public DLL names.  Re-run the
  # same directory transaction in the staging prefix so the raw wheel always
  # contains a complete private-name dependency closure before delvewheel
  # performs its general dependency repair.
  install(CODE
      "execute_process(
          COMMAND \"${Python3_EXECUTABLE}\"
                  \"${PROJECT_SOURCE_DIR}/scripts/private_runtime_directory.py\"
                  \"\${CMAKE_INSTALL_PREFIX}/neuronbridge\"
          RESULT_VARIABLE _nr_private_runtime_install_result)
       if(NOT _nr_private_runtime_install_result EQUAL 0)
         message(FATAL_ERROR
             \"Failed to privatize MSVC/OpenMP runtimes in the install staging directory\")
       endif()")
endif()

if(NR_BUILD_TESTS)
  add_test(NAME neuronbridge_python_api
      COMMAND "${Python3_EXECUTABLE}" -m pytest -q
              --basetemp "${CMAKE_BINARY_DIR}/pytest-api-$<CONFIG>"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_api.py")
  add_test(NAME neuronbridge_python_catalog
      COMMAND "${Python3_EXECUTABLE}" -m pytest -q
              --basetemp "${CMAKE_BINARY_DIR}/pytest-catalog-$<CONFIG>"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_catalog.py")
  add_test(NAME neuronbridge_python_native_smoke
      COMMAND "${Python3_EXECUTABLE}" -m pytest -q
              --basetemp "${CMAKE_BINARY_DIR}/pytest-native-smoke-$<CONFIG>"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native_smoke.py")
  add_test(NAME neuronbridge_python_nbnet
      COMMAND "${Python3_EXECUTABLE}" -m pytest -q
              --basetemp "${CMAKE_BINARY_DIR}/pytest-nbnet-$<CONFIG>"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nbnet.py")

  set(_nr_python_tests
      neuronbridge_python_api
      neuronbridge_python_catalog
      neuronbridge_python_native_smoke
      neuronbridge_python_nbnet)
  set_tests_properties(${_nr_python_tests} PROPERTIES
      ENVIRONMENT_MODIFICATION
      "PYTHONPATH=path_list_prepend:${CMAKE_BINARY_DIR}/python/$<CONFIG>")
  if(WIN32 AND NR_USING_BUNDLED_DEPENDENCIES)
    set_property(TEST ${_nr_python_tests} APPEND PROPERTY
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/bin;PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/bin")
  endif()
  if(WIN32 AND NR_ENABLE_CUDA)
    set_property(TEST ${_nr_python_tests} APPEND PROPERTY
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NR_USING_BUNDLED_DEPENDENCIES)
    set_property(TEST ${_nr_python_tests} APPEND PROPERTY
        ENVIRONMENT_MODIFICATION
        "LD_LIBRARY_PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/lib;LD_LIBRARY_PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/lib")
  endif()
endif()
