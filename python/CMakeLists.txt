find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

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

# Assemble an importable package in the build tree without writing binaries
# into python/src/neuronbridge.
add_custom_command(TARGET neuronbridge_python_core POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/src/neuronbridge"
            "$<TARGET_FILE_DIR:neuronbridge_python_core>"
    VERBATIM)

if(NR_USING_BUNDLED_DEPENDENCIES)
  add_custom_command(TARGET neuronbridge_python_core POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${NR_PINOCCHIO_PREFIX}/bin/pinocchio_default.dll"
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${NR_PINOCCHIO_PREFIX}/bin/boost_serialization.dll"
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${NR_ZEROMQ_PREFIX}/bin/libzmq-mt-4_3_5.dll"
              "$<TARGET_FILE_DIR:neuronbridge_python_core>"
      VERBATIM)
endif()
if(NR_ENABLE_CUDA)
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

install(TARGETS neuronbridge_python_core
    LIBRARY DESTINATION neuronbridge
    RUNTIME DESTINATION neuronbridge)

if(NR_USING_BUNDLED_DEPENDENCIES)
  install(FILES
      "${NR_PINOCCHIO_PREFIX}/bin/pinocchio_default.dll"
      "${NR_PINOCCHIO_PREFIX}/bin/boost_serialization.dll"
      "${NR_ZEROMQ_PREFIX}/bin/libzmq-mt-4_3_5.dll"
      DESTINATION neuronbridge)
endif()
if(NR_ENABLE_CUDA)
  install(FILES ${_nr_cudart_runtime_dlls} DESTINATION neuronbridge)
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

  set(_nr_python_tests
      neuronbridge_python_api
      neuronbridge_python_catalog
      neuronbridge_python_native_smoke)
  set_tests_properties(${_nr_python_tests} PROPERTIES
      ENVIRONMENT_MODIFICATION
      "PYTHONPATH=path_list_prepend:${CMAKE_BINARY_DIR}/python/$<CONFIG>")
  if(NR_USING_BUNDLED_DEPENDENCIES)
    set_property(TEST ${_nr_python_tests} APPEND PROPERTY
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/bin;PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/bin")
  endif()
  if(NR_ENABLE_CUDA)
    set_property(TEST ${_nr_python_tests} APPEND PROPERTY
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
  endif()
endif()
