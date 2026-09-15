if(NOT NR_ENABLE_PYTHON OR NOT NR_BUILD_EXAMPLES)
  return()
endif()

add_test(NAME neuronbridge_python_migrated_examples
    COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            --basetemp "${CMAKE_BINARY_DIR}/pytest-examples-$<CONFIG>"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_migrated_examples.py")
set_tests_properties(neuronbridge_python_migrated_examples PROPERTIES
    LABELS "python;examples;integration"
    ENVIRONMENT_MODIFICATION
    "PYTHONPATH=path_list_prepend:${CMAKE_BINARY_DIR}/python/$<CONFIG>")
if(NR_USING_BUNDLED_DEPENDENCIES)
  set_property(TEST neuronbridge_python_migrated_examples APPEND PROPERTY
      ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/bin;PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/bin")
endif()
if(NR_ENABLE_CUDA)
  set_property(TEST neuronbridge_python_migrated_examples APPEND PROPERTY
      ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
