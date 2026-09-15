add_test(NAME neuronbridge_release_tools
    COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            --basetemp "${CMAKE_BINARY_DIR}/pytest-release-$<CONFIG>"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_release_tools.py")
set_tests_properties(neuronbridge_release_tools PROPERTIES
    LABELS "python;release;packaging")
