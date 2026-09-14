add_executable(neuronbridge_dependency_probe dependency_probe.cpp)
target_compile_features(neuronbridge_dependency_probe PRIVATE cxx_std_17)
target_link_libraries(neuronbridge_dependency_probe
    PRIVATE NeuronBridge::Dependencies)

add_test(NAME neuronbridge_dependency_probe
         COMMAND neuronbridge_dependency_probe)

if(NR_USING_BUNDLED_DEPENDENCIES)
  set_tests_properties(neuronbridge_dependency_probe PROPERTIES
      ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/bin;PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/bin")
endif()
