add_test(
  NAME neuronbridge_model_codegen_python
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s tools/model_codegen/tests -p test_*.py -v)
set_tests_properties(neuronbridge_model_codegen_python PROPERTIES
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

function(nr_add_model_codegen_runtime_test target_name source_file)
  add_executable(${target_name} "${source_file}")
  target_link_libraries(${target_name} PRIVATE NeuronBridge::Runtime)
  add_dependencies(${target_name} neuronbridge_model_codegen)
  add_test(NAME ${target_name} COMMAND ${target_name} ${ARGN})
  set_property(GLOBAL APPEND PROPERTY NR_MODEL_CODEGEN_RUNTIME_TESTS ${target_name})
endfunction()

nr_add_model_codegen_runtime_test(
  neuronbridge_model_codegen_lif_baseline lif_baseline.cpp)
nr_add_model_codegen_runtime_test(
  neuronbridge_model_codegen_learning_rule_catalog learning_rule_catalog.cpp)
nr_add_model_codegen_runtime_test(
  neuronbridge_neuron_model_catalog_parameters neuron_model_catalog_parameters.cpp)
nr_add_model_codegen_runtime_test(
  neuronbridge_model_codegen_rstdp_baseline rstdp_baseline.cpp
  "${CMAKE_BINARY_DIR}/model_codegen/rstdp_trace")

if(NR_ENABLE_CUDA)
  nr_add_model_codegen_runtime_test(
    neuronbridge_model_codegen_rstdp_gpu_trajectory rstdp_gpu_trajectory.cpp
    "${CMAKE_BINARY_DIR}/model_codegen/rstdp_gpu_trace")
  nr_add_model_codegen_runtime_test(
    neuronbridge_model_codegen_dense_lif_gpu_baseline dense_lif_gpu_baseline.cpp)
  nr_add_model_codegen_runtime_test(
    neuronbridge_model_codegen_legacy_gpu_lif_baseline legacy_gpu_lif_baseline.cpp)
  nr_add_model_codegen_runtime_test(
    neuronbridge_model_codegen_pair_stdp_gpu_trajectory pair_stdp_gpu_trajectory.cpp)

  add_executable(neuronbridge_model_codegen_math_functions math_functions_compile.cu)
  target_link_libraries(neuronbridge_model_codegen_math_functions PRIVATE CUDA::cudart)
  set_target_properties(neuronbridge_model_codegen_math_functions PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME neuronbridge_model_codegen_math_functions
           COMMAND neuronbridge_model_codegen_math_functions)
  set_property(GLOBAL APPEND PROPERTY NR_MODEL_CODEGEN_RUNTIME_TESTS
      neuronbridge_model_codegen_math_functions)
endif()

get_property(_nr_codegen_runtime_tests GLOBAL PROPERTY NR_MODEL_CODEGEN_RUNTIME_TESTS)
if(NR_USING_BUNDLED_DEPENDENCIES)
  set_tests_properties(${_nr_codegen_runtime_tests} PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/bin;PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/bin")
endif()
if(NR_ENABLE_CUDA)
  set_property(TEST ${_nr_codegen_runtime_tests} APPEND PROPERTY
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
