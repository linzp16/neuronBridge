add_executable(neuronbridge_native_lif_smoke native_lif_smoke.cpp)
target_link_libraries(neuronbridge_native_lif_smoke PRIVATE NeuronBridge::Runtime)
add_test(NAME neuronbridge_native_lif_smoke
         COMMAND neuronbridge_native_lif_smoke)

if(NR_ENABLE_CUDA)
  add_executable(neuronbridge_dense_cuda_smoke dense_cuda_smoke.cpp)
  target_link_libraries(neuronbridge_dense_cuda_smoke PRIVATE
      NeuronBridge::Runtime)
  add_test(NAME neuronbridge_dense_cuda_smoke
           COMMAND neuronbridge_dense_cuda_smoke)
endif()

set(_nr_native_tests neuronbridge_native_lif_smoke)
if(NR_ENABLE_CUDA)
  list(APPEND _nr_native_tests neuronbridge_dense_cuda_smoke)
endif()
if(NR_USING_BUNDLED_DEPENDENCIES)
  set_tests_properties(${_nr_native_tests} PROPERTIES
      ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${NR_PINOCCHIO_PREFIX}/bin;PATH=path_list_prepend:${NR_ZEROMQ_PREFIX}/bin")
endif()
if(NR_ENABLE_CUDA)
  set_property(TEST ${_nr_native_tests} APPEND PROPERTY
      ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
endif()
