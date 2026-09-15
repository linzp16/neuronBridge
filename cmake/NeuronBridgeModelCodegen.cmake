include_guard(GLOBAL)

function(nr_configure_model_codegen)
  set(_nr_output_root "${CMAKE_BINARY_DIR}/generated/model_codegen")
  set(_nr_include_root "${_nr_output_root}/include")
  set(_nr_src_root "${_nr_output_root}/src")
  set(_nr_spec_root "${CMAKE_SOURCE_DIR}/models/specs")

  file(GLOB_RECURSE _nr_codegen_tools CONFIGURE_DEPENDS
      "${CMAKE_SOURCE_DIR}/tools/model_codegen/*.py")
  file(GLOB_RECURSE _nr_codegen_specs CONFIGURE_DEPENDS
      "${_nr_spec_root}/*.nbmodel.json")

  set(_nr_cpp_sources
      "${_nr_src_root}/GeneratedNeuronCatalogEntries.cpp"
      "${_nr_src_root}/GeneratedNeuronModels.cpp"
      "${_nr_src_root}/GeneratedLearningRuleModels.cpp"
      "${_nr_src_root}/GeneratedLearningRuleCatalogEntries.cpp")
  set(_nr_cuda_sources
      "${_nr_src_root}/GeneratedLegacyGpuNeuronModels.cu")
  set(_nr_headers
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedModels.h"
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedLegacyGpuModels.cuh"
      "${_nr_include_root}/neuronbridge_codegen/CustomLegacyNeuronRegistry.inc"
      "${_nr_include_root}/neuronbridge_codegen/CustomLegacyGpuNeuronRegistry.inc"
      "${_nr_include_root}/neuronbridge_codegen/CustomNeuronCatalog.inc"
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedDenseNeuronModels.h"
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedDenseNeuronDeviceUpdates.cuh"
      "${_nr_include_root}/neuronbridge_codegen/CustomDenseNeuronModelList.inc"
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedLearningRules.h"
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedDenseLearningRules.h"
      "${_nr_include_root}/neuronbridge_codegen/CustomGeneratedDenseLearningDeviceUpdates.cuh"
      "${_nr_include_root}/neuronbridge_codegen/CustomDenseLearningModelList.inc"
      "${_nr_include_root}/neuronbridge_codegen/CustomLegacyLearningRuleRegistry.inc")
  set(_nr_manifest "${_nr_output_root}/manifest.json")
  set(_nr_outputs
      ${_nr_cpp_sources}
      ${_nr_cuda_sources}
      ${_nr_headers}
      "${_nr_manifest}")

  add_custom_command(
    OUTPUT ${_nr_outputs}
    COMMAND "${Python3_EXECUTABLE}" -m tools.model_codegen.cli
      --spec-root "${_nr_spec_root}"
      --output-root "${_nr_output_root}"
      --manifest "${_nr_manifest}"
    DEPENDS ${_nr_codegen_tools} ${_nr_codegen_specs}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Generating maintainer-owned NeuronBridge models"
    VERBATIM)

  add_custom_target(neuronbridge_model_codegen DEPENDS ${_nr_outputs})
  add_custom_target(nr_model_codegen DEPENDS neuronbridge_model_codegen)

  set(NR_CODEGEN_OUTPUT_ROOT "${_nr_output_root}" PARENT_SCOPE)
  set(NR_CODEGEN_INCLUDE_ROOT "${_nr_include_root}" PARENT_SCOPE)
  set(NR_CODEGEN_MANIFEST "${_nr_manifest}" PARENT_SCOPE)
  set(NR_CODEGEN_CPP_SOURCES ${_nr_cpp_sources} PARENT_SCOPE)
  set(NR_CODEGEN_CUDA_SOURCES ${_nr_cuda_sources} PARENT_SCOPE)
endfunction()
