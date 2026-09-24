include_guard(GLOBAL)

set(NR_NATIVE_ROOT "${CMAKE_SOURCE_DIR}/src/native")
set(NR_LEGACY_ROOT "${NR_NATIVE_ROOT}/legacy")
set(NR_LEGACY_SRC_ROOT "${NR_LEGACY_ROOT}/source_file_realtime_v1_async")
set(NR_GPU_ROOT "${NR_NATIVE_ROOT}/gpu")
set(NR_SHARED_ROOT "${NR_NATIVE_ROOT}/shared")
set(NR_SHARED_INCLUDE_ROOT "${NR_SHARED_ROOT}/include")

file(GLOB_RECURSE NR_LEGACY_CPP_SOURCES CONFIGURE_DEPENDS
    "${NR_LEGACY_SRC_ROOT}/*.cpp"
    "${NR_LEGACY_SRC_ROOT}/*.cc"
    "${NR_LEGACY_SRC_ROOT}/*.cxx")

set(NR_SHARED_SOURCES
    "${NR_SHARED_ROOT}/src/learning_rule/LearningRuleCatalog.cpp"
    "${NR_SHARED_ROOT}/src/input_conv/InputConvModelCatalog.cpp"
    "${NR_SHARED_ROOT}/src/neuron_model/NeuronModelCatalog.cpp"
    "${NR_SHARED_ROOT}/src/streaming_build/NbnetReader.cpp")

if(NR_ENABLE_MODEL_CODEGEN)
  list(APPEND NR_SHARED_SOURCES ${NR_CODEGEN_CPP_SOURCES})
else()
  list(APPEND NR_SHARED_SOURCES
      "${NR_SHARED_ROOT}/src/learning_rule/GeneratedLearningRuleCatalogEmpty.cpp"
      "${NR_SHARED_ROOT}/src/neuron_model/GeneratedNeuronCatalogEmpty.cpp")
endif()

set(NR_ZMQ_DRIVER_SOURCES
    "${NR_LEGACY_SRC_ROOT}/communication/src/ZmqSocket.cpp"
    "${NR_LEGACY_SRC_ROOT}/communication/src/ZMQInputOutputSpikeDriver.cpp"
    "${NR_LEGACY_SRC_ROOT}/communication/src/ZMQInputConvFrameDriver.cpp"
    "${NR_LEGACY_SRC_ROOT}/communication/src/ZMQAsyncInputConvFrameDriver.cpp"
    "${NR_LEGACY_SRC_ROOT}/communication/src/ZMQAsyncInputOutputSpikeDriver.cpp")

set(NR_OUTER_DYNAMIC_SOURCES
    "${NR_LEGACY_SRC_ROOT}/ModelFactory/OuterDynamicModelFactory.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/OuterDynamicModel.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/OuterDynamicSpikeCounter.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/OuterDynamicSpikeBuffer.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/PlanarArm2DOFPinocchio.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/ROKAEArm.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/StrictMatlabPlanarArm2DOF.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/StrictMatlabPlanarArm2DOFPinocchioPlant.cpp"
    "${NR_LEGACY_SRC_ROOT}/OuterDynamic/src/StrictMatlabPlanarArm2DOFOuterDynamic.cpp")

set(NR_CORE_CPP_SOURCES ${NR_LEGACY_CPP_SOURCES} ${NR_SHARED_SOURCES})
list(REMOVE_ITEM NR_CORE_CPP_SOURCES
    ${NR_ZMQ_DRIVER_SOURCES}
    ${NR_OUTER_DYNAMIC_SOURCES})

add_library(neuronbridge_native_settings INTERFACE)
target_include_directories(neuronbridge_native_settings INTERFACE
    "${NR_LEGACY_ROOT}"
    "${NR_LEGACY_SRC_ROOT}"
    "${NR_SHARED_INCLUDE_ROOT}"
    "${NR_GPU_ROOT}/include"
    "${NR_LEGACY_SRC_ROOT}/MotionEnergy/inc"
    "${NR_ZEROMQ_INCLUDE_DIR}")
if(NR_ENABLE_CUDA)
  # Some legacy host-side .cpp files call the CUDA runtime directly. CUDA
  # language targets receive this include path automatically, C++ targets do not.
  target_include_directories(neuronbridge_native_settings INTERFACE
      "${CUDAToolkit_INCLUDE_DIRS}")
endif()
if(NR_ENABLE_MODEL_CODEGEN)
  target_include_directories(neuronbridge_native_settings INTERFACE
      "${NR_CODEGEN_INCLUDE_ROOT}")
endif()
target_compile_definitions(neuronbridge_native_settings INTERFACE
    SNN_WITH_ZMQ=1
    SNN_WITH_PINOCCHIO=1
    NPGR_ENABLE_CUDA=$<BOOL:${NR_ENABLE_CUDA}>)
if(WIN32)
  target_compile_definitions(neuronbridge_native_settings INTERFACE
      _CRT_SECURE_NO_WARNINGS
      NOMINMAX
      WIN32=1
      _WIN32_WINNT=0x0A00
      WIN32_LEAN_AND_MEAN)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(neuronbridge_native_settings INTERFACE
      NR_PLATFORM_LINUX=1
      # Pinocchio raises the MPL sequence limits to 30.  Boost's preprocessed
      # MPL headers only cover the default limit, so force on-demand header
      # generation to keep PropertyTree/MultiIndex well-formed on GCC.
      BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS)
endif()
if(NR_ENABLE_MODEL_CODEGEN)
  target_compile_definitions(neuronbridge_native_settings INTERFACE
      NR_ENABLE_MODEL_CODEGEN=1)
endif()
target_link_libraries(neuronbridge_native_settings INTERFACE
    NeuronBridge::Dependencies)
if(MSVC)
  target_compile_options(neuronbridge_native_settings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:/EHsc /Zc:__cplusplus /bigobj /openmp /utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-forward-slash-prefix-opts>
      $<$<COMPILE_LANGUAGE:CUDA>:-forward-unknown-to-host-compiler>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(neuronbridge_native_settings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif()

add_library(neuronbridge_core_cpp STATIC ${NR_CORE_CPP_SOURCES})
add_library(NeuronBridge::CoreCpp ALIAS neuronbridge_core_cpp)
if(NR_ENABLE_MODEL_CODEGEN)
  add_dependencies(neuronbridge_core_cpp neuronbridge_model_codegen)
endif()
target_link_libraries(neuronbridge_core_cpp
    PUBLIC neuronbridge_native_settings)

if(NR_ENABLE_CUDA)
  if(NR_CUDA_RUNTIME_LINKAGE STREQUAL "STATIC" OR
     (NR_CUDA_RUNTIME_LINKAGE STREQUAL "AUTO" AND
      CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    set(NR_CUDART_TARGET CUDA::cudart_static)
  else()
    set(NR_CUDART_TARGET CUDA::cudart)
  endif()

  set(NR_CORE_CUDA_SOURCES
      "${NR_LEGACY_SRC_ROOT}/Openmp/src/OpenmpGPU.cu"
      "${NR_LEGACY_SRC_ROOT}/Network/src/Network.cu"
      "${NR_LEGACY_SRC_ROOT}/Simulation/src/Simulation.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/NeuralStateVector/Neuron_State_Vector_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/EDLUTLikeLIF_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/PoissonRate_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/TimeDrivenIzhikevic_Exponential_Decay_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Exponential_Decay_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Exponential_double_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Exponential_triple_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Voltage_jump_GPU_Interface.cu"
      "${NR_LEGACY_SRC_ROOT}/NeuralModel/src/TimeDrivenGPU/TimeDrivenNeuralModelGPU_Interface.cu")
  if(NR_ENABLE_MODEL_CODEGEN)
    list(APPEND NR_CORE_CUDA_SOURCES ${NR_CODEGEN_CUDA_SOURCES})
  endif()

  add_library(neuronbridge_core_cuda STATIC ${NR_CORE_CUDA_SOURCES})
  add_library(NeuronBridge::CoreCuda ALIAS neuronbridge_core_cuda)
  if(NR_ENABLE_MODEL_CODEGEN)
    add_dependencies(neuronbridge_core_cuda neuronbridge_model_codegen)
  endif()
  target_link_libraries(neuronbridge_core_cuda PUBLIC
      neuronbridge_native_settings
      ${NR_CUDART_TARGET}
      CUDA::cuda_driver)
  set_target_properties(neuronbridge_core_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
      CUDA_PROPAGATE_HOST_FLAGS OFF)
  if(NR_CUDA_ARCHITECTURES)
    set_property(TARGET neuronbridge_core_cuda PROPERTY
        CUDA_ARCHITECTURES "${NR_CUDA_ARCHITECTURES}")
  endif()

  set(NR_DENSE_RUNTIME_SOURCES
      "${NR_GPU_ROOT}/src/bridge/LegacyNetworkBridge.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/DebugMonitorTypes.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/DebugMonitorWriter.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/DenseSubnetworkDebugSource.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/InputConvDebugSource.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/MainNetworkDebugSource.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/OuterDynamicDebugSource.cpp"
      "${NR_GPU_ROOT}/src/debug_monitor/SimulationDebugMonitor.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/DenseSubnetworkEvent.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/DenseSubnetworkFiringTable.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/model/IDenseNeuronModel.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/model/DenseNeuronFieldAccess.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/model/DenseNeuronFieldTableBuilder.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/model/DenseNeuronModelFieldIndex.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/model/DenseNeuronModelFactory.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/learning/DenseLearningRuleFactory.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/DenseUnifiedNeuronRuntime.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/DenseUnifiedNeuronRuntime.cu"
      "${NR_GPU_ROOT}/src/dense_subnetwork/DenseSubnetworkOutputExtractor.cpp"
      "${NR_GPU_ROOT}/src/dense_subnetwork/DenseSubnetworkRuntimeGpu.cpp"
      "${NR_GPU_ROOT}/src/gpu_runtime/GpuPropagationLayout.cpp"
      "${NR_GPU_ROOT}/src/gpu_runtime/GpuPropagationRuntime.cpp"
      "${NR_GPU_ROOT}/src/gpu_runtime/GpuPropagationRuntime.cu"
      "${NR_GPU_ROOT}/src/input_conv/InputConvV1CudaHelpers.cu"
      "${NR_GPU_ROOT}/src/input_conv/Motion.cu"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseBuildShared.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseSubnetworkBuildFinalizer.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseSubnetworkLayoutCompactor.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseInterfaceCurrentNeuronModel.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseInterfaceSpikeNeuronModel.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/SimulationCommonHost.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/SimulationWeightIO.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseSubnetworkModel.cpp"
      "${NR_GPU_ROOT}/src/simulation_dense/DenseSubnetworkUpdateEvent.cpp")

  add_library(neuronbridge_dense_runtime STATIC ${NR_DENSE_RUNTIME_SOURCES})
  add_library(NeuronBridge::DenseRuntime ALIAS neuronbridge_dense_runtime)
  if(NR_ENABLE_MODEL_CODEGEN)
    add_dependencies(neuronbridge_dense_runtime neuronbridge_model_codegen)
  endif()
  target_link_libraries(neuronbridge_dense_runtime PUBLIC
      neuronbridge_native_settings
      neuronbridge_core_cpp
      neuronbridge_core_cuda
      ${NR_CUDART_TARGET}
      CUDA::cuda_driver)
  set_target_properties(neuronbridge_dense_runtime PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF
      CUDA_PROPAGATE_HOST_FLAGS OFF)
  if(NR_CUDA_ARCHITECTURES)
    set_property(TARGET neuronbridge_dense_runtime PROPERTY
        CUDA_ARCHITECTURES "${NR_CUDA_ARCHITECTURES}")
  endif()
endif()

add_library(neuronbridge_core INTERFACE)
add_library(NeuronBridge::Core ALIAS neuronbridge_core)
target_link_libraries(neuronbridge_core INTERFACE neuronbridge_core_cpp)
if(NR_ENABLE_CUDA)
  target_link_libraries(neuronbridge_core INTERFACE neuronbridge_core_cuda)
endif()

add_library(neuronbridge_runtime_support STATIC
    ${NR_ZMQ_DRIVER_SOURCES}
    ${NR_OUTER_DYNAMIC_SOURCES})
add_library(NeuronBridge::RuntimeSupport ALIAS neuronbridge_runtime_support)
target_link_libraries(neuronbridge_runtime_support PUBLIC
    neuronbridge_native_settings)

add_library(neuronbridge_runtime INTERFACE)
add_library(NeuronBridge::Runtime ALIAS neuronbridge_runtime)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NR_ENABLE_CUDA)
  # The legacy host core and the dense CUDA runtime reference each other.
  # ELF static archives are scanned from left to right, so a plain library
  # list can leave symbols unresolved depending on which archive is seen
  # first. RESCAN emits a linker group and resolves the complete cycle.
  target_link_libraries(neuronbridge_runtime INTERFACE
      "$<LINK_GROUP:RESCAN,neuronbridge_runtime_support,neuronbridge_core_cpp,neuronbridge_core_cuda,neuronbridge_dense_runtime>")
else()
  target_link_libraries(neuronbridge_runtime INTERFACE
      neuronbridge_runtime_support
      neuronbridge_core)
  if(NR_ENABLE_CUDA)
    target_link_libraries(neuronbridge_runtime INTERFACE
        neuronbridge_dense_runtime)
  endif()
endif()
