#include "../source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h"

#include "neuron_model/NeuronModelCatalog.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/InputCurrentNeuronModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/InputSpikeNeuronModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/OuterDynamicInterfaceNeuronModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/TriggerRelayNeuronModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/HandwritingTimeDrivenModel.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/PoissonRate.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenIzhikevic_Exponential_Decay.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_Decay.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_double.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_triple.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Voltage_jump.h"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/EDLUTLikeLIF_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/PoissonRate_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenIzhikevic_Exponential_Decay_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_Decay_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_double_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Exponential_triple_GPU_Interface.cuh"
#include "../source_file_realtime_v1_async/NeuralModel/inc/TimeDrivenGPU/TimeDrivenLIF_Voltage_jump_GPU_Interface.cuh"

#include <iostream>
#include <memory>
#include <stdexcept>
#if NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomGeneratedModels.h"
#include "neuronbridge_codegen/CustomGeneratedLegacyGpuModels.cuh"
#endif

namespace {

typedef NeuronModel* (*LegacyNeuronCreator)(
    const std::map<std::string, boost::any>&,
    int,
    float,
    int);

template <typename ModelT>
NeuronModel* CreateTimedModel(
    const std::map<std::string, boost::any>& parameters,
    int timestepsize,
    float basetimestepsize,
    int) {
    std::unique_ptr<ModelT> model(new ModelT(timestepsize));
    model->SetParameters(parameters, basetimestepsize);
    return model.release();
}

template <typename ModelT>
NeuronModel* CreateGpuTimedModel(
    const std::map<std::string, boost::any>& parameters,
    int timestepsize,
    float basetimestepsize,
    int) {
    ModelT* model = new ModelT(timestepsize);
    model->SetParameters(parameters, basetimestepsize);
    model->timestepdouble = basetimestepsize;
    return model;
}

template <typename ModelT>
NeuronModel* CreateSimpleModel(
    const std::map<std::string, boost::any>&,
    int,
    float,
    int) {
    return new ModelT();
}

template <typename ModelT>
NeuronModel* CreateConfiguredSimpleModel(
    const std::map<std::string, boost::any>& parameters,
    int,
    float,
    int) {
    ModelT* model = new ModelT();
    model->ConfigureFromParameters(parameters);
    return model;
}

std::map<std::string, LegacyNeuronCreator> BuildRegistry() {
    std::map<std::string, LegacyNeuronCreator> registry;
    // Built-in legacy model creators are generated from one central list.
    // This keeps the public factory header small and avoids another long,
    // fragile string branch in the construction path.
#define NPGR_LEGACY_TIMED_MODEL(implementation_name, class_name) \
    registry[implementation_name] = &CreateTimedModel<class_name>;
#define NPGR_LEGACY_GPU_TIMED_MODEL(implementation_name, class_name) \
    registry[implementation_name] = &CreateGpuTimedModel<class_name>;
#define NPGR_LEGACY_SIMPLE_MODEL(implementation_name, class_name) \
    registry[implementation_name] = &CreateSimpleModel<class_name>;
#define NPGR_LEGACY_CONFIGURED_SIMPLE_MODEL(implementation_name, class_name, configure_fn) \
    registry[implementation_name] = &CreateConfiguredSimpleModel<class_name>;
#include "../source_file_realtime_v1_async/ModelFactory/LegacyNeuronModelList.inc"
#undef NPGR_LEGACY_CONFIGURED_SIMPLE_MODEL
#undef NPGR_LEGACY_SIMPLE_MODEL
#undef NPGR_LEGACY_GPU_TIMED_MODEL
#undef NPGR_LEGACY_TIMED_MODEL
#if NR_ENABLE_MODEL_CODEGEN
#define NPGR_CUSTOM_LEGACY_NEURON(implementation_name, class_name) \
    registry[implementation_name] = &CreateTimedModel<class_name>;
#include "neuronbridge_codegen/CustomLegacyNeuronRegistry.inc"
#undef NPGR_CUSTOM_LEGACY_NEURON
#define NPGR_CUSTOM_LEGACY_GPU_NEURON(implementation_name, class_name) \
    registry[implementation_name] = &CreateGpuTimedModel<class_name>;
#include "neuronbridge_codegen/CustomLegacyGpuNeuronRegistry.inc"
#undef NPGR_CUSTOM_LEGACY_GPU_NEURON
#endif
    return registry;
}

const std::map<std::string, LegacyNeuronCreator>& Registry() {
    static const std::map<std::string, LegacyNeuronCreator> registry = BuildRegistry();
    return registry;
}

}  // namespace

NeuronModel* NeuronModelFactory::createNeuronModel(
    const std::string& type,
    std::map<std::string, boost::any> NeuronParameter,
    int timestepsize,
    float basetimestepsize,
    int QueueIndex) {
    const bool requests_gpu_backend = type.find("_GPU") != std::string::npos;
    // The catalog resolves aliases to the concrete backend implementation name.
    // Explicit *_GPU names continue to select the legacy GPU backend.
    const npgr::NeuronBackend backend =
        requests_gpu_backend ? npgr::NeuronBackend::LegacyGpu : npgr::NeuronBackend::LegacyCpu;
    std::string catalog_reason;
    const std::string resolved_type =
        npgr::NeuronModelCatalog::Instance().ResolveImplementationName(
            type,
            backend,
            &catalog_reason);
    const std::string model_type = resolved_type.empty() ? type : resolved_type;

    const std::map<std::string, LegacyNeuronCreator>& registry = Registry();
    std::map<std::string, LegacyNeuronCreator>::const_iterator found =
        registry.find(model_type);
    if (found != registry.end() && found->second != 0) {
        return found->second(NeuronParameter, timestepsize, basetimestepsize, QueueIndex);
    }
    if (!catalog_reason.empty()) {
        std::cout << catalog_reason << std::endl;
    }
    std::cout << "Unknown neuron model type: " << type << std::endl;
    return 0;
}

std::map<std::string, boost::any> NeuronModelFactory::QueryDefaultParameters(
    const std::string& type,
    npgr::NeuronBackend backend,
    int timestepsize,
    float basetimestepsize) {
    return QueryParameters(type, backend, {}, timestepsize, basetimestepsize);
}

std::map<std::string, boost::any> NeuronModelFactory::QueryParameters(
    const std::string& type,
    npgr::NeuronBackend backend,
    const std::map<std::string, boost::any>& parameters,
    int timestepsize,
    float basetimestepsize) {
    if (timestepsize <= 0) {
        throw std::invalid_argument("neuron model parameter query requires a positive timestep size");
    }
    if (!(basetimestepsize > 0.0f)) {
        throw std::invalid_argument("neuron model parameter query requires a positive base timestep");
    }

    std::string reason;
    const std::string implementation =
        npgr::NeuronModelCatalog::Instance().ResolveImplementationName(type, backend, &reason);
    if (implementation.empty()) {
        throw std::invalid_argument(reason.empty() ? "unknown neuron model: " + type : reason);
    }

    const std::map<std::string, LegacyNeuronCreator>& registry = Registry();
    const std::map<std::string, LegacyNeuronCreator>::const_iterator found =
        registry.find(implementation);
    if (found == registry.end() || found->second == 0) {
        throw std::runtime_error(
            "neuron model implementation cannot provide parameters: " + implementation);
    }

    std::unique_ptr<NeuronModel> model(
        found->second(parameters, timestepsize, basetimestepsize, 0));
    if (!model) {
        throw std::runtime_error("failed to construct neuron model: " + implementation);
    }
    return model->getParameters();
}
