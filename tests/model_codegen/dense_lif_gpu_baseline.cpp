#include "dense_subnetwork/DenseSubnetworkEvent.h"
#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "dense_subnetwork/model/IDenseNeuronModel.h"
#include "neuron_model/NeuronModelCatalog.h"
#include "simulation_dense/DenseSubnetworkBuildFinalizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Sample {
    float v = 0.0f;
    float gexc = 0.0f;
    float ginh = 0.0f;
    unsigned char fired = 0;
};

npgr::sim_support::DenseSubnetworkBuildSpec BuildSpec(
    int factory_model_id, const char* model_name) {
    npgr::sim_support::DenseSubnetworkBuildSpec spec;
    spec.name = std::string(model_name) + "_gpu_baseline";
    spec.layout.stats.neuron_count = 1;
    spec.config.dt_ms = 0.1f;

    npgr::DenseNeuronModelSpec model;
    model.factory_model_id = factory_model_id;
    model.legacy_model_name = model_name;
    model.model_id = 0;
    model.range = {0, 1};
    model.per_neuron_params["tau"] = {20.0f};
    model.per_neuron_params["gexc_tau"] = {5.0f};
    model.per_neuron_params["ginh_tau"] = {10.0f};
    model.per_neuron_params["V_rest"] = {-60.0f};
    model.per_neuron_params["V_reset"] = {-60.0f};
    model.per_neuron_params["V_th"] = {-50.0f};
    model.per_neuron_params["R"] = {1.0f};
    model.per_neuron_params["Eexc"] = {0.0f};
    model.per_neuron_params["Einh"] = {-80.0f};
    model.per_neuron_params["t_ref"] = {2.0f};
    spec.neuron_models = {model};
    spec.internal_model_ids = {0};
    spec.neuron_model_id_by_neuron = {0};

    for (int channel = 0; channel < 3; ++channel) {
        spec.input_source_main_ids.push_back(-1);
        spec.input_interface_main_ids.push_back(-1);
        spec.input_target_local_ids.push_back(0);
        spec.input_uses_current.push_back(channel == 2 ? 1 : 0);
        spec.input_pending_channels.push_back(static_cast<std::uint8_t>(channel));
        spec.input_scales.push_back(1.0f);
        spec.input_representative_source_main_ids.push_back(-1);
    }
    spec.input_connection_source_main_ids = {-1};
    spec.input_connection_slot_indices = {2};
    spec.input_connection_delays = {0};
    spec.input_connection_weights = {1.0f};
    spec.input_connection_max_weights = {1.0f};
    spec.input_connection_types = {3};
    spec.input_connection_pending_channels = {
        static_cast<std::uint8_t>(npgr::PendingChannel::Current)};
    spec.input_connection_effect_scales = {1.0f};
    spec.output_neuron_mask = {1};
    std::string reason;
    Require(npgr::sim_support::FinalizeDenseSubnetworkBuildSpec(&spec, &reason), reason);
    return spec;
}

std::vector<Sample> Run(int factory_model_id, const char* model_name) {
    std::string reason;
    npgr::RuntimeConfig config;
    config.dt_ms = 0.1f;
    config.steps_to_keep = 64;
    npgr::DenseSubnetworkRuntimeGpu runtime;
    const auto spec = BuildSpec(factory_model_id, model_name);
    Require(runtime.Initialize(spec, config, &reason), reason);
    Require(runtime.IsGpuBackendReady(), "CUDA backend unavailable");
    std::vector<Sample> result;
    for (int tick = 0; tick < 160; ++tick) {
        if (tick % 17 == 0)
            Require(runtime.QueueInterfaceSpike({0, tick, 4.0f, false}, &reason), reason);
        if (tick % 29 == 11)
            Require(runtime.QueueInterfaceSpike({1, tick, 0.8f, true}, &reason), reason);
        Require(runtime.SetInterfaceCurrentConnection(0, tick % 13 == 0 ? 2.0f : 0.0f,
                                                      tick, &reason), reason);
        npgr::DenseSubnetworkEventResult event_result;
        npgr::DenseSubnetworkEvent event(tick);
        Require(event.Execute(&runtime, &event_result, &reason), reason);
        const auto snapshot = runtime.BuildDebugSnapshot(&reason);
        Require(reason.empty() && snapshot.model_debug_states.size() == 1,
                std::string("debug snapshot: ") + reason);
        const auto& state = snapshot.model_debug_states.front();
        Sample sample;
        sample.v = state.float_state_vectors.at("v").at(0);
        sample.gexc = state.float_state_vectors.at("gexc").at(0);
        sample.ginh = state.float_state_vectors.at("ginh").at(0);
        sample.fired = state.byte_state_vectors.at("fired").at(0);
        result.push_back(sample);
    }
    return result;
}

}  // namespace

int main() {
    try {
        auto& catalog = npgr::NeuronModelCatalog::Instance();
        auto& factory = npgr::DenseNeuronModelFactory::Instance();
        Require(catalog.IsSupported("CustomLifConductanceV1", npgr::NeuronBackend::DenseGpu),
                "generated LIF is not marked DenseGpu supported");
        const auto* generated_model = factory.FindByLegacyName("CustomLifConductanceV1");
        Require(generated_model != nullptr, "generated dense LIF factory registration missing");
        Require(generated_model->FactoryModelId() ==
                    npgr::DenseNeuronModelFactory::kDenseCustomLifConductanceV1ModelId,
                "generated dense LIF model id mismatch");

        const auto reference = Run(
            npgr::DenseNeuronModelFactory::kLifExponentialDoubleModelId,
            "TimeDrivenLIF_Exponential_double");
        const auto generated = Run(
            npgr::DenseNeuronModelFactory::kDenseCustomLifConductanceV1ModelId,
            "CustomLifConductanceV1");
        Require(reference.size() == generated.size(), "dense LIF trace length mismatch");
        float max_error = 0.0f;
        int spikes = 0;
        for (std::size_t index = 0; index < reference.size(); ++index) {
            max_error = std::max(max_error, std::abs(reference[index].v - generated[index].v));
            max_error = std::max(max_error, std::abs(reference[index].gexc - generated[index].gexc));
            max_error = std::max(max_error, std::abs(reference[index].ginh - generated[index].ginh));
            Require(reference[index].fired == generated[index].fired, "dense LIF spike mismatch");
            spikes += generated[index].fired != 0;
        }
        Require(max_error <= 1.0e-5f, "dense LIF state mismatch");
        Require(spikes > 0, "dense LIF baseline did not produce spikes");
        std::cout << "generated_dense_lif=PASS samples=" << generated.size()
                  << " spikes=" << spikes << " max_abs_error=" << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generated_dense_lif=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
