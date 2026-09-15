#include "dense_subnetwork/DenseSubnetworkEvent.h"
#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "simulation_dense/DenseBuildShared.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

npgr::sim_support::DenseSubnetworkBuildSpec BuildDescription() {
    npgr::sim_support::DenseSubnetworkBuildSpec description;
    description.name = "stage3_dense_cuda_smoke";
    description.layout.stats.neuron_count = 3;
    description.layout.stats.delay_slot_count = 2;
    description.layout.stats.model_count = 1;
    description.layout.stats.synapse_count = 2;

    description.layout.synapses.post_neuron = {1, 2};
    description.layout.synapses.post_model = {0, 0};
    description.layout.synapses.weight = {12.0f, 8.0f};
    description.layout.synapses.type = {0, 0};
    description.layout.synapses.effect_channel = {
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance),
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance)};
    description.layout.synapses.effect_scale = {1.0f, 1.0f};
    description.layout.synapses.plastic_rule_id = {-1, -1};
    description.layout.synapses.plastic_model_id = {-1, -1};
    description.layout.synapses.plastic_flags = {0, 0};
    description.layout.synapses.plastic_state_index = {-1, -1};
    description.layout.synapses.trigger_rule_id = {-1, -1};
    description.layout.synapses.trigger_model_id = {-1, -1};
    description.layout.synapses.trigger_flags = {0, 0};

    description.layout.pre_delay_slices.neuron_count = 3;
    description.layout.pre_delay_slices.delay_slot_count = 2;
    description.layout.pre_delay_slices.start.assign(6, 0);
    description.layout.pre_delay_slices.count.assign(6, 0);
    description.layout.pre_delay_slices.synapse_ids = {0, 1};
    description.layout.pre_delay_slices.start[0] = 0;
    description.layout.pre_delay_slices.count[0] = 1;
    description.layout.pre_delay_slices.start[3] = 1;
    description.layout.pre_delay_slices.count[3] = 1;

    description.internal_model_ids = {0};
    npgr::DenseNeuronModelSpec model;
    model.factory_model_id =
        npgr::DenseNeuronModelFactory::kLifExponentialDoubleModelId;
    model.legacy_model_name = "TimeDrivenLIF_Exponential_double";
    model.model_id = 0;
    model.range = npgr::DenseNeuronRange{0, 3};
    model.per_neuron_params["V_th"] = std::vector<float>(3, -64.8f);
    description.neuron_models.push_back(model);
    description.neuron_model_id_by_neuron = {0, 0, 0};

    description.input_source_main_ids = {-1};
    description.input_interface_main_ids = {-1};
    description.input_target_local_ids = {0};
    description.input_uses_current = {0};
    description.input_pending_channels = {
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance)};
    description.input_scales = {1.0f};
    description.input_representative_source_main_ids = {-1};
    description.output_source_local_ids = {1};
    description.output_target_main_ids = {101};
    description.output_delays = {0};
    description.output_weights = {1.0f};
    description.output_types = {0};
    description.output_neuron_mask = {0, 1, 0};
    return description;
}

}  // namespace

int main() {
    std::string reason;
    auto description = BuildDescription();
    if (!description.IsValid(&reason)) {
        std::cerr << "result=FAIL reason=" << reason << '\n';
        return EXIT_FAILURE;
    }

    npgr::RuntimeConfig config;
    config.steps_to_keep = 4;
    npgr::DenseSubnetworkRuntimeGpu runtime;
    if (!runtime.Initialize(description, config, &reason)) {
        std::cerr << "result=FAIL reason=" << reason << '\n';
        return EXIT_FAILURE;
    }
    if (!runtime.QueueInterfaceSpike({0, 0, 12.0f, false}, &reason)) {
        std::cerr << "result=FAIL reason=" << reason << '\n';
        return EXIT_FAILURE;
    }

    int output_spikes = 0;
    for (int step = 0; step < 2; ++step) {
        npgr::DenseSubnetworkEventResult result;
        npgr::DenseSubnetworkEvent event(step);
        if (!event.Execute(&runtime, &result, &reason)) {
            std::cerr << "result=FAIL step=" << step << " reason=" << reason << '\n';
            return EXIT_FAILURE;
        }
        output_spikes += static_cast<int>(result.output_spikes.size());
    }

    const auto* history0 = runtime.firing_table().HistoryAtTime(0);
    if (history0 == nullptr || history0->firing_ids != std::vector<int>{0}) {
        std::cerr << "result=FAIL reason=t0 firing baseline mismatch\n";
        return EXIT_FAILURE;
    }
    std::cout << "backend=DenseGpu steps=2 firing_events="
              << history0->firing_ids.size()
              << " output_spikes=" << output_spikes << "\nresult=PASS\n";
    return EXIT_SUCCESS;
}
