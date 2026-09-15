#include "dense_subnetwork/DenseSubnetworkEvent.h"
#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"
#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#include "dense_subnetwork/learning/IDenseLearningRuleModel.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "simulation_dense/DenseSubnetworkBuildFinalizer.h"
#include "source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "source_file_realtime_v1_async/LearningRule/inc/SynapseState.h"
#include "source_file_realtime_v1_async/ModelFactory/LearningRuleModelFactory.h"
#include "source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Events { bool pre = false; bool post = false; };

struct CpuFixture {
    std::unique_ptr<LearningRule> rule;
    Neuron target;
    Interconnections plastic;

    explicit CpuFixture(const char* name) {
        rule.reset(LearningRuleModelFactory::createLearningRuleModel({name, {}}));
        Require(rule != nullptr, std::string("missing CPU rule: ") + name);
        rule->LearningRuleID = 0;
        rule->InitState(1, 1, 1.0f);
        plastic.weight = 8.0f;
        plastic.maximum_weight = 20.0f;
        plastic.TargetNeuron = &target;
        plastic.LearningRuleIndex_withPost = 0;
        target.NumberOfRule = 1;
        target.PostSynapticLearning_Number = new int[1]{1};
        target.PostSynapticLearning = new Interconnections**[1]{};
        target.PostSynapticLearning[0] = new Interconnections*[1]{&plastic};
        target.IndexOfInputLearningIndex[0] = new int*[1]{};
        target.IndexOfInputLearningIndex[0][0] = new int[1]{0};
    }
};

std::vector<float> RunCpu(const char* rule_name, const std::vector<Events>& events,
                          int* state_count) {
    CpuFixture fixture(rule_name);
    if (state_count != nullptr) *state_count = fixture.rule->State->NumberOfState;
    Simulation context({}, {}, {}, 32, 1.0f, 1, EVENT_QUEUE_HEAP, 0);
    std::vector<float> weights{fixture.plastic.weight};
    for (int tick = 0; tick < static_cast<int>(events.size()); ++tick) {
        if (events[tick].pre)
            fixture.rule->ApplyPreSynaticSpike(&fixture.plastic, tick, &context);
        if (events[tick].post)
            fixture.rule->ApplyPostSynaticSpike(&fixture.target, tick, &context);
        weights.push_back(fixture.plastic.weight);
    }
    return weights;
}

npgr::sim_support::DenseSubnetworkBuildSpec BuildGpuSpec(std::string* reason) {
    npgr::sim_support::DenseSubnetworkBuildSpec spec;
    spec.name = "generated_pair_stdp_gpu_trajectory";
    spec.layout.stats.neuron_count = 2;
    spec.config.dt_ms = 1.0f;
    LearningRuleDescription rule;
    rule.RuleName = "CustomPairStdpV1";
    Require(npgr::DenseLearningRuleFactory::Instance().BuildLegacyRuleTable(
                {rule}, &spec.learning_rule_table, &spec.learning_rule_fields, reason), *reason);

    spec.layout.synapses.post_neuron = {1};
    spec.layout.synapses.post_model = {0};
    spec.layout.synapses.weight = {8.0f};
    spec.layout.synapses.max_weight = {20.0f};
    spec.layout.synapses.type = {0};
    spec.layout.synapses.effect_channel = {
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance)};
    spec.layout.synapses.effect_scale = {1.0f};
    spec.layout.synapses.plastic_rule_id = {0};
    spec.layout.synapses.plastic_state_index = {0};
    spec.layout.synapses.trigger_rule_id = {-1};
    spec.synapse_source_local_ids = {0};
    spec.synapse_delay_slots = {1};

    npgr::DenseNeuronModelSpec lif;
    lif.factory_model_id =
        npgr::DenseNeuronModelFactory::kDenseCustomLifConductanceV1ModelId;
    lif.legacy_model_name = "CustomLifConductanceV1";
    lif.model_id = 0;
    lif.range = {0, 2};
    lif.per_neuron_params["V_th"] = {-59.8f, -59.8f};
    lif.per_neuron_params["V_rest"] = {-60.0f, -60.0f};
    lif.per_neuron_params["V_reset"] = {-60.0f, -60.0f};
    lif.per_neuron_params["t_ref"] = {0.0f, 0.0f};
    spec.neuron_models = {lif};
    spec.internal_model_ids = {0};
    spec.neuron_model_id_by_neuron = {0, 0};
    spec.input_source_main_ids = {-1};
    spec.input_interface_main_ids = {-1};
    spec.input_target_local_ids = {0};
    spec.input_uses_current = {0};
    spec.input_pending_channels = {
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance)};
    spec.input_scales = {1.0f};
    spec.input_representative_source_main_ids = {-1};
    spec.output_neuron_mask = {0, 1};
    Require(npgr::sim_support::FinalizeDenseSubnetworkBuildSpec(&spec, reason), *reason);
    return spec;
}

struct GpuResult {
    std::vector<float> weights;
    std::vector<Events> events;
};

GpuResult RunGpu() {
    std::string reason;
    const auto spec = BuildGpuSpec(&reason);
    npgr::RuntimeConfig config;
    config.dt_ms = 1.0f;
    config.steps_to_keep = 32;
    npgr::DenseSubnetworkRuntimeGpu runtime;
    Require(runtime.Initialize(spec, config, &reason), reason);
    Require(runtime.IsGpuBackendReady(), "CUDA backend unavailable");
    GpuResult result;
    float weight = 0.0f;
    Require(runtime.GetSynapseWeight(0, &weight, &reason), reason);
    result.weights.push_back(weight);
    result.events.resize(16);
    std::vector<std::vector<bool>> fired(16, std::vector<bool>(2, false));
    for (int tick = 0; tick < 16; ++tick) {
        if (tick == 0 || tick == 7)
            Require(runtime.QueueInterfaceSpike({0, tick, 1.0f, false}, &reason), reason);
        npgr::DenseSubnetworkEventResult event_result;
        npgr::DenseSubnetworkEvent event(tick);
        Require(event.Execute(&runtime, &event_result, &reason), reason);
        Require(runtime.GetSynapseWeight(0, &weight, &reason), reason);
        result.weights.push_back(weight);
        const auto* entry = runtime.firing_table().HistoryAtTime(tick);
        if (entry != nullptr) {
            for (int id : entry->firing_ids)
                if (id >= 0 && id < 2) fired[tick][id] = true;
        }
    }
    for (int tick = 0; tick < 16; ++tick) {
        result.events[tick].pre = tick >= 1 && fired[tick - 1][0];
        result.events[tick].post = fired[tick][1];
    }
    return result;
}

}  // namespace

int main() {
    try {
        const auto gpu = RunGpu();
        int generated_states = 0;
        int builtin_states = 0;
        const auto generated = RunCpu("CustomPairStdpV1", gpu.events, &generated_states);
        const auto builtin = RunCpu("STDP", gpu.events, &builtin_states);
        Require(generated_states == 2 && builtin_states == 2,
                "pair-STDP must use two states and differ from R-STDP layout");
        Require(generated.size() == builtin.size() && generated.size() == gpu.weights.size(),
                "pair-STDP trajectory length mismatch");
        float cpu_error = 0.0f;
        float gpu_error = 0.0f;
        int changed = 0;
        for (std::size_t index = 0; index < generated.size(); ++index) {
            cpu_error = std::max(cpu_error, std::abs(generated[index] - builtin[index]));
            gpu_error = std::max(gpu_error, std::abs(generated[index] - gpu.weights[index]));
            if (index > 0 && generated[index] != generated[index - 1]) ++changed;
        }
        if (cpu_error > 1.0e-6f || gpu_error > 5.0e-6f) {
            std::cerr << "pair_stdp_diagnostic cpu_error=" << cpu_error
                      << " gpu_error=" << gpu_error << '\n';
            for (std::size_t index = 0; index < generated.size(); ++index) {
                const Events event = index == 0 ? Events{} : gpu.events[index - 1];
                std::cerr << "sample=" << index << " pre=" << event.pre
                          << " post=" << event.post << " builtin=" << builtin[index]
                          << " cpu=" << generated[index] << " gpu=" << gpu.weights[index] << '\n';
            }
        }
        Require(cpu_error <= 1.0e-6f, "generated pair-STDP differs from builtin STDP");
        Require(gpu_error <= 5.0e-6f, "generated pair-STDP CPU/GPU mismatch");
        Require(changed > 0, "pair-STDP trajectory did not change weight");
        const auto* dense = npgr::DenseLearningRuleFactory::Instance().FindByLegacyName(
            "CustomPairStdpV1");
        Require(dense != nullptr &&
                    (dense->Flags() & npgr::kDenseLearningUsesTrigger) == 0,
                "pair-STDP dense trigger contract mismatch");
        std::cout << "generated_pair_stdp=PASS samples=" << generated.size()
                  << " changed_steps=" << changed
                  << " cpu_builtin_max_error=" << cpu_error
                  << " cpu_gpu_max_error=" << gpu_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generated_pair_stdp=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
