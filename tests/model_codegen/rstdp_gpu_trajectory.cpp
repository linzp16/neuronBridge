#include "dense_subnetwork/DenseSubnetworkEvent.h"
#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"
#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"
#include "simulation_dense/DenseSubnetworkBuildFinalizer.h"
#include "source_file_realtime_v1_async/LearningRule/inc/LearningRule.h"
#include "source_file_realtime_v1_async/ModelFactory/LearningRuleModelFactory.h"
#include "source_file_realtime_v1_async/Neuron/inc/Neuron.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct CpuRuleFixture {
    std::unique_ptr<LearningRule> rule;
    Neuron target;
    Interconnections plastic;
    Interconnections trigger;

    explicit CpuRuleFixture(const char* name) {
        rule.reset(LearningRuleModelFactory::createLearningRuleModel({name, {}}));
        Require(rule != nullptr, std::string("missing CPU rule: ") + name);
        rule->LearningRuleID = 0;
        rule->InitState(2, 1, 1.0f);

        plastic.weight = 12.0f;
        plastic.maximum_weight = 20.0f;
        plastic.TargetNeuron = &target;
        plastic.LearningRuleIndex_withPostAndTrigger = 0;
        plastic.TriggerLearning = false;

        trigger.weight = 1.0f;
        trigger.maximum_weight = 1.0f;
        trigger.TargetNeuron = &target;
        trigger.LearningRuleIndex_withPostAndTrigger = 1;
        trigger.TriggerLearning = true;
        trigger.type = 0;

        target.NumberOfRule = 1;
        target.TriggerAndPostSynapticLearning_Number = new int[1]{2};
        target.TriggerAndPostSynapticLearning = new Interconnections**[1]{};
        target.TriggerAndPostSynapticLearning[0] = new Interconnections*[2]{&plastic, &trigger};
        target.IndexOfInputLearningIndex[2] = new int*[1]{};
        target.IndexOfInputLearningIndex[2][0] = new int[2]{0, 1};
    }
};

struct RuleEvents {
    bool pre = false;
    bool post = false;
    bool trigger = false;
};

std::vector<float> RunCpuTrace(const char* rule_name,
                               const std::vector<RuleEvents>& events) {
    CpuRuleFixture fixture(rule_name);
    Simulation context({}, {}, {}, 16, 1.0f, 1, EVENT_QUEUE_HEAP, 0);
    std::vector<float> trace{fixture.plastic.weight};
    for (int tick = 0; tick < static_cast<int>(events.size()); ++tick) {
        // Dense propagation applies pre then trigger updates before the neuron
        // kernel produces the current step's post event.
        if (events[static_cast<std::size_t>(tick)].pre)
            fixture.rule->ApplyPreSynaticSpike(&fixture.plastic, tick, &context);
        if (events[static_cast<std::size_t>(tick)].trigger)
            fixture.rule->ApplyPreSynaticSpike(&fixture.trigger, tick, &context);
        if (events[static_cast<std::size_t>(tick)].post)
            fixture.rule->ApplyPostSynaticSpike(&fixture.target, tick, &context);
        trace.push_back(fixture.plastic.weight);
    }
    return trace;
}

npgr::sim_support::DenseSubnetworkBuildSpec BuildGpuSpec(const std::string& rule_name,
                                                          std::string* reason) {
    npgr::sim_support::DenseSubnetworkBuildSpec spec;
    spec.name = "generated_rstdp_gpu_trajectory";
    spec.layout.stats.neuron_count = 3;
    spec.layout.stats.model_count = 2;
    spec.config.dt_ms = 1.0f;

    LearningRuleDescription rule;
    rule.RuleName = rule_name;
    std::list<LearningRuleDescription> rules{rule};
    if (!npgr::DenseLearningRuleFactory::Instance().BuildLegacyRuleTable(
            rules, &spec.learning_rule_table, &spec.learning_rule_fields, reason)) {
        return spec;
    }

    spec.layout.synapses.post_neuron = {1, 1};
    spec.layout.synapses.post_model = {0, 0};
    spec.layout.synapses.weight = {12.0f, 1.0f};
    spec.layout.synapses.max_weight = {20.0f, 1.0f};
    spec.layout.synapses.type = {0, 0};
    spec.layout.synapses.effect_channel = {
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance),
        static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance)};
    spec.layout.synapses.effect_scale = {1.0f, 1.0f};
    spec.layout.synapses.plastic_rule_id = {0, -1};
    spec.layout.synapses.plastic_state_index = {0, -1};
    spec.layout.synapses.trigger_rule_id = {-1, 0};
    spec.synapse_source_local_ids = {0, 2};
    spec.synapse_delay_slots = {1, 3};

    npgr::DenseNeuronModelSpec lif;
    lif.factory_model_id = npgr::DenseNeuronModelFactory::kLifExponentialDoubleModelId;
    lif.legacy_model_name = "TimeDrivenLIF_Exponential_double";
    lif.model_id = 0;
    lif.range = npgr::DenseNeuronRange{0, 2};
    lif.per_neuron_params["V_th"] = std::vector<float>{-64.8f, -64.8f};
    lif.per_neuron_params["t_ref"] = std::vector<float>{0.0f, 0.0f};

    npgr::DenseNeuronModelSpec relay;
    relay.factory_model_id = npgr::DenseNeuronModelFactory::kTriggerRelayModelId;
    relay.legacy_model_name = "TriggerRelayNeuronModel";
    relay.model_id = 1;
    relay.range = npgr::DenseNeuronRange{2, 1};
    relay.per_neuron_params["threshold"] = std::vector<float>{0.5f};
    spec.neuron_models = {lif, relay};
    spec.internal_model_ids = {0, 1};
    spec.neuron_model_id_by_neuron = {0, 0, 1};

    for (int local_id : {0, 2}) {
        spec.input_source_main_ids.push_back(-1);
        spec.input_interface_main_ids.push_back(-1);
        spec.input_target_local_ids.push_back(local_id);
        spec.input_uses_current.push_back(0);
        spec.input_pending_channels.push_back(
            static_cast<std::uint8_t>(npgr::PendingChannel::ExcitatoryConductance));
        spec.input_scales.push_back(1.0f);
        spec.input_representative_source_main_ids.push_back(-1);
    }
    spec.output_neuron_mask = {0, 0, 1};
    npgr::sim_support::FinalizeDenseSubnetworkBuildSpec(&spec, reason);
    return spec;
}

struct GpuTrace {
    std::vector<float> weights;
    std::vector<RuleEvents> events;
};

GpuTrace RunGpuTrace(const char* rule_name) {
    std::string reason;
    npgr::sim_support::DenseSubnetworkBuildSpec spec = BuildGpuSpec(rule_name, &reason);
    Require(reason.empty() && spec.IsValid(&reason), std::string("GPU spec: ") + reason);
    npgr::RuntimeConfig config;
    config.steps_to_keep = 16;
    config.dt_ms = 1.0f;
    npgr::DenseSubnetworkRuntimeGpu runtime;
    Require(runtime.Initialize(spec, config, &reason), std::string("GPU init: ") + reason);
    Require(runtime.IsGpuBackendReady(), "CUDA device backend is not ready");

    float weight = 0.0f;
    Require(runtime.GetSynapseWeight(0, &weight, &reason), reason);
    GpuTrace trace;
    trace.weights.push_back(weight);
    trace.events.resize(10);
    std::vector<std::vector<bool> > fired(10, std::vector<bool>(3, false));
    for (int tick = 0; tick <= 9; ++tick) {
        if (tick == 0) {
            Require(runtime.QueueInterfaceSpike({0, 0, 12.0f, false}, &reason), reason);
        }
        if (tick == 1 || tick == 5) {
            Require(runtime.QueueInterfaceSpike({1, tick, 1.0f, false}, &reason), reason);
        }
        npgr::DenseSubnetworkEventResult result;
        npgr::DenseSubnetworkEvent event(tick);
        Require(event.Execute(&runtime, &result, &reason), reason);
        Require(runtime.GetSynapseWeight(0, &weight, &reason), reason);
        trace.weights.push_back(weight);
        const npgr::DenseFiringHistoryEntry* entry = runtime.firing_table().HistoryAtTime(tick);
        if (entry != nullptr) {
            for (int neuron_id : entry->firing_ids) {
                if (neuron_id >= 0 && neuron_id < 3)
                    fired[static_cast<std::size_t>(tick)][static_cast<std::size_t>(neuron_id)] = true;
            }
        }
    }
    for (int tick = 0; tick <= 9; ++tick) {
        RuleEvents& event = trace.events[static_cast<std::size_t>(tick)];
        event.pre = tick >= 1 && fired[static_cast<std::size_t>(tick - 1)][0];
        event.trigger = tick >= 3 && fired[static_cast<std::size_t>(tick - 3)][2];
        event.post = fired[static_cast<std::size_t>(tick)][1];
    }
    return trace;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        constexpr float kTolerance = 5.0e-6f;
        const std::filesystem::path output_dir = argc > 1 ? argv[1] : "rstdp_gpu_trace";
        std::filesystem::create_directories(output_dir);
        std::ofstream csv(output_dir / "cpu_gpu_weight_trace.csv");
        csv << std::setprecision(std::numeric_limits<float>::max_digits10);
        csv << "rule,sample_index,time_step,pre_event,trigger_event,post_event,cpu_weight,gpu_weight,abs_error\n";

        float max_error = 0.0f;
        int comparisons = 0;
        for (const char* rule : {"CustomRStdpV1", "CustomRStdpPersistentV1"}) {
            const GpuTrace gpu_result = RunGpuTrace(rule);
            const std::vector<float> cpu = RunCpuTrace(rule, gpu_result.events);
            const std::vector<float>& gpu = gpu_result.weights;
            Require(cpu.size() == gpu.size(), "CPU/GPU trajectory length mismatch");
            for (std::size_t index = 0; index < cpu.size(); ++index) {
                const float error = std::abs(cpu[index] - gpu[index]);
                max_error = std::max(max_error, error);
                ++comparisons;
                const RuleEvents event = index == 0
                    ? RuleEvents{} : gpu_result.events[index - 1];
                csv << rule << ',' << index << ',' << static_cast<int>(index) - 1 << ','
                    << event.pre << ',' << event.trigger << ',' << event.post << ','
                    << cpu[index] << ',' << gpu[index] << ',' << error << '\n';
            }
        }
        csv.close();
        std::ofstream json(output_dir / "cpu_gpu_error_report.json");
        json << "{\n"
             << "  \"result\": \"" << (max_error <= kTolerance ? "PASS" : "FAIL") << "\",\n"
             << "  \"rules\": 2,\n"
             << "  \"samples_per_backend\": " << comparisons << ",\n"
             << "  \"max_abs_weight_error\": " << std::setprecision(9) << max_error << ",\n"
             << "  \"tolerance\": 5e-6,\n"
             << "  \"cuda_device_ready\": true\n"
             << "}\n";
        Require(max_error <= kTolerance, "CPU/GPU generated-rule weight mismatch");
        std::cout << "generated_rstdp_cpu_gpu=PASS rules=2 samples=" << comparisons
                  << " max_abs_weight_error=" << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generated_rstdp_cpu_gpu=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
