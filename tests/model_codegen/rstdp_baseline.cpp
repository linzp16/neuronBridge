#include "source_file_realtime_v1_async/ModelFactory/LearningRuleModelFactory.h"
#include "source_file_realtime_v1_async/LearningRule/inc/Rule/R_STDP.h"
#include "source_file_realtime_v1_async/LearningRule/inc/SynapseState.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "learning_rule/LearningRuleCatalog.h"
#include "neuronbridge_codegen/CustomGeneratedLearningRules.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct Totals {
    long long comparisons = 0, rows = 0, events = 0, lower = 0, upper = 0;
    float max_error = 0;
    void Compare(float a, float b) {
        Require(std::isfinite(a) && std::isfinite(b), "nonfinite trajectory");
        ++comparisons;
        max_error = std::max(max_error, std::abs(a - b));
        Require(a == b, "trajectory mismatch");
    }
};

// Own the same lists as Network, including a nonzero rule ID, skipped entry,
// trigger entries, and a separate target that must not receive target-0 events.
struct Fixture {
    std::unique_ptr<LearningRule> rule;
    std::array<Neuron, 2> targets;
    std::array<Interconnections, 6> connections;
    Fixture(const char* name, const std::map<std::string, boost::any>& parameters, float dt) {
        rule.reset(LearningRuleModelFactory::createLearningRuleModel({name, parameters}));
        Require(rule != nullptr, "factory missing rule");
        rule->LearningRuleID = 1;
        rule->InitState(6, 2, dt);
        for (int i = 0; i < 6; ++i) {
            auto& c = connections[i];
            c.weight = 0.25f + 0.1f * i; c.maximum_weight = 1.0f;
            c.TargetNeuron = &targets[i == 2 ? 1 : 0];
            c.LearningRuleIndex_withPostAndTrigger = i;
            c.TriggerLearning = (i == 3 || i == 4);
            c.type = i == 4 ? 1 : 0;
        }
        for (int target = 0; target < 2; ++target) {
            auto& n = targets[target];
            n.NumberOfRule = 2;
            n.TriggerAndPostSynapticLearning_Number = new int[2]{0, target == 0 ? 5 : 1};
            n.TriggerAndPostSynapticLearning = new Interconnections**[2]{};
            n.IndexOfInputLearningIndex[2] = new int*[2]{};
            int count = n.TriggerAndPostSynapticLearning_Number[1];
            n.TriggerAndPostSynapticLearning[1] = new Interconnections*[count];
            n.IndexOfInputLearningIndex[2][1] = new int[count];
            std::array<int, 5> ids = target == 0 ? std::array<int, 5>{0, 1, 3, 4, 5} : std::array<int, 5>{2};
            for (int j = 0; j < count; ++j) {
                n.TriggerAndPostSynapticLearning[1][j] = &connections[ids[j]];
                n.IndexOfInputLearningIndex[2][1][j] = ids[j] == 5 ? -1 : ids[j];
            }
        }
    }
    void Event(int kind, int tick, Simulation& simulation) {
        if (kind == 6 || kind == 7) rule->ApplyPostSynaticSpike(&targets[kind - 6], tick, &simulation);
        else rule->ApplyPreSynaticSpike(&connections[kind], tick, &simulation);
    }
};

void Snapshot(std::ofstream& out, int scenario, int ordinal, int tick, int event,
              Fixture& a, Fixture& b, Totals& total) {
    for (int c = 0; c < 6; ++c) {
        auto* x = a.rule->State; auto* y = b.rule->State;
        total.Compare(a.connections[c].weight, b.connections[c].weight);
        if (c >= 3) {
            Require(a.connections[c].weight == 0.25f + 0.1f * c, "trigger/skipped connection weight changed");
            Require(x->LastUpdate[c] == 0, "trigger/skipped state was advanced");
        }
        Require(x->LastUpdate[c] == y->LastUpdate[c], "last update mismatch");
        out << scenario << ',' << ordinal << ',' << tick << ',' << event << ',' << c
            << ',' << a.connections[c].weight << ',' << b.connections[c].weight;
        for (int slot = 0; slot < 3; ++slot) {
            float av = x->GetStateValue(c, slot), bv = y->GetStateValue(c, slot);
            total.Compare(av, bv);
            if (c >= 3) Require(av == 0, "trigger/skipped trace changed");
            out << ',' << av << ',' << bv;
        }
        out << ',' << x->LastUpdate[c] << ',' << y->LastUpdate[c] << '\n';
        ++total.rows;
        total.lower += a.connections[c].weight == 0;
        total.upper += a.connections[c].weight == a.connections[c].maximum_weight;
    }
}

std::vector<float> NetworkTrace(const char* name, bool clear, int delay,
                               std::vector<std::pair<int, int>>& spikes, std::vector<std::string>& fields) {
    NeuronLayerDescription input; input.ModelName = "InputSpikeNeuronModel"; input.numberofneuron = 3;
    NeuronLayerDescription output; output.ModelName = "TimeDrivenLIF_Exponential_double";
    output.numberofneuron = 1; output.isOutput = true;
    output.NeuronParameter["t_ref"] = 3;
    ConnectionDescription c;
    c.SourceNeuron = {0, 1, 2}; c.TargetNeuron = {3, 3, 3};
    c.Type = {0, 0, 1}; c.Weight = {3.0f, 0.0f, 0.0f}; c.MaxWeight = {6.0f, 1.0f, 1.0f};
    c.Delay = {delay, delay, delay}; c.SynapseRule = {0, -1, -1}; c.TriggerRule = {-1, 0, 0};
    LearningRuleDescription rule{name, {{"ClearEligibilityAfterTrigger", clear}}};
    Simulation simulation({input, output}, {c}, {rule}, 500, 0.1f, 1, EVENT_QUEUE_HEAP, 0);
    simulation.InitSimulation();
    std::vector<int> times, neurons;
    for (int tick = 1; tick < 400; tick += 7) {
        times.push_back(tick); neurons.push_back(0);
        if (tick % 3 == 0) { times.push_back(tick); neurons.push_back(1); }
        if (tick % 5 == 0) { times.push_back(tick); neurons.push_back(2); }
    }
    simulation.AddExternalSpikeActivity(times, neurons);
    std::vector<float> result;
    for (int tick = 0; tick < 500; ++tick) {
        simulation.RunSimulationStep(1);
        auto* state = simulation.network->LearningRules[0]->State;
        for (int i = 0; i < simulation.network->intersNum; ++i) {
            result.push_back(simulation.network->inters[i].weight);
            if (tick == 0) fields.push_back("connection_" + std::to_string(i) + "_weight");
        }
        for (int i = 0; i < state->NumberOfConnections; ++i) {
            for (int slot = 0; slot < 3; ++slot) {
                result.push_back(state->GetStateValue(i, slot));
                if (tick == 0) fields.push_back("state_" + std::to_string(i) + "_" + std::array<std::string, 3>{"pre", "post", "eligibility"}[slot]);
            }
            result.push_back(static_cast<float>(state->LastUpdate[i]));
            if (tick == 0) fields.push_back("state_" + std::to_string(i) + "_last_update");
        }
    }
    for (const auto& spike : simulation.output_spike_driver->OutputBuffer) spikes.emplace_back(spike.time, spike.neuron);
    std::sort(spikes.begin(), spikes.end());
    return result;
}

double Measure(const char* name, Simulation& context) {
    Fixture fixture(name, {}, context.basetimesteps);
    for (int tick = 0; tick < 100; ++tick) {
        fixture.Event(0, tick, context); fixture.Event(6, tick, context); fixture.Event(3, tick, context);
    }
    const auto start = std::chrono::steady_clock::now();
    for (int tick = 100; tick < 100100; ++tick) {
        fixture.Event(0, tick, context); fixture.Event(6, tick, context); fixture.Event(3, tick, context);
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int main(int argc, char** argv) {
    try {
        std::filesystem::path directory = argc > 1 ? argv[1] : "rstdp_trace";
        std::filesystem::create_directories(directory);
        std::ofstream out(directory / "event_weight_trace.csv");
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << std::setprecision(std::numeric_limits<float>::max_digits10);
        out << "scenario,event_index,tick,event_kind,connection,reference_weight,generated_weight,reference_pre,generated_pre,reference_post,generated_post,reference_eligibility,generated_eligibility,reference_last_update,generated_last_update\n";
        for (const char* name : {"CustomRStdpV1", "CustomRStdpPersistentV1"}) {
            const auto* entry = npgr::LearningRuleCatalog::Instance().Resolve(name);
            Require(entry && entry->dense_gpu_supported && !entry->public_user_model, "catalog capability mismatch");
        }
        {
            std::unique_ptr<LearningRule> first(LearningRuleModelFactory::createLearningRuleModel({"CustomRStdpV1", {}}));
            std::unique_ptr<LearningRule> second(LearningRuleModelFactory::createLearningRuleModel({"CustomRStdpPersistentV1", {}}));
            Require(dynamic_cast<CustomRStdpV1*>(first.get()) != nullptr, "first generated factory type mismatch");
            Require(dynamic_cast<CustomRStdpPersistentV1*>(second.get()) != nullptr, "second generated factory type mismatch");
            Require(boost::any_cast<bool>(first->GetParameters().at("ClearEligibilityAfterTrigger")), "first generated default changed");
            Require(!boost::any_cast<bool>(second->GetParameters().at("ClearEligibilityAfterTrigger")), "second generated default changed");
        }
        for (const char* name : {"R_STDP", "R-STDP"}) {
            std::unique_ptr<LearningRule> rule(LearningRuleModelFactory::createLearningRuleModel({name, {}}));
            Require(dynamic_cast<R_STDP*>(rule.get()) != nullptr, "default/alias factory changed");
        }
        for (const auto& bad : std::vector<std::map<std::string, boost::any>>{
                 {{"LTP_tau", 0.0f}}, {{"LTD_tau", -1.0f}}, {{"unknown", 1.0f}},
                 {{"RewardFactor", std::numeric_limits<float>::infinity()}}}) {
            bool rejected = false;
            try { CustomRStdpV1 rule(bad); } catch (const std::invalid_argument&) { rejected = true; }
            Require(rejected, "invalid native parameter accepted");
        }
        Totals total;
        int scenario = 0;
        for (const char* generated_name : {"CustomRStdpV1", "CustomRStdpPersistentV1"})
        for (float dt : {0.1f, 1.0f}) {
            NeuronLayerDescription layer; layer.ModelName = "InputSpikeNeuronModel"; layer.numberofneuron = 1;
            Simulation context({layer}, {}, {}, 1000, dt, 1, EVENT_QUEUE_HEAP, 0);
            context.InitSimulation();
            for (bool clear : {false, true}) for (bool amplified : {false, true}) {
                std::array<int, 3> order{0, 3, 6};
                do {
                    std::map<std::string, boost::any> parameters{{"ClearEligibilityAfterTrigger", clear}};
                    if (amplified) { parameters["RewardFactor"] = 20.0f; parameters["PunishmentFactor"] = -20.0f; }
                    Fixture reference("R_STDP", parameters, dt), generated(generated_name, parameters, dt);
                    Require(dynamic_cast<R_STDP*>(reference.rule.get()) != nullptr, "handwritten factory changed");
                    Require(dynamic_cast<CustomRStdpV1*>(generated.rule.get()) != nullptr ||
                            dynamic_cast<CustomRStdpPersistentV1*>(generated.rule.get()) != nullptr,
                            "generated factory wrong");
                    Require(generated.rule->ImplementPostSynaptic() && generated.rule->ImplementTriggerSynaptic(), "event flags missing");
                    for (const auto& [key, value] : reference.rule->GetParameters()) {
                        const auto other = generated.rule->GetParameters().at(key);
                        if (key == "ClearEligibilityAfterTrigger") Require(boost::any_cast<bool>(value) == boost::any_cast<bool>(other), "bool parameter mismatch");
                        else Require(boost::any_cast<float>(value) == boost::any_cast<float>(other), "float parameter mismatch");
                    }
                    int ordinal = 0;
                    Snapshot(out, scenario, ordinal++, 0, -1, reference, generated, total);
                    auto run = [&](int kind, int tick) {
                        std::array<float, 6> old_weights;
                        for (int c = 0; c < 6; ++c) old_weights[c] = generated.connections[c].weight;
                        float other_eligibility = generated.rule->State->GetStateValue(2, 2);
                        reference.Event(kind, tick, context); generated.Event(kind, tick, context);
                        if (kind != 3 && kind != 4) {
                            for (int c = 0; c < 6; ++c) Require(generated.connections[c].weight == old_weights[c], "non-trigger changed weight");
                        } else {
                            Require(generated.rule->State->GetStateValue(2, 2) == other_eligibility, "trigger crossed target boundary");
                            if (clear) for (int c : {0, 1}) Require(generated.rule->State->GetStateValue(c, 2) == 0, "eligibility not cleared");
                        }
                        Snapshot(out, scenario, ordinal++, tick, kind, reference, generated, total);
                        ++total.events;
                    };
                    run(3, 0); run(4, 0); // Empty eligibility must not change any weight.
                    for (int cycle = 0; cycle < 30; ++cycle) {
                        int tick = cycle * 23 + 1;
                        for (int kind : order) run(kind, tick);
                        run(1, tick + 1); run(2, tick + 2); run(7, tick + 2);
                        run(4, tick + 3); run(3, tick + 3); run(6, tick + 19);
                    }
                    // Large gap exercises exponential underflow and repeated trigger semantics.
                    run(3, 1000000); run(4, 1000000);
                    ++scenario;
                } while (std::next_permutation(order.begin(), order.end()));
            }
        }
        Require(total.lower > 0 && total.upper > 0, "clamp boundaries not exercised");
        out.close();
        std::ofstream network_out(directory / "simulation_weight_trace.csv");
        network_out.exceptions(std::ios::badbit | std::ios::failbit);
        network_out << std::setprecision(std::numeric_limits<float>::max_digits10);
        network_out << "generated_rule,clear,delay,step,field,reference,generated\n";
        long long network_samples = 0, matched_spikes = 0;
        int simulation_cases = 0;
        for (const char* generated_name : {"CustomRStdpV1", "CustomRStdpPersistentV1"})
        for (bool clear : {false, true}) for (int delay : {1, 3, 7}) {
            std::vector<std::pair<int, int>> ref_spikes, gen_spikes;
            std::vector<std::string> ref_fields, gen_fields;
            auto reference = NetworkTrace("R_STDP", clear, delay, ref_spikes, ref_fields);
            auto generated = NetworkTrace(generated_name, clear, delay, gen_spikes, gen_fields);
            Require(ref_fields == gen_fields && !ref_fields.empty(), "network state layout mismatch");
            Require(reference.size() == generated.size() && ref_spikes == gen_spikes, "network mismatch");
            Require(!ref_spikes.empty(), "network has no postsynaptic spikes");
            for (size_t i = 0; i < reference.size(); ++i) {
                total.Compare(reference[i], generated[i]);
                network_out << generated_name << ',' << clear << ',' << delay << ',' << i / ref_fields.size() + 1 << ','
                            << ref_fields[i % ref_fields.size()] << ',' << reference[i] << ',' << generated[i] << '\n';
            }
            network_samples += reference.size(); matched_spikes += ref_spikes.size();
            ++simulation_cases;
        }
        network_out.close();
        NeuronLayerDescription layer; layer.ModelName = "InputSpikeNeuronModel"; layer.numberofneuron = 1;
        Simulation context({layer}, {}, {}, 1000, 0.1f, 1, EVENT_QUEUE_HEAP, 0);
        context.InitSimulation();
        std::vector<double> ref_times, gen_times;
        for (int round = 0; round < 7; ++round) {
            if (round % 2) { gen_times.push_back(Measure("CustomRStdpV1", context)); ref_times.push_back(Measure("R_STDP", context)); }
            else { ref_times.push_back(Measure("R_STDP", context)); gen_times.push_back(Measure("CustomRStdpV1", context)); }
        }
        std::sort(ref_times.begin(), ref_times.end()); std::sort(gen_times.begin(), gen_times.end());
        std::cout << "event_benchmark_reference_ms=" << ref_times[3] << " generated_ms=" << gen_times[3]
                  << " ratio=" << gen_times[3] / ref_times[3] << '\n';
        std::cout << "factory=PASS catalog=PASS scenarios=" << scenario << " events=" << total.events
                  << " trace_rows=" << total.rows << " comparisons=" << total.comparisons
                  << " max_abs_error=" << total.max_error << " lower_bound_samples=" << total.lower
                  << " upper_bound_samples=" << total.upper << '\n'
                  << "simulation_cases=" << simulation_cases << " simulation_samples=" << network_samples << " matched_spikes=" << matched_spikes
                  << " result=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
