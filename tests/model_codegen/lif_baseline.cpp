#include "source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/TimeDrivenLIF_Exponential_double.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "neuronbridge_codegen/CustomLifConductanceV1.h"
#include "neuron_model/NeuronModelCatalog.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <list>

std::vector<std::pair<int, int>> RunNetwork(const char* model, int delay, bool split, std::vector<float>& trace) {
    std::list<NeuronLayerDescription> layers;
    NeuronLayerDescription input;
    input.ModelName = "InputSpikeNeuronModel"; input.numberofneuron = 1;
    layers.push_back(input);
    NeuronLayerDescription lif;
    lif.ModelName = model; lif.numberofneuron = 1; lif.isOutput = true;
    lif.NeuronParameter["t_ref"] = 3;
    layers.push_back(lif);
    lif.ModelName = "TimeDrivenLIF_Exponential_double";
    // Keep the two LIF layers in separate model groups in both networks.
    lif.NeuronParameter["t_ref"] = split ? 4 : 3;
    layers.push_back(lif);
    ConnectionDescription connections;
    connections.SourceNeuron = {0, 1}; connections.TargetNeuron = {1, 2};
    connections.Type = {0, 0}; connections.Weight = {3.0f, 3.0f};
    connections.MaxWeight = connections.Weight; connections.Delay = {delay, delay};
    connections.SynapseRule = {-1, -1}; connections.TriggerRule = {-1, -1};
    Simulation simulation(layers, {connections}, {}, 1000, 0.1f, 1, EVENT_QUEUE_HEAP, 0);
    simulation.InitSimulation();
    std::vector<int> cells, times;
    for (int time = 1; time < 700; time += 19) { cells.push_back(0); times.push_back(time); }
    simulation.AddExternalSpikeActivity(times, cells);
    for (int step = 0; step < 1000; ++step) {
        simulation.RunSimulationStep(1);
        for (int id = 1; id <= 2; ++id) {
            auto& neuron = simulation.network->neurons[id];
            auto state = neuron.neuron_model->StateVector;
            for (int slot = 0; slot < 4; ++slot)
                trace.push_back(state->GetNeuronState(neuron.index_in_NeuronModel)[slot]);
        }
    }
    std::vector<std::pair<int, int>> result;
    for (const auto& spike : simulation.output_spike_driver->OutputBuffer)
        result.emplace_back(spike.time, spike.neuron);
    return result;
}

double MeasureUpdates(const char* name) {
    std::unique_ptr<NeuronModel> model(NeuronModelFactory::createNeuronModel(name, {}, 1, 0.1f, 0));
    model->InitStateVector(4096, 0);
    Interconnections input; input.type = 0;
    model->CheckType(&input);
    for (int i = 0; i < 4096; ++i) model->StateVector->SetNeuronState(i, 3, 25.0f);
    for (int i = 0; i < 50; ++i) model->UpdateState(-1, i, nullptr);
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i) model->UpdateState(-1, i, nullptr);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        auto& catalog = npgr::NeuronModelCatalog::Instance();
        Require(catalog.Resolve("CustomLifConductanceV1") != nullptr, "catalog registration missing");
        Require(catalog.IsSupported("CustomLifConductanceV1", npgr::NeuronBackend::LegacyCpu), "CPU missing");
        Require(catalog.IsSupported("CustomLifConductanceV1", npgr::NeuronBackend::LegacyGpu), "Legacy GPU missing");
        Require(catalog.ResolveImplementationName(
                    "CustomLifConductanceV1", npgr::NeuronBackend::LegacyGpu) ==
                    "CustomLifConductanceV1_GPU",
                "Legacy GPU implementation mismatch");
        Require(catalog.IsSupported("CustomLifConductanceV1", npgr::NeuronBackend::DenseGpu), "Dense support missing");
        Require(!catalog.IsPublicUserModel("CustomLifConductanceV1"), "experimental model public");
        float max_error = 0;
        int spikes = 0, comparisons = 0;
        for (int stride : {1, 2, 4}) {
            for (int mode = 0; mode < 4; ++mode) {
                std::map<std::string, boost::any> p;
                p["t_ref"] = 3;
                p["random_mu"] = std::array<float, 4>{0, 0.25f, 0.1f, 0};
                std::unique_ptr<NeuronModel> reference(NeuronModelFactory::createNeuronModel(
                    "TimeDrivenLIF_Exponential_double", p, stride, 0.1f, 0));
                std::unique_ptr<NeuronModel> generated(NeuronModelFactory::createNeuronModel(
                    "CustomLifConductanceV1", p, stride, 0.1f, 0));
                Require(dynamic_cast<CustomLifConductanceV1*>(generated.get()) != nullptr, "factory returned wrong type");
                reference->InitStateVector(3, 0); generated->InitStateVector(3, 0);
                Neuron target_a, target_b;
                target_a.index_in_NeuronModel = target_b.index_in_NeuronModel = 0;
                Interconnections exc, inh, current_a[2], current_b[2];
                exc.type = 0; exc.TargetNeuronModelIndex = 0; exc.weight = 2.0f;
                inh.type = 1; inh.TargetNeuronModelIndex = 0; inh.weight = 0.4f;
                if (mode >= 1) { reference->CheckType(&exc); generated->CheckType(&exc); }
                if (mode >= 2) { reference->CheckType(&inh); generated->CheckType(&inh); }
                for (int i = 0; i < 2; ++i) {
                    current_a[i].type = current_b[i].type = 3;
                    current_a[i].TargetNeuron = &target_a; current_b[i].TargetNeuron = &target_b;
                    reference->CheckType(&current_a[i]); generated->CheckType(&current_b[i]);
                }
                reference->InitializeInputCurrentSynapseStructure(); generated->InitializeInputCurrentSynapseStructure();
                // Equal-to-threshold and reset boundaries on two additional neurons.
                reference->StateVector->SetNeuronState(1, 0, -50.0f);
                generated->StateVector->SetNeuronState(1, 0, -50.0f);
                reference->StateVector->LastSpike[1] = generated->StateVector->LastSpike[1] = 0;
                reference->StateVector->SetNeuronState(2, 0, -49.0f);
                generated->StateVector->SetNeuronState(2, 0, -49.0f);
                for (int step = 0; step < 2000; ++step) {
                    if (mode >= 1 && step % 17 == 0) { reference->ProcessSpike(&exc, step); generated->ProcessSpike(&exc, step); }
                    if (mode >= 2 && step % 23 == 0) { reference->ProcessSpike(&inh, step); generated->ProcessSpike(&inh, step); }
                    if (mode == 3 && step % 7 == 0) {
                        const int slot = (step / 7) % 2;
                        float value = (step % 31) * 1.5f;
                        reference->ProcessCurrent(&current_a[slot], &target_a, value);
                        generated->ProcessCurrent(&current_b[slot], &target_b, value);
                    }
                    reference->UpdateState(-1, step * stride, nullptr);
                    generated->UpdateState(-1, step * stride, nullptr);
                    auto a = reference->StateVector; auto b = generated->StateVector;
                    Require(a->NumberofSpike == b->NumberofSpike, "spike count differs");
                    spikes += b->NumberofSpike;
                    for (int i = 0; i < a->NumberofSpike; ++i) Require(a->SpikeIndex[i] == b->SpikeIndex[i], "spike index differs");
                    for (int i = 0; i < 3; ++i) {
                        Require(a->LastSpike[i] == b->LastSpike[i], "refractory counter differs");
                        Require(a->LastUpdate[i] == b->LastUpdate[i], "update time differs");
                        for (int j = 0; j < 4; ++j) {
                            float error = std::abs(a->GetNeuronState(i)[j] - b->GetNeuronState(i)[j]);
                            Require(std::isfinite(error) && error <= 1e-5f, "state differs");
                            max_error = std::max(max_error, error); ++comparisons;
                        }
                    }
                }
                Interconnections invalid; invalid.type = 2;
                bool rejected = false;
                try { generated->CheckType(&invalid); } catch (const std::invalid_argument&) { rejected = true; }
                Require(rejected, "unsupported NMDA was accepted");
            }
        }
        Require(spikes > 0, "vacuous baseline without spikes");
        int network_spikes = 0, reordered_cases = 0, network_comparisons = 0;
        float network_error = 0;
        for (bool split : {false, true}) {
        for (int delay : {1, 3, 7}) {
            std::vector<float> reference_trace, generated_trace;
            auto reference = RunNetwork("TimeDrivenLIF_Exponential_double", delay, split, reference_trace);
            auto generated = RunNetwork("CustomLifConductanceV1", delay, split, generated_trace);
            Require(reference_trace.size() == generated_trace.size(), "trace length differs");
            for (std::size_t i = 0; i < reference_trace.size(); ++i) {
                float error = std::abs(reference_trace[i] - generated_trace[i]);
                Require(std::isfinite(error) && error <= 1e-5f, "Simulation state differs");
                network_error = std::max(network_error, error); ++network_comparisons;
            }
            auto time_order = [](auto a, auto b) { return a.first < b.first; };
            Require(std::is_sorted(reference.begin(), reference.end(), time_order), "reference time went backwards");
            Require(std::is_sorted(generated.begin(), generated.end(), time_order), "generated time went backwards");
            if (reference != generated) ++reordered_cases;
            // Heap ties have no neuron-ID ordering contract. Preserve duplicate
            // events but canonicalize same-tick output order for comparison.
            std::sort(reference.begin(), reference.end());
            std::sort(generated.begin(), generated.end());
            if (reference != generated) {
                std::cerr << "delay=" << delay << " reference_events=" << reference.size() << " generated_events=" << generated.size() << '\n';
                for (std::size_t i = 0; i < std::min(reference.size(), generated.size()); ++i)
                    if (reference[i] != generated[i]) {
                        std::cerr << "first_mismatch=" << i << " reference=" << reference[i].first << ',' << reference[i].second
                                  << " generated=" << generated[i].first << ',' << generated[i].second << '\n'; break;
                    }
            }
            Require(reference == generated, "Simulation output sequence differs");
            Require(generated.size() > 20, "insufficient activity from input train");
            Require(std::any_of(generated.begin(), generated.end(), [](auto s) { return s.second == 1; }), "middle LIF never fired");
            Require(std::any_of(generated.begin(), generated.end(), [](auto s) { return s.second == 2; }), "downstream LIF never fired");
            network_spikes += static_cast<int>(generated.size());
        }
        }
        std::vector<double> reference_ms, generated_ms;
        for (int round = 0; round < 7; ++round) {
            if (round % 2 == 0) {
                reference_ms.push_back(MeasureUpdates("TimeDrivenLIF_Exponential_double"));
                generated_ms.push_back(MeasureUpdates("CustomLifConductanceV1"));
            } else {
                generated_ms.push_back(MeasureUpdates("CustomLifConductanceV1"));
                reference_ms.push_back(MeasureUpdates("TimeDrivenLIF_Exponential_double"));
            }
        }
        std::sort(reference_ms.begin(), reference_ms.end());
        std::sort(generated_ms.begin(), generated_ms.end());
        std::cout << "simulation_cases=6 simulation_matched_spikes=" << network_spikes
                  << " simulation_mismatches=0\nbenchmark_neurons=4096 benchmark_updates=1000 rounds=7\nreference_median_ms="
                  << reference_ms[3] << " generated_median_ms=" << generated_ms[3]
                  << " ratio=" << generated_ms[3] / reference_ms[3] << '\n';
        std::cout << "same_tick_reordered_cases=" << reordered_cases << " simulation_state_comparisons="
                  << network_comparisons << " simulation_max_abs_error=" << network_error << '\n';
        std::cout << "factory=PASS catalog=PASS\ncases=12 steps_per_case=2000 state_comparisons=" << comparisons
                  << "\nmax_abs_error=" << max_error << "\nmatched_spikes=" << spikes
                  << "\nspike_mismatches=0\nresult=PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "result=FAIL reason=" << e.what() << '\n'; return 1;
    }
}
