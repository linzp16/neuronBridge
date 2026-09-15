#include "source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector_Interface.cuh"
#include "source_file_realtime_v1_async/Openmp/inc/Openmp.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"
#include "neuron_model/NeuronModelCatalog.h"
#include "neuronbridge_codegen/CustomLifConductanceV1_GPU.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    std::unique_ptr<NeuronModel> cpu;
    std::unique_ptr<NeuronModel> gpu;
    Neuron cpu_target;
    Neuron gpu_target;
    Interconnections cpu_exc, gpu_exc, cpu_inh, gpu_inh;
    Interconnections cpu_current[2], gpu_current[2];

    Fixture() {
        std::map<std::string, boost::any> parameters;
        parameters["t_ref"] = 3;
        cpu.reset(NeuronModelFactory::createNeuronModel(
            "CustomLifConductanceV1", parameters, 1, 0.1f, 0));
        gpu.reset(NeuronModelFactory::createNeuronModel(
            "CustomLifConductanceV1_GPU", parameters, 1, 0.1f, 0));
        Require(cpu != nullptr && gpu != nullptr, "generated factory construction failed");
        Require(dynamic_cast<CustomLifConductanceV1_GPU*>(gpu.get()) != nullptr,
                "factory returned the wrong generated GPU class");
        cpu->InitStateVector(1, 0);
        gpu->InitStateVector(1, 0);
        gpu->StateVector->IsMonitored = true;

        cpu_target.index_in_NeuronModel = gpu_target.index_in_NeuronModel = 0;
        cpu_exc.type = gpu_exc.type = 0;
        cpu_exc.TargetNeuronModelIndex = gpu_exc.TargetNeuronModelIndex = 0;
        cpu_exc.weight = gpu_exc.weight = 3.5f;
        cpu_inh.type = gpu_inh.type = 1;
        cpu_inh.TargetNeuronModelIndex = gpu_inh.TargetNeuronModelIndex = 0;
        cpu_inh.weight = gpu_inh.weight = 0.75f;
        cpu->CheckType(&cpu_exc); gpu->CheckType(&gpu_exc);
        cpu->CheckType(&cpu_inh); gpu->CheckType(&gpu_inh);
        for (int index = 0; index < 2; ++index) {
            cpu_current[index].type = gpu_current[index].type = 3;
            cpu_current[index].TargetNeuron = &cpu_target;
            gpu_current[index].TargetNeuron = &gpu_target;
            cpu->CheckType(&cpu_current[index]);
            gpu->CheckType(&gpu_current[index]);
        }
        cpu->InitializeInputCurrentSynapseStructure();
        gpu->InitializeInputCurrentSynapseStructure();
    }
};

std::vector<std::pair<int, int>> RunMainNetwork(const char* model_name) {
    std::list<NeuronLayerDescription> layers;
    NeuronLayerDescription input;
    input.ModelName = "InputSpikeNeuronModel";
    input.numberofneuron = 1;
    layers.push_back(input);
    NeuronLayerDescription output;
    output.ModelName = model_name;
    output.numberofneuron = 1;
    output.isOutput = true;
    output.NeuronParameter["t_ref"] = 3;
    layers.push_back(output);
    ConnectionDescription connections;
    connections.SourceNeuron = {0};
    connections.TargetNeuron = {1};
    connections.Type = {0};
    connections.Weight = {3.5f};
    connections.MaxWeight = connections.Weight;
    connections.Delay = {1};
    connections.SynapseRule = {-1};
    connections.TriggerRule = {-1};
    Simulation simulation(layers, {connections}, {}, 1000, 0.1f, 1,
                          EVENT_QUEUE_HEAP, 0);
    simulation.InitSimulation();
    std::vector<int> cells;
    std::vector<int> times;
    for (int time = 1; time < 900; time += 13) {
        cells.push_back(0);
        times.push_back(time);
    }
    simulation.AddExternalSpikeActivity(times, cells);
    for (int tick = 0; tick < 1000; ++tick) simulation.RunSimulationStep(1);
    std::vector<std::pair<int, int>> result;
    for (const auto& spike : simulation.output_spike_driver->OutputBuffer)
        result.emplace_back(spike.time, spike.neuron);
    return result;
}

}  // namespace

int main() {
    try {
        setNumberOfOpenMPQueues(1);
        Require(NumberOfGPU > 0, "no CUDA device available for generated main-network GPU test");
        auto& catalog = npgr::NeuronModelCatalog::Instance();
        Require(catalog.IsSupported("CustomLifConductanceV1", npgr::NeuronBackend::LegacyGpu),
                "generated main-network GPU backend is not advertised");
        std::string reason;
        Require(catalog.ResolveImplementationName(
                    "CustomLifConductanceV1", npgr::NeuronBackend::LegacyGpu, &reason) ==
                    "CustomLifConductanceV1_GPU",
                "generated main-network GPU implementation does not resolve");

        float max_error = 0.0f;
        int spike_count = 0;
        {
        Fixture fixture;
        for (int tick = 0; tick < 1000; ++tick) {
            if (tick % 17 == 0) {
                fixture.cpu->ProcessSpike(&fixture.cpu_exc, tick);
                fixture.gpu->ProcessSpike(&fixture.gpu_exc, tick);
            }
            if (tick % 29 == 11) {
                fixture.cpu->ProcessSpike(&fixture.cpu_inh, tick);
                fixture.gpu->ProcessSpike(&fixture.gpu_inh, tick);
            }
            const int current_slot = (tick / 7) % 2;
            const float current = tick % 7 == 0 ? 16.0f : 0.0f;
            fixture.cpu->ProcessCurrent(&fixture.cpu_current[current_slot],
                                        &fixture.cpu_target, current);
            fixture.gpu->ProcessCurrent(&fixture.gpu_current[current_slot],
                                        &fixture.gpu_target, current);
            fixture.cpu->UpdateState(-1, tick, nullptr);
            fixture.gpu->UpdateState(-1, tick, nullptr);
            for (int state = 0; state < 4; ++state) {
                const float cpu_value = fixture.cpu->StateVector->GetPrintableValuesAt(0, state);
                const float gpu_value = fixture.gpu->StateVector->GetPrintableValuesAt(0, state);
                max_error = std::max(max_error, std::abs(cpu_value - gpu_value));
            }
            const bool cpu_fired = fixture.cpu->StateVector->NumberofSpike != 0;
            const bool gpu_fired =
                static_cast<Neuron_State_Vector_Interface*>(fixture.gpu->StateVector)
                    ->getInternalSpike()[0];
            Require(cpu_fired == gpu_fired, "generated CPU/GPU spike mismatch");
            spike_count += gpu_fired ? 1 : 0;
        }
        Require(max_error <= 2.0e-5f, "generated CPU/GPU state trajectory mismatch");
        Require(spike_count > 0, "generated main-network GPU baseline produced no spikes");
        }
        const auto reference_events = RunMainNetwork("CustomLifConductanceV1");
        const auto generated_events = RunMainNetwork("CustomLifConductanceV1_GPU");
        Require(!generated_events.empty(), "generated GPU Simulation emitted no output spikes");
        Require(reference_events == generated_events,
                "generated CPU/GPU Simulation event streams differ");
        const auto reverse_generated_events = RunMainNetwork("CustomLifConductanceV1_GPU");
        const auto reverse_reference_events = RunMainNetwork("CustomLifConductanceV1");
        Require(reverse_reference_events == reverse_generated_events,
                "generated CPU/GPU reverse-order event streams differ");
        Require(generated_events == reverse_generated_events,
                "generated GPU Simulation depends on construction order");
        std::cout << "generated_legacy_gpu_lif=PASS samples=1000 spikes=" << spike_count
                  << " network_events=" << generated_events.size() << " orders=2"
                  << " max_abs_error=" << max_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generated_legacy_gpu_lif=FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
