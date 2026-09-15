#include "source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h"
#include "source_file_realtime_v1_async/NeuralModel/inc/NeuralStateVector/Neuron_State_Vector.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<NeuronModel> model(
        NeuronModelFactory::createNeuronModel(
            "TimeDrivenLIF_Exponential_double", {}, 1, 0.1f, 0));
    if (!model) {
        std::cerr << "result=FAIL reason=factory returned null\n";
        return EXIT_FAILURE;
    }

    model->InitStateVector(1, 0);
    Interconnections excitatory;
    excitatory.type = 0;
    excitatory.TargetNeuronModelIndex = 0;
    excitatory.weight = 3.0f;
    model->CheckType(&excitatory);

    int spike_count = 0;
    double voltage_sum = 0.0;
    for (int step = 0; step < 2000; ++step) {
        if (step % 19 == 0) {
            model->ProcessSpike(&excitatory, step);
        }
        model->UpdateState(-1, step, nullptr);
        spike_count += model->StateVector->NumberofSpike;
        const float voltage = model->StateVector->GetNeuronState(0)[0];
        if (!std::isfinite(voltage)) {
            std::cerr << "result=FAIL reason=non-finite voltage\n";
            return EXIT_FAILURE;
        }
        voltage_sum += voltage;
    }

    constexpr int kExpectedSpikeCount = 36;
    constexpr double kExpectedVoltageSum = -118974.0;
    if (spike_count != kExpectedSpikeCount ||
        std::abs(voltage_sum - kExpectedVoltageSum) > 1.0) {
        std::cerr << "result=FAIL reason=LIF trajectory baseline mismatch"
                  << " spike_count=" << spike_count
                  << " voltage_sum=" << voltage_sum << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "model=TimeDrivenLIF_Exponential_double\n"
              << "steps=2000 input_period=19 spike_count=" << spike_count << '\n'
              << "voltage_sum=" << voltage_sum << '\n'
              << "result=PASS\n";
    return EXIT_SUCCESS;
}
