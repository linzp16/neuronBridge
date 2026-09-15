#include "../source_file_realtime_v1_async/NeuralModel/inc/EventDriven/TriggerRelayNeuronModel.h"

TriggerRelayNeuronModel::TriggerRelayNeuronModel() : EventDrivenInputDevice() {
    std::string name = "TriggerRelayNeuronModel";
    this->setModelName(name);
    this->TimeDriven = false;
}

TriggerRelayNeuronModel::TriggerRelayNeuronModel(int timestep)
    : EventDrivenInputDevice(timestep) {
    std::string name = "TriggerRelayNeuronModel";
    this->setModelName(name);
    this->TimeDriven = false;
}

TriggerRelayNeuronModel::~TriggerRelayNeuronModel() {}
