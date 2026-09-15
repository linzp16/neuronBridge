#include "simulation_dense/DenseSubnetworkUpdateEvent.h"

#include "simulation_dense/DenseSubnetworkModel.h"
#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"

namespace npgr {

DenseSubnetworkUpdateEvent::DenseSubnetworkUpdateEvent(int time, int queue_index, DenseSubnetworkModel* model)
    : Event(time, queue_index), model_(model) {}

DenseSubnetworkUpdateEvent::~DenseSubnetworkUpdateEvent() {}

void DenseSubnetworkUpdateEvent::ProcessEvent(Simulation* simulation) {
    if (model_ == nullptr) {
        return;
    }
    const int current_time = this->getTime();
    std::string reason;
    model_->AdvanceStep(current_time, simulation->EventHeap, &reason);
    simulation->EventHeap->Insert_a_Event(
        new DenseSubnetworkUpdateEvent(
            current_time + model_->update_timestep(),
            model_->queue_index(),
            model_),
        model_->queue_index());
}

void DenseSubnetworkUpdateEvent::ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level) {
    (void)level;
    this->ProcessEvent(simulation);
}

enum EventPriority DenseSubnetworkUpdateEvent::getPriority() {
    // Dense subnetworks read InputConv device outputs through D2D bindings.
    // EventQueue executes larger priority values first for the same timestep,
    // so OUTERUPDATEEVENT runs after INPUTCONVEVENT and sees the latest buffer.
    return OUTERUPDATEEVENT;
}

}  // namespace npgr
