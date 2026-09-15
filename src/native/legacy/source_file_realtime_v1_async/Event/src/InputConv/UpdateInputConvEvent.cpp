#include "../source_file_realtime_v1_async/Event/inc/InputConv/UpdateInputConvEvent.h"

#include "../source_file_realtime_v1_async/InputConv/inc/InputConvModel.h"
#include "../source_file_realtime_v1_async/Simulation/inc/Simulation.h"

UpdateInputConvEvent::UpdateInputConvEvent(int time, int queue_index, InputConvModel* model)
    : Event(time, queue_index), model_(model) {}

UpdateInputConvEvent::~UpdateInputConvEvent() {}

void UpdateInputConvEvent::ProcessEvent(Simulation* simulation) {
    const int current_time = this->getTime();
    this->model_->Update(current_time, simulation);
    simulation->FlushInputConvMainNetworkCurrent(this->model_, current_time);
    simulation->EventHeap->Insert_a_Event(
        new UpdateInputConvEvent(
            current_time + this->model_->getTimestepSize(),
            this->model_->getQueueIndex(),
            this->model_),
        this->model_->getQueueIndex());
}

void UpdateInputConvEvent::ProcessEvent(Simulation* simulation, RealTimeRestrictionLevel level) {
    (void)level;
    this->ProcessEvent(simulation);
}

enum EventPriority UpdateInputConvEvent::getPriority() {
    return INPUTCONVEVENT;
}
