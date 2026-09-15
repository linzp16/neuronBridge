#include "dense_subnetwork/DenseSubnetworkEvent.h"

namespace npgr {

DenseSubnetworkEvent::DenseSubnetworkEvent() : time_step_(0) {}

DenseSubnetworkEvent::DenseSubnetworkEvent(int time_step) : time_step_(time_step) {}

int DenseSubnetworkEvent::time_step() const {
    return time_step_;
}

void DenseSubnetworkEvent::set_time_step(int time_step) {
    time_step_ = time_step;
}

bool DenseSubnetworkEvent::Execute(DenseSubnetworkRuntimeGpu* runtime,
                                   DenseSubnetworkEventResult* result,
                                   std::string* reason) const {
    if (runtime == nullptr) {
        if (reason != nullptr) {
            *reason = "dense subnetwork runtime must not be null";
        }
        return false;
    }
    if (!runtime->BeginStep(time_step_, reason)) {
        return false;
    }
    if (!runtime->RunStep(reason)) {
        return false;
    }
    if (result != nullptr) {
        result->time_step = time_step_;
        result->output_firing_ids = runtime->current_step_output_firing_ids();
        if (result->expand_output_spikes) {
            result->output_spikes = runtime->ExpandCurrentStepOutputSpikes();
        } else {
            result->output_spikes.clear();
        }
    }
    return true;
}

}  // namespace npgr
