#include "simulation_dense/DenseInterfaceSpikeNeuronModel.h"

#include "simulation_dense/DenseSubnetworkModel.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

namespace npgr {

DenseInterfaceSpikeNeuronModel::DenseInterfaceSpikeNeuronModel()
    : InputSpikeNeuronModel(),
      owner_(nullptr),
      interface_buffer_slot_count_(0),
      last_buffered_time_step_(-1),
      has_pending_interface_buffer_(false) {
    this->setModelName("DenseInterfaceSpikeNeuronModel");
}

DenseInterfaceSpikeNeuronModel::DenseInterfaceSpikeNeuronModel(DenseSubnetworkModel* owner)
    : InputSpikeNeuronModel(),
      owner_(owner),
      interface_buffer_slot_count_(0),
      last_buffered_time_step_(-1),
      has_pending_interface_buffer_(false) {
    this->setModelName("DenseInterfaceSpikeNeuronModel");
}

DenseInterfaceSpikeNeuronModel::~DenseInterfaceSpikeNeuronModel() {}

void DenseInterfaceSpikeNeuronModel::SetOwner(DenseSubnetworkModel* owner) {
    owner_ = owner;
}

DenseSubnetworkModel* DenseInterfaceSpikeNeuronModel::owner() const {
    return owner_;
}

  InternalSpike* DenseInterfaceSpikeNeuronModel::ProcessSpike(Interconnections* inter, int time) {
    if (owner_ == nullptr || inter == nullptr) {
        return nullptr;
    }
    if (inter->TargetNeuronModel != this) {
        return nullptr;
    }
    const int interface_slot_index = inter->TargetNeuronModelIndex;
    const float weight = inter->weight;
    const bool inhibitory = inter->type == 1;
    std::string reason;
    this->AccumulateInterfaceSpikeDelta(interface_slot_index, weight, inhibitory, time, &reason);
    return nullptr;
}

void DenseInterfaceSpikeNeuronModel::UpdateState(int index, int time, Simulation* simulation) {
    (void)index;
    (void)simulation;
    if (owner_ == nullptr || !has_pending_interface_buffer_ || time < last_buffered_time_step_) {
        return;
    }
    std::string reason;
    owner_->FlushInterfaceStateBuffers(time, &reason);
}

void DenseInterfaceSpikeNeuronModel::ResetInterfaceBufferState(int interface_buffer_slot_count) {
    interface_buffer_slot_count_ = interface_buffer_slot_count;
    last_buffered_time_step_ = -1;
    has_pending_interface_buffer_ = false;
}

bool DenseInterfaceSpikeNeuronModel::AccumulateInterfaceSpikeDelta(int interface_buffer_slot_index,
                                                                  float weight,
                                                                  bool inhibitory,
                                                                  int time_step,
                                                                  std::string* reason) {
    if (interface_buffer_slot_index < 0 ||
        interface_buffer_slot_index >= interface_buffer_slot_count_) {
        if (reason != nullptr) {
            *reason = "interface buffer slot index is out of range for interface spike buffers";
        }
        return false;
    }
    if (owner_ == nullptr) {
        if (reason != nullptr) {
            *reason = "interface spike owner is not bound";
        }
        return false;
    }
    if (time_step < 0) {
        if (reason != nullptr) {
            *reason = "interface spike buffer time_step must be non-negative";
        }
        return false;
    }
    if (!owner_->QueueInterfaceSpikeSlot(
            interface_buffer_slot_index, weight, inhibitory, time_step, reason)) {
        return false;
    }
    last_buffered_time_step_ = time_step;
    has_pending_interface_buffer_ = true;
    return true;
}

bool DenseInterfaceSpikeNeuronModel::HasPendingInterfaceBufferForTimeStep(int time_step) const {
    return has_pending_interface_buffer_ && last_buffered_time_step_ >= 0 && time_step >= last_buffered_time_step_;
}

void DenseInterfaceSpikeNeuronModel::ClearInterfaceStateBuffers() {
    last_buffered_time_step_ = -1;
    has_pending_interface_buffer_ = false;
}

bool DenseInterfaceSpikeNeuronModel::compare(NeuronModel* neuronModel) {
    if (!InputSpikeNeuronModel::compare(neuronModel)) {
        return false;
    }
    DenseInterfaceSpikeNeuronModel* other = dynamic_cast<DenseInterfaceSpikeNeuronModel*>(neuronModel);
    return other != nullptr && other->owner_ == this->owner_;
}

std::map<std::string, boost::any> DenseInterfaceSpikeNeuronModel::getParameters() {
    return std::map<std::string, boost::any>();
}

}  // namespace npgr
