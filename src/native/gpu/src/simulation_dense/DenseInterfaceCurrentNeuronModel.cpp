#include "simulation_dense/DenseInterfaceCurrentNeuronModel.h"

#include "simulation_dense/DenseSubnetworkModel.h"
#include "source_file_realtime_v1_async/connections/inc/Interconnections.h"

#include <chrono>

namespace npgr {

namespace {

long long CurrentNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

DenseInterfaceCurrentNeuronModel::DenseInterfaceCurrentNeuronModel()
    : InputCurrentNeuronModel(), owner_(nullptr) {
    this->setModelName("DenseInterfaceCurrentNeuronModel");
}

DenseInterfaceCurrentNeuronModel::DenseInterfaceCurrentNeuronModel(DenseSubnetworkModel* owner)
    : InputCurrentNeuronModel(), owner_(owner) {
    this->setModelName("DenseInterfaceCurrentNeuronModel");
}

DenseInterfaceCurrentNeuronModel::~DenseInterfaceCurrentNeuronModel() {}

void DenseInterfaceCurrentNeuronModel::SetOwner(DenseSubnetworkModel* owner) {
    owner_ = owner;
}

DenseSubnetworkModel* DenseInterfaceCurrentNeuronModel::owner() const {
    return owner_;
}

void DenseInterfaceCurrentNeuronModel::ProcessCurrent(Interconnections* inter, Neuron* Target, float current) {
    const long long start_ns = CurrentNowNs();
    (void)Target;
    if (owner_ == nullptr || inter == nullptr) {
        return;
    }
    if (inter->TargetNeuronModel != this) {
        return;
    }
    const int current_connection_index = inter->TargetNeuronModelIndex;
    std::string reason;
    // Current inputs are stateful per legacy connection: a new current event
    // replaces that connection's previous value, then the dense runtime sums
    // all current connections feeding the same slot during flush.
    owner_->SetInterfaceCurrentConnection(
        current_connection_index,
        current,
        owner_->current_time_step(),
        &reason);
    DenseSubnetworkModel::RecordCurrentProcessTime(CurrentNowNs() - start_ns);
}

bool DenseInterfaceCurrentNeuronModel::compare(NeuronModel* neuronModel) {
    if (!InputCurrentNeuronModel::compare(neuronModel)) {
        return false;
    }
    DenseInterfaceCurrentNeuronModel* other = dynamic_cast<DenseInterfaceCurrentNeuronModel*>(neuronModel);
    return other != nullptr && other->owner_ == this->owner_;
}

std::map<std::string, boost::any> DenseInterfaceCurrentNeuronModel::getParameters() {
    return std::map<std::string, boost::any>();
}

}  // namespace npgr
