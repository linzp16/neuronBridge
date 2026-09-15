#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"

OuterDynamicModel::OuterDynamicModel()
    : name_(), timestep_size_(1), queue_index_(0), communication_interval_(1) {}

OuterDynamicModel::OuterDynamicModel(int timestep_size)
    : name_(), timestep_size_(timestep_size), queue_index_(0), communication_interval_(1) {}

OuterDynamicModel::~OuterDynamicModel() {}

void OuterDynamicModel::AccumulateInputSpike(int joint_id, int type, float weight, int time) {
    (void)joint_id;
    (void)type;
    (void)weight;
    (void)time;
}

void OuterDynamicModel::ClearAccumulatedInputs() {}

const std::string& OuterDynamicModel::name() const {
    return this->name_;
}

void OuterDynamicModel::setName(const std::string& name) {
    this->name_ = name;
}

int OuterDynamicModel::getTimestepSize() const {
    return this->timestep_size_;
}

void OuterDynamicModel::setTimestepSize(int timestep_size) {
    this->timestep_size_ = timestep_size;
}

int OuterDynamicModel::getQueueIndex() const {
    return this->queue_index_;
}

void OuterDynamicModel::setQueueIndex(int queue_index) {
    this->queue_index_ = queue_index;
}

int OuterDynamicModel::getCommunicationInterval() const {
    return this->communication_interval_;
}

void OuterDynamicModel::setCommunicationInterval(int interval) {
    this->communication_interval_ = interval;
}
