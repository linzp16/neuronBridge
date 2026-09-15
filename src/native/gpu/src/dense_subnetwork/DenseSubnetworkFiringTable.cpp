#include "dense_subnetwork/DenseSubnetworkFiringTable.h"

namespace npgr {

DenseSubnetworkFiringTable::DenseSubnetworkFiringTable()
    : steps_to_keep_(0),
      current_time_step_(-1) {}

bool DenseSubnetworkFiringTable::Initialize(int steps_to_keep, std::string* reason) {
    if (steps_to_keep <= 0) {
        if (reason != nullptr) {
            *reason = "steps_to_keep must be positive";
        }
        return false;
    }
    steps_to_keep_ = steps_to_keep;
    current_time_step_ = -1;
    current_firing_ids_.clear();
    history_ring_.assign(static_cast<std::size_t>(steps_to_keep_), DenseFiringHistoryEntry{});
    return true;
}

bool DenseSubnetworkFiringTable::BeginStep(int time_step, std::string* reason) {
    if (steps_to_keep_ <= 0) {
        if (reason != nullptr) {
            *reason = "dense firing table is not initialized";
        }
        return false;
    }
    if (time_step < 0) {
        if (reason != nullptr) {
            *reason = "time_step must be non-negative";
        }
        return false;
    }
    current_time_step_ = time_step;
    current_firing_ids_.clear();
    return true;
}

void DenseSubnetworkFiringTable::CaptureCurrentStep(const std::vector<int>& firing_ids) {
    current_firing_ids_ = firing_ids;
    DenseFiringHistoryEntry& slot = history_ring_[static_cast<std::size_t>(current_time_step_ % steps_to_keep_)];
    slot.time_step = current_time_step_;
    slot.firing_ids = current_firing_ids_;
}

void DenseSubnetworkFiringTable::LoadHistorySnapshot(const std::vector<DenseFiringHistoryEntry>& history, int current_time_step) {
    current_time_step_ = current_time_step;
    history_ring_ = history;
    current_firing_ids_.clear();
    const DenseFiringHistoryEntry* entry = this->HistoryAtTime(current_time_step);
    if (entry != nullptr) {
        current_firing_ids_ = entry->firing_ids;
    }
}

int DenseSubnetworkFiringTable::current_time_step() const {
    return current_time_step_;
}

const std::vector<int>& DenseSubnetworkFiringTable::current_firing_ids() const {
    return current_firing_ids_;
}

const DenseFiringHistoryEntry* DenseSubnetworkFiringTable::HistoryAtTime(int time_step) const {
    if (steps_to_keep_ <= 0 || time_step < 0) {
        return nullptr;
    }
    const DenseFiringHistoryEntry& slot = history_ring_[static_cast<std::size_t>(time_step % steps_to_keep_)];
    if (slot.time_step != time_step) {
        return nullptr;
    }
    return &slot;
}

}  // namespace npgr
