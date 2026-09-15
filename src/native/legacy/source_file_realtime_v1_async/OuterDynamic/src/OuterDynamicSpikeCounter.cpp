#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicSpikeCounter.h"

#include <boost/any.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
T GetAnyOrDefault(const std::map<std::string, boost::any>& parameters,
                  const char* name,
                  T default_value) {
    std::map<std::string, boost::any>::const_iterator found = parameters.find(name);
    if (found == parameters.end()) {
        return default_value;
    }
    return boost::any_cast<T>(found->second);
}

}  // namespace

OuterDynamicSpikeCounter::OuterDynamicSpikeCounter()
    : OuterDynamicModel(), slot_count_(0), type_count_(kDefaultTypeCount) {}

OuterDynamicSpikeCounter::OuterDynamicSpikeCounter(int timestep_size)
    : OuterDynamicModel(timestep_size), slot_count_(0), type_count_(kDefaultTypeCount) {}

OuterDynamicSpikeCounter::~OuterDynamicSpikeCounter() {}

void OuterDynamicSpikeCounter::Initialize(const OuterDynamicDescription& description, Simulation* simulation) {
    (void)simulation;
    this->setTimestepSize(description.update_timestep);
    this->setQueueIndex(description.queue_index);
    this->setCommunicationInterval(description.communication_interval);

    this->slot_count_ = GetAnyOrDefault<int>(description.ModelParameter, "slot_count", 0);
    this->type_count_ = GetAnyOrDefault<int>(description.ModelParameter, "type_count", kDefaultTypeCount);
    if (this->slot_count_ <= 0) {
        throw std::runtime_error("OuterDynamicSpikeCounter requires positive ModelParameter[\"slot_count\"].");
    }
    if (this->type_count_ <= 0) {
        throw std::runtime_error("OuterDynamicSpikeCounter requires positive ModelParameter[\"type_count\"].");
    }

    std::lock_guard<std::mutex> lock(this->mutex_);
    this->spike_counts_.assign(static_cast<std::size_t>(this->slot_count_), 0);
    this->weighted_sums_.assign(static_cast<std::size_t>(this->slot_count_), 0.0f);
    this->last_spike_times_.assign(static_cast<std::size_t>(this->slot_count_), -1);
    this->spike_counts_by_type_.assign(
        static_cast<std::size_t>(this->slot_count_) * static_cast<std::size_t>(this->type_count_), 0);
    this->weighted_sums_by_type_.assign(
        static_cast<std::size_t>(this->slot_count_) * static_cast<std::size_t>(this->type_count_), 0.0f);
}

void OuterDynamicSpikeCounter::Update(int time, Simulation* simulation) {
    (void)time;
    (void)simulation;
    // No plant dynamics are advanced here. Incoming spikes update the counter
    // state immediately in AccumulateInputSpike(), and users clear that state
    // explicitly through ClearAllSlots() or ClearSlot().
}

int OuterDynamicSpikeCounter::GetJointCount() const {
    return this->GetSlotCount();
}

void OuterDynamicSpikeCounter::AccumulateInputSpike(int slot_id, int type, float weight, int time) {
    if (!this->IsValidSlot(slot_id)) {
        return;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    // The aggregate arrays expose the total count/weight received by each
    // slot, independent of the excitatory/inhibitory connection type.
    this->spike_counts_[static_cast<std::size_t>(slot_id)] += 1;
    this->weighted_sums_[static_cast<std::size_t>(slot_id)] += weight;
    this->last_spike_times_[static_cast<std::size_t>(slot_id)] = time;
    if (this->IsValidType(type)) {
        // Per-type arrays are optional diagnostics for callers that want to
        // distinguish normal excitatory/inhibitory connection semantics.
        const int offset = this->TypeOffset(type, slot_id);
        this->spike_counts_by_type_[static_cast<std::size_t>(offset)] += 1;
        this->weighted_sums_by_type_[static_cast<std::size_t>(offset)] += weight;
    }
}

void OuterDynamicSpikeCounter::ClearAccumulatedInputs() {
    // Intentionally no-op: this model treats accumulated counts as persistent
    // user-visible state rather than as a per-update torque window.
}

int OuterDynamicSpikeCounter::GetSlotCount() const {
    return this->slot_count_;
}

int OuterDynamicSpikeCounter::GetSlotSpikeCount(int slot_id) const {
    if (!this->IsValidSlot(slot_id)) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->spike_counts_[static_cast<std::size_t>(slot_id)];
}

float OuterDynamicSpikeCounter::GetSlotWeightedSum(int slot_id) const {
    if (!this->IsValidSlot(slot_id)) {
        return 0.0f;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->weighted_sums_[static_cast<std::size_t>(slot_id)];
}

int OuterDynamicSpikeCounter::GetSlotLastSpikeTime(int slot_id) const {
    if (!this->IsValidSlot(slot_id)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->last_spike_times_[static_cast<std::size_t>(slot_id)];
}

std::vector<int> OuterDynamicSpikeCounter::GetSpikeCounts() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->spike_counts_;
}

std::vector<float> OuterDynamicSpikeCounter::GetWeightedSums() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->weighted_sums_;
}

std::vector<int> OuterDynamicSpikeCounter::GetLastSpikeTimes() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->last_spike_times_;
}

int OuterDynamicSpikeCounter::GetSlotSpikeCountByType(int slot_id, int type) const {
    if (!this->IsValidSlot(slot_id) || !this->IsValidType(type)) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->spike_counts_by_type_[static_cast<std::size_t>(this->TypeOffset(type, slot_id))];
}

float OuterDynamicSpikeCounter::GetSlotWeightedSumByType(int slot_id, int type) const {
    if (!this->IsValidSlot(slot_id) || !this->IsValidType(type)) {
        return 0.0f;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->weighted_sums_by_type_[static_cast<std::size_t>(this->TypeOffset(type, slot_id))];
}

std::vector<int> OuterDynamicSpikeCounter::GetSpikeCountsByType(int type) const {
    std::vector<int> result;
    if (!this->IsValidType(type)) {
        return result;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    result.reserve(static_cast<std::size_t>(this->slot_count_));
    for (int slot = 0; slot < this->slot_count_; ++slot) {
        result.push_back(this->spike_counts_by_type_[static_cast<std::size_t>(this->TypeOffset(type, slot))]);
    }
    return result;
}

std::vector<float> OuterDynamicSpikeCounter::GetWeightedSumsByType(int type) const {
    std::vector<float> result;
    if (!this->IsValidType(type)) {
        return result;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    result.reserve(static_cast<std::size_t>(this->slot_count_));
    for (int slot = 0; slot < this->slot_count_; ++slot) {
        result.push_back(this->weighted_sums_by_type_[static_cast<std::size_t>(this->TypeOffset(type, slot))]);
    }
    return result;
}

void OuterDynamicSpikeCounter::ClearAllSlots() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    std::fill(this->spike_counts_.begin(), this->spike_counts_.end(), 0);
    std::fill(this->weighted_sums_.begin(), this->weighted_sums_.end(), 0.0f);
    std::fill(this->last_spike_times_.begin(), this->last_spike_times_.end(), -1);
    std::fill(this->spike_counts_by_type_.begin(), this->spike_counts_by_type_.end(), 0);
    std::fill(this->weighted_sums_by_type_.begin(), this->weighted_sums_by_type_.end(), 0.0f);
}

bool OuterDynamicSpikeCounter::ClearSlot(int slot_id) {
    if (!this->IsValidSlot(slot_id)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->spike_counts_[static_cast<std::size_t>(slot_id)] = 0;
    this->weighted_sums_[static_cast<std::size_t>(slot_id)] = 0.0f;
    this->last_spike_times_[static_cast<std::size_t>(slot_id)] = -1;
    for (int type = 0; type < this->type_count_; ++type) {
        const int offset = this->TypeOffset(type, slot_id);
        this->spike_counts_by_type_[static_cast<std::size_t>(offset)] = 0;
        this->weighted_sums_by_type_[static_cast<std::size_t>(offset)] = 0.0f;
    }
    return true;
}

bool OuterDynamicSpikeCounter::IsValidSlot(int slot_id) const {
    return slot_id >= 0 && slot_id < this->slot_count_;
}

bool OuterDynamicSpikeCounter::IsValidType(int type) const {
    return type >= 0 && type < this->type_count_;
}

int OuterDynamicSpikeCounter::TypeOffset(int type, int slot_id) const {
    return type * this->slot_count_ + slot_id;
}
