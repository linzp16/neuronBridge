#ifndef OUTER_DYNAMIC_SPIKE_COUNTER_H
#define OUTER_DYNAMIC_SPIKE_COUNTER_H

#include "../source_file_realtime_v1_async/OuterDynamic/inc/OuterDynamicModel.h"

#include <mutex>
#include <vector>

// Stateful OuterDynamic sink that counts incoming network spikes per slot.
// TargetJoint in OuterDynamicConnectionDescription is interpreted as slot_id.
// The counter state is persistent and is cleared only through ClearAllSlots()
// or ClearSlot(), which makes this model useful as a user-readable spike sink.
class OuterDynamicSpikeCounter : public OuterDynamicModel {
public:
    OuterDynamicSpikeCounter();
    explicit OuterDynamicSpikeCounter(int timestep_size);
    virtual ~OuterDynamicSpikeCounter();

    virtual void Initialize(const OuterDynamicDescription& description, Simulation* simulation);
    virtual void Update(int time, Simulation* simulation);
    virtual int GetJointCount() const;
    virtual void AccumulateInputSpike(int slot_id, int type, float weight, int time);
    virtual void ClearAccumulatedInputs();

    int GetSlotCount() const;
    int GetSlotSpikeCount(int slot_id) const;
    float GetSlotWeightedSum(int slot_id) const;
    int GetSlotLastSpikeTime(int slot_id) const;
    std::vector<int> GetSpikeCounts() const;
    std::vector<float> GetWeightedSums() const;
    std::vector<int> GetLastSpikeTimes() const;

    int GetSlotSpikeCountByType(int slot_id, int type) const;
    float GetSlotWeightedSumByType(int slot_id, int type) const;
    std::vector<int> GetSpikeCountsByType(int type) const;
    std::vector<float> GetWeightedSumsByType(int type) const;

    void ClearAllSlots();
    bool ClearSlot(int slot_id);

private:
    static constexpr int kDefaultTypeCount = 4;

    int slot_count_;
    int type_count_;
    std::vector<int> spike_counts_;
    std::vector<float> weighted_sums_;
    std::vector<int> last_spike_times_;
    std::vector<int> spike_counts_by_type_;
    std::vector<float> weighted_sums_by_type_;
    mutable std::mutex mutex_;

    bool IsValidSlot(int slot_id) const;
    bool IsValidType(int type) const;
    int TypeOffset(int type, int slot_id) const;
};

#endif
