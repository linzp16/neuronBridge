#ifndef NPGR_DENSE_SUBNETWORK_FIRING_TABLE_H
#define NPGR_DENSE_SUBNETWORK_FIRING_TABLE_H

#include <string>
#include <vector>

namespace npgr {

struct DenseFiringHistoryEntry {
    // Simulation step represented by this host-side history slot.
    int time_step = -1;
    // Dense-local firing neuron ids for time_step.
    std::vector<int> firing_ids;
};

// Host-side diagnostic mirror of dense firing history.
//
// This class is not on the GPU main path. Dense propagation uses
// GpuPropagationRuntime device firing-history buffers; this table is populated
// from exported history for debug snapshots, tests, and monitor views.
class DenseSubnetworkFiringTable {
public:
    DenseSubnetworkFiringTable();

    bool Initialize(int steps_to_keep, std::string* reason = nullptr);
    bool BeginStep(int time_step, std::string* reason = nullptr);
    void CaptureCurrentStep(const std::vector<int>& firing_ids);
    void LoadHistorySnapshot(const std::vector<DenseFiringHistoryEntry>& history, int current_time_step);

    int current_time_step() const;
    const std::vector<int>& current_firing_ids() const;
    const DenseFiringHistoryEntry* HistoryAtTime(int time_step) const;

private:
    // Number of host history slots retained for diagnostic views.
    int steps_to_keep_;
    // Current simulation step represented by current_firing_ids_.
    int current_time_step_;
    // Host mirror of current-step firing ids.
    std::vector<int> current_firing_ids_;
    // Ring buffer of host-side firing history snapshots.
    std::vector<DenseFiringHistoryEntry> history_ring_;
};

}  // namespace npgr

#endif
