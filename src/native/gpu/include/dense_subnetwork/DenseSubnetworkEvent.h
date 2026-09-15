#ifndef NPGR_DENSE_SUBNETWORK_EVENT_H
#define NPGR_DENSE_SUBNETWORK_EVENT_H

#include "dense_subnetwork/DenseSubnetworkRuntimeGpu.h"

#include <string>
#include <vector>

namespace npgr {

struct DenseSubnetworkEventResult {
    int time_step = 0;
    std::vector<int> output_firing_ids;
    std::vector<DenseOutputSpike> output_spikes;
    bool expand_output_spikes = false;
};

class DenseSubnetworkEvent {
public:
    DenseSubnetworkEvent();
    explicit DenseSubnetworkEvent(int time_step);

    int time_step() const;
    void set_time_step(int time_step);

    bool Execute(DenseSubnetworkRuntimeGpu* runtime,
                 DenseSubnetworkEventResult* result,
                 std::string* reason = nullptr) const;

private:
    int time_step_;
};

}  // namespace npgr

#endif
