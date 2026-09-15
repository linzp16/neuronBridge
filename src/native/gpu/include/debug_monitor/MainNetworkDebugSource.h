#ifndef NPGR_MAIN_NETWORK_DEBUG_SOURCE_H
#define NPGR_MAIN_NETWORK_DEBUG_SOURCE_H

#include "debug_monitor/IDebugMonitorSource.h"

#include <vector>

class Network;

namespace npgr {

class MainNetworkDebugSource : public IDebugMonitorSource {
public:
    explicit MainNetworkDebugSource(const Network* network,
                                    const std::vector<int>& original_to_main_neuron_id);

    DebugComponentKind kind() const override;
    int component_index() const override;
    const std::string& component_name() const override;
    bool HasAnyTarget(const DebugMonitorConfig& config) const override;
    bool Capture(int time_step,
                 const DebugMonitorConfig& config,
                 DebugMonitorFrame* frame,
                 std::string* reason) override;

private:
    bool IsSelected(int global_neuron_id, const DebugMonitorConfig& config) const;

    const Network* network_;
    std::string name_;
    std::vector<int> monitored_main_ids_;
    std::vector<int> main_to_original_ids_;
};

}  // namespace npgr

#endif
