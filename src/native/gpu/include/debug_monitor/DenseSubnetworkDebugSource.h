#ifndef NPGR_DENSE_SUBNETWORK_DEBUG_SOURCE_H
#define NPGR_DENSE_SUBNETWORK_DEBUG_SOURCE_H

#include "debug_monitor/IDebugMonitorSource.h"

namespace npgr {

class DenseSubnetworkModel;

class DenseSubnetworkDebugSource : public IDebugMonitorSource {
public:
    DenseSubnetworkDebugSource(DenseSubnetworkModel* subnetwork, int subnetwork_index);

    DebugComponentKind kind() const override;
    int component_index() const override;
    const std::string& component_name() const override;
    bool HasAnyTarget(const DebugMonitorConfig& config) const override;
    bool Capture(int time_step,
                 const DebugMonitorConfig& config,
                 DebugMonitorFrame* frame,
                 std::string* reason) override;

private:
    DenseSubnetworkModel* subnetwork_;
    int subnetwork_index_;
    std::string name_;
};

}  // namespace npgr

#endif
