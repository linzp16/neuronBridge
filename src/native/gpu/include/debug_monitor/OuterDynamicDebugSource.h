#ifndef NPGR_OUTER_DYNAMIC_DEBUG_SOURCE_H
#define NPGR_OUTER_DYNAMIC_DEBUG_SOURCE_H

#include "debug_monitor/IDebugMonitorSource.h"

class Simulation;

namespace npgr {

class OuterDynamicDebugSource : public IDebugMonitorSource {
public:
    explicit OuterDynamicDebugSource(const Simulation* simulation);

    DebugComponentKind kind() const override;
    int component_index() const override;
    const std::string& component_name() const override;
    bool HasAnyTarget(const DebugMonitorConfig& config) const override;
    bool Capture(int time_step,
                 const DebugMonitorConfig& config,
                 DebugMonitorFrame* frame,
                 std::string* reason) override;

private:
    const Simulation* simulation_;
    std::string name_;
};

}  // namespace npgr

#endif
