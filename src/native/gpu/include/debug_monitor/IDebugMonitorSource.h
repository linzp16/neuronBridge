#ifndef NPGR_IDEBUG_MONITOR_SOURCE_H
#define NPGR_IDEBUG_MONITOR_SOURCE_H

#include "debug_monitor/DebugMonitorTypes.h"

#include <string>

namespace npgr {

class IDebugMonitorSource {
public:
    virtual ~IDebugMonitorSource() {}

    virtual DebugComponentKind kind() const = 0;
    virtual int component_index() const = 0;
    virtual const std::string& component_name() const = 0;
    virtual bool HasAnyTarget(const DebugMonitorConfig& config) const = 0;

    virtual bool Capture(int time_step,
                         const DebugMonitorConfig& config,
                         DebugMonitorFrame* frame,
                         std::string* reason) = 0;
};

}  // namespace npgr

#endif
