#ifndef NPGR_SIMULATION_DEBUG_MONITOR_H
#define NPGR_SIMULATION_DEBUG_MONITOR_H

#include "debug_monitor/DebugMonitorWriter.h"
#include "debug_monitor/IDebugMonitorSource.h"

#include <memory>
#include <string>
#include <vector>

class Simulation;

namespace npgr {

class SimulationDebugMonitor {
public:
    SimulationDebugMonitor();
    ~SimulationDebugMonitor();

    bool Initialize(Simulation* simulation,
                    const DebugMonitorConfig& config,
                    std::string* reason = nullptr);
    bool CaptureStep(int time_step, std::string* reason = nullptr);
    bool Flush(std::string* reason = nullptr);
    bool EnableInputConvMonitor(int inputconv_index, std::string* reason = nullptr);
    bool DisableInputConvMonitor(int inputconv_index, std::string* reason = nullptr);
    bool enabled() const;
    const DebugMonitorConfig& config() const;

private:
    void RegisterSources(Simulation* simulation);
    bool HasAnySourceTarget() const;

    Simulation* simulation_;
    DebugMonitorConfig config_;
    bool enabled_;
    int samples_since_flush_;
    std::vector<std::unique_ptr<IDebugMonitorSource> > sources_;
    DebugMonitorWriter writer_;
};

}  // namespace npgr

#endif
