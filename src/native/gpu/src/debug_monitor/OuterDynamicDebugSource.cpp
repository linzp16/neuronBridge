#include "debug_monitor/OuterDynamicDebugSource.h"

#include "source_file_realtime_v1_async/Simulation/inc/Simulation.h"

namespace npgr {

OuterDynamicDebugSource::OuterDynamicDebugSource(const Simulation* simulation)
    : simulation_(simulation),
      name_("outer_dynamic") {}

DebugComponentKind OuterDynamicDebugSource::kind() const {
    return DebugComponentKind::OuterDynamic;
}

int OuterDynamicDebugSource::component_index() const {
    return 0;
}

const std::string& OuterDynamicDebugSource::component_name() const {
    return name_;
}

bool OuterDynamicDebugSource::HasAnyTarget(const DebugMonitorConfig& config) const {
    return simulation_ != nullptr && config.record_outer_dynamic_state;
}

bool OuterDynamicDebugSource::Capture(int time_step,
                                      const DebugMonitorConfig& config,
                                      DebugMonitorFrame* frame,
                                      std::string* reason) {
    (void)reason;
    if (frame == nullptr || simulation_ == nullptr || !config.record_outer_dynamic_state) {
        return true;
    }
    std::vector<OuterDynamicStateSnapshot> states;
    if (!simulation_->GetLatestOuterDynamicStates(states)) {
        return true;
    }
    const char* groups[] = {"q", "qv", "qdd", "q_des", "qv_des", "tau_total"};
    for (std::size_t component = 0; component < states.size(); ++component) {
        const OuterDynamicStateSnapshot& snapshot = states[component];
        const std::array<double, 2>* component_values[] = {
            &snapshot.state.q,
            &snapshot.state.qv,
            &snapshot.state.qdd,
            &snapshot.state.q_des,
            &snapshot.state.qv_des,
            &snapshot.state.tau_total,
        };
        for (int group = 0; group < 6; ++group) {
            for (int index = 0; index < 2; ++index) {
                DebugOuterDynamicStateRecord record;
                record.time_step = time_step;
                record.component_index = snapshot.component_index;
                record.component_name = snapshot.component_name;
                record.field_name = groups[group];
                record.index = index;
                record.value = (*component_values[group])[static_cast<std::size_t>(index)];
                frame->outer_dynamic_states.push_back(record);
            }
        }
    }
    return true;
}

}  // namespace npgr
