#include "debug_monitor/DebugMonitorTypes.h"

namespace npgr {

void DebugMonitorFrame::Clear() {
    neuron_states.clear();
    spikes.clear();
    pending_channels.clear();
    inputconv_outputs.clear();
    inputconv_inputs.clear();
    inputconv_states.clear();
    outer_dynamic_states.clear();
    weights.clear();
}

bool DebugMonitorFrame::empty() const {
    return neuron_states.empty() &&
           spikes.empty() &&
           pending_channels.empty() &&
           inputconv_outputs.empty() &&
           inputconv_inputs.empty() &&
           inputconv_states.empty() &&
           outer_dynamic_states.empty() &&
           weights.empty();
}

const char* DebugComponentKindName(DebugComponentKind kind) {
    switch (kind) {
        case DebugComponentKind::MainNetwork:
            return "MainNetwork";
        case DebugComponentKind::DenseSubnetwork:
            return "DenseSubnetwork";
        case DebugComponentKind::InputConv:
            return "InputConv";
        case DebugComponentKind::OuterDynamic:
            return "OuterDynamic";
        default:
            return "Unknown";
    }
}

}  // namespace npgr
