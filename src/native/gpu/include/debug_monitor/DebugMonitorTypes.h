#ifndef NPGR_DEBUG_MONITOR_TYPES_H
#define NPGR_DEBUG_MONITOR_TYPES_H

#include <map>
#include <string>
#include <vector>

namespace npgr {

enum class DebugComponentKind {
    MainNetwork,
    DenseSubnetwork,
    InputConv,
    OuterDynamic
};

struct DebugMonitorConfig {
    bool enabled = false;
    std::string output_dir = "debug_monitor";
    int sample_interval_steps = 1;
    int flush_interval_steps = 100;

    bool record_spikes = true;
    bool record_state = true;
    bool record_weights = false;
    bool record_pending_channels = true;
    bool record_outer_dynamic_state = true;

    bool all_neurons = false;
    std::vector<int> neuron_ids;
    std::map<std::string, std::vector<int> > dense_local_neuron_ids;

    bool record_inputconv_outputs = true;
    bool record_inputconv_inputs = false;
    bool record_inputconv_internal_state = false;
    bool monitor_all_inputconv = false;
    std::vector<int> monitored_inputconv_indices;
    std::vector<std::string> monitored_inputconv_names;
};

struct DebugNeuronStateRecord {
    int time_step = 0;
    DebugComponentKind component_kind = DebugComponentKind::MainNetwork;
    int component_index = -1;
    std::string component_name;
    int global_neuron_id = -1;
    int local_neuron_id = -1;
    std::string field_name;
    float value = 0.0f;
};

struct DebugSpikeRecord {
    int time_step = 0;
    DebugComponentKind component_kind = DebugComponentKind::MainNetwork;
    int component_index = -1;
    std::string component_name;
    int global_neuron_id = -1;
    int local_neuron_id = -1;
};

struct DebugPendingChannelRecord {
    int time_step = 0;
    int subnetwork_index = -1;
    std::string subnetwork_name;
    int channel = -1;
    int local_neuron_id = -1;
    int global_neuron_id = -1;
    float value = 0.0f;
};

struct DebugInputConvOutputRecord {
    int time_step = 0;
    int inputconv_index = -1;
    std::string inputconv_name;
    int output_index = -1;
    float value = 0.0f;
};

struct DebugInputConvInputRecord {
    int time_step = 0;
    int inputconv_index = -1;
    std::string inputconv_name;
    int input_index = -1;
    float value = 0.0f;
};

struct DebugInputConvStateRecord {
    int time_step = 0;
    int inputconv_index = -1;
    std::string inputconv_name;
    std::string field_name;
    int state_index = -1;
    float value = 0.0f;
};

struct DebugOuterDynamicStateRecord {
    int time_step = 0;
    int component_index = -1;
    std::string component_name;
    std::string field_name;
    int index = -1;
    double value = 0.0;
};

struct DebugWeightRecord {
    int time_step = 0;
    DebugComponentKind component_kind = DebugComponentKind::MainNetwork;
    int component_index = -1;
    std::string component_name;
    int synapse_id = -1;
    float value = 0.0f;
};

struct DebugMonitorFrame {
    std::vector<DebugNeuronStateRecord> neuron_states;
    std::vector<DebugSpikeRecord> spikes;
    std::vector<DebugPendingChannelRecord> pending_channels;
    std::vector<DebugInputConvOutputRecord> inputconv_outputs;
    std::vector<DebugInputConvInputRecord> inputconv_inputs;
    std::vector<DebugInputConvStateRecord> inputconv_states;
    std::vector<DebugOuterDynamicStateRecord> outer_dynamic_states;
    std::vector<DebugWeightRecord> weights;

    void Clear();
    bool empty() const;
};

const char* DebugComponentKindName(DebugComponentKind kind);

}  // namespace npgr

#endif
