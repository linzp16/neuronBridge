#ifndef NPGR_GPU_PROPAGATION_LAYOUT_H
#define NPGR_GPU_PROPAGATION_LAYOUT_H

#include "dense_subnetwork/learning/DenseLearningModelSpec.h"

#include <cstdint>
#include <string>
#include <vector>

namespace npgr {

enum class PendingChannel : std::uint8_t {
    ExcitatoryConductance = 0,
    InhibitoryConductance = 1,
    Current = 2,
    NmdaConductance = 3,
    Modulatory = 4,
};

struct SynapseArrays {
    std::vector<int> post_neuron;
    std::vector<int> post_model;
    std::vector<float> weight;
    std::vector<float> max_weight;
    std::vector<std::uint8_t> type;
    // Compiled spike-effect channel per synapse. Propagation writes to
    // pending_channels[channel][post_neuron] instead of hard-coding gexc/ginh.
    std::vector<std::uint8_t> effect_channel;
    std::vector<float> effect_scale;
    // Plastic learning attachment derived from legacy SynapseRule.
    std::vector<int> plastic_rule_id;
    std::vector<int> plastic_model_id;
    std::vector<unsigned char> plastic_flags;
    std::vector<int> plastic_state_index;
    // Trigger learning attachment derived from legacy TriggerRule.
    std::vector<int> trigger_rule_id;
    std::vector<int> trigger_model_id;
    std::vector<unsigned char> trigger_flags;
};

struct PreDelaySlices {
    int neuron_count = 0;
    int delay_slot_count = 0;
    std::vector<int> start;
    std::vector<int> count;
    std::vector<int> synapse_ids;

    int FlatIndex(int pre_neuron, int delay_slot) const;
};

struct LayoutStats {
    int neuron_count = 0;
    int synapse_count = 0;
    int delay_slot_count = 0;
    int model_count = 0;
    int pending_channel_count = 0;
};

class GpuPropagationLayout {
public:
    LayoutStats stats;
    SynapseArrays synapses;
    PreDelaySlices pre_delay_slices;
    DenseLearningHostFieldTable learning_fields;
    DenseLearningModelFieldIndexTable learning_model_indices;
    DenseLearningTriggerRouteTable learning_trigger_routes;
    DenseLearningSpikeBufferTable learning_spike_buffers;

    bool IsValid(std::string* reason = nullptr) const;
};

}  // namespace npgr

#endif
