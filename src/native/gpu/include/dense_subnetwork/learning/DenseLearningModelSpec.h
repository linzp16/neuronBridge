#ifndef NPGR_DENSE_LEARNING_MODEL_SPEC_H
#define NPGR_DENSE_LEARNING_MODEL_SPEC_H

#include "dense_subnetwork/learning/DenseLearningFieldTable.h"

#include <string>
#include <vector>

namespace npgr {

struct DenseLearningModelSpec {
    // Registered model id used by the host factory and CUDA dispatch.
    int factory_model_id = -1;
    // Dense runtime span id. The current registered-table layout keeps this
    // equal to factory_model_id so kernels can index spans directly.
    int model_id = -1;
    // Optional legacy rule metadata retained for diagnostics and future
    // selective-table builds. Runtime learning uses per-synapse rule ids.
    int legacy_rule_id = -1;
    int rule_flags = 0;
    int route_group_id = -1;
    std::string legacy_rule_name;
};

// Rule-indexed flat parameter fields. Pools are addressed as
// pool[span.offset + rule_id], unlike runtime learning fields which are
// synapse-indexed. This keeps model-specific parameters out of
// DenseLearningRuleInfo while preserving compact array storage.
using DenseLearningRuleFieldTable = DenseLearningHostFieldTable;

struct DenseLearningModelSpan {
    // Runtime span id and registered factory model id. Kernels fetch this span
    // using syn_plastic_model_id/syn_trigger_model_id.
    int model_id = -1;
    int factory_model_id = -1;
    // Slice inside DenseLearningModelFieldIndexTable::field_indices containing
    // this model's slot -> field id bindings.
    int field_index_offset = 0;
    int field_index_count = 0;
};

struct DenseLearningModelFieldIndexTable {
    // One span per registered learning model. Each span points into
    // field_indices, which stores model-local slot bindings.
    std::vector<DenseLearningModelSpan> spans;
    std::vector<int> field_indices;
};

struct DenseLearningTriggerRouteTable {
    // Trigger-synapse indexed CSR. Each route lists ordinary plastic synapses
    // on the same target neuron and legacy rule id.
    std::vector<int> start;
    std::vector<int> count;
    std::vector<int> synapse_ids;
};

struct DenseLearningSpikeBufferTable {
    // Per-synapse bucket id. Cerebellar plastic and trigger synapses sharing
    // (target_neuron, rule_id) point at the same bucket; non-cerebellar entries are -1.
    std::vector<int> synapse_bucket_id;
    // Bucket-major ring metadata. bucket_head advances monotonically and wraps
    // through bucket_capacity when choosing a payload slot.
    std::vector<int> bucket_start;
    std::vector<int> bucket_capacity;
    std::vector<int> bucket_head;
    std::vector<int> bucket_count;
    // Ring payload storing ordinary plastic pre-spike history for the
    // Cerebellar trigger rule. Each entry keeps the spike time and the plastic
    // synapse id that should receive a kernel-table update when triggered.
    std::vector<float> spike_time;
    std::vector<int> spike_synapse_id;
    std::vector<unsigned char> valid;
};

struct DenseDeviceLearningModelFieldIndexTable {
    DenseLearningModelSpan* spans = nullptr;
    int span_count = 0;
    int* field_indices = nullptr;
    int field_index_count = 0;
};

struct DenseDeviceLearningTriggerRouteTable {
    int* start = nullptr;
    int* count = nullptr;
    int* synapse_ids = nullptr;
    int synapse_count = 0;
};

struct DenseDeviceLearningSpikeBufferTable {
    // Device mirror of DenseLearningSpikeBufferTable. Kernels treat it as a
    // fixed-capacity ring per (target_neuron, rule_id) bucket.
    int* synapse_bucket_id = nullptr;
    int* bucket_start = nullptr;
    int* bucket_capacity = nullptr;
    int* bucket_head = nullptr;
    int* bucket_count = nullptr;
    float* spike_time = nullptr;
    int* spike_synapse_id = nullptr;
    unsigned char* valid = nullptr;
    int bucket_count_value = 0;
    int payload_count = 0;
};

}  // namespace npgr

#endif
