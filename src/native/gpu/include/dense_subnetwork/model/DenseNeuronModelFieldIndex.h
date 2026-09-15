#ifndef NPGR_DENSE_NEURON_MODEL_FIELD_INDEX_H
#define NPGR_DENSE_NEURON_MODEL_FIELD_INDEX_H

#include "dense_subnetwork/DenseNeuronModelSpec.h"
#include "dense_subnetwork/model/DenseNeuronFieldTable.h"

#include <string>
#include <vector>

namespace npgr {

struct DenseModelFieldIndexSpan {
    int model_id = -1;
    int factory_model_id = DenseNeuronModelFactory::kUnknownModelId;
    int field_index_offset = 0;
    int field_index_count = 0;
};

struct DenseModelFieldIndexTable {
    std::vector<DenseModelFieldIndexSpan> spans;
    std::vector<int> field_indices;
    std::vector<int> span_index_by_neuron;
};

struct DenseDeviceModelFieldIndexTable {
    DenseModelFieldIndexSpan* spans = nullptr;
    int span_count = 0;
    int* field_indices = nullptr;
    int field_index_count = 0;
    int* span_index_by_neuron = nullptr;
    int neuron_count = 0;
};

bool BuildDenseModelFieldIndexTable(const std::vector<DenseNeuronModelSpec>& specs,
                                    const DenseNeuronHostFieldTable& fields,
                                    int neuron_count,
                                    DenseModelFieldIndexTable* table,
                                    std::string* reason = nullptr);

}  // namespace npgr

#endif
