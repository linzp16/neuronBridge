#ifndef NPGR_DENSE_LEARNING_FIELD_TABLE_H
#define NPGR_DENSE_LEARNING_FIELD_TABLE_H

#include "dense_subnetwork/model/DenseNeuronFieldTable.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace npgr {

// Host-side flat learning fields. Each field is field-major:
// pool[span.offset + synapse_id].
struct DenseLearningHostFieldTable {
    std::vector<DenseFieldSpan> fields;
    std::unordered_map<std::string, int> field_id_by_name;
    std::vector<float> float_pool;
    std::vector<int> int_pool;
    std::vector<unsigned char> byte_pool;
};

// Device-side learning field view consumed by synapse-parallel learning kernels.
struct DenseLearningDeviceFieldTable {
    DenseDeviceFieldSpan* field_spans = nullptr;
    int field_count = 0;
    float* float_pool = nullptr;
    int* int_pool = nullptr;
    unsigned char* byte_pool = nullptr;
};

}  // namespace npgr

#endif
