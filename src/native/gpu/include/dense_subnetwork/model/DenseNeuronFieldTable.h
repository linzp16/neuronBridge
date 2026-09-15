#ifndef NPGR_DENSE_NEURON_FIELD_TABLE_H
#define NPGR_DENSE_NEURON_FIELD_TABLE_H

#include "dense_subnetwork/model/DenseNeuronFieldSchema.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace npgr {

// One field occupies a contiguous field-major span: pool[offset + neuron_id].
struct DenseFieldSpan {
    int field_id = -1;
    std::string name;
    DenseFieldStorage storage = DenseFieldStorage::Float32;
    DenseFieldRole role = DenseFieldRole::Parameter;
    int offset = 0;
    int count = 0;
};

// Host-side flat field pools grouped by storage type.
struct DenseNeuronHostFieldTable {
    std::vector<DenseFieldSpan> fields;
    std::unordered_map<std::string, int> field_id_by_name;
    std::vector<float> float_pool;
    std::vector<int> int_pool;
    std::vector<unsigned char> byte_pool;
};

// POD-only span uploaded to CUDA kernels. Human-readable names stay host-side.
struct DenseDeviceFieldSpan {
    int field_id = -1;
    DenseFieldStorage storage = DenseFieldStorage::Float32;
    DenseFieldRole role = DenseFieldRole::Parameter;
    int offset = 0;
    int count = 0;
};

// Device-side view uploaded from DenseNeuronHostFieldTable.
struct DenseNeuronDeviceFieldTable {
    DenseDeviceFieldSpan* field_spans = nullptr;
    int field_count = 0;
    float* float_pool = nullptr;
    int* int_pool = nullptr;
    unsigned char* byte_pool = nullptr;
};

}  // namespace npgr

#endif
