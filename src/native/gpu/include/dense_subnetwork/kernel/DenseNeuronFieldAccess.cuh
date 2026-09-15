#ifndef NPGR_DENSE_NEURON_FIELD_ACCESS_CUH
#define NPGR_DENSE_NEURON_FIELD_ACCESS_CUH

#include "dense_subnetwork/model/DenseNeuronFieldTable.h"

namespace npgr {

// Device helpers for model-owned flat fields. Each field is stored as
// field-major data: pool[field.offset + neuron_id].
__device__ inline float& DenseFloatField(DenseNeuronDeviceFieldTable table,
                                         int field_id,
                                         int neuron_id) {
    const DenseFieldSpan span = table.field_spans[field_id];
    return table.float_pool[span.offset + neuron_id];
}

__device__ inline int& DenseIntField(DenseNeuronDeviceFieldTable table,
                                     int field_id,
                                     int neuron_id) {
    const DenseFieldSpan span = table.field_spans[field_id];
    return table.int_pool[span.offset + neuron_id];
}

__device__ inline unsigned char& DenseByteField(DenseNeuronDeviceFieldTable table,
                                                int field_id,
                                                int neuron_id) {
    const DenseFieldSpan span = table.field_spans[field_id];
    return table.byte_pool[span.offset + neuron_id];
}

}  // namespace npgr

#endif
