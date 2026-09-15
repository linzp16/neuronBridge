#ifndef NPGR_DENSE_NEURON_DEVICE_RUNTIME_CUH
#define NPGR_DENSE_NEURON_DEVICE_RUNTIME_CUH

#include "dense_subnetwork/model/DenseNeuronFieldTable.h"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"

namespace npgr {

struct DensePendingChannelDeviceView {
    // Device-side view over the channel-major pending input buffer for one
    // neuron. Model update functions interpret and drain only the channels
    // they own, keeping channel semantics out of the unified kernel.
    float* pending_channels;
    int channel_count;
    int channel_stride;
    int neuron_index;
};

using DenseDeviceModelUpdateFn = unsigned char (*)(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int neuron_index,
    int current_time_step,
    DensePendingChannelDeviceView pending,
    float dt_ms,
    int* fired_field_id);

__device__ inline float* DeviceFloatField(DenseNeuronDeviceFieldTable fields, int field_id) {
    const DenseDeviceFieldSpan span = fields.field_spans[field_id];
    return fields.float_pool + span.offset;
}

__device__ inline int* DeviceIntField(DenseNeuronDeviceFieldTable fields, int field_id) {
    const DenseDeviceFieldSpan span = fields.field_spans[field_id];
    return fields.int_pool + span.offset;
}

__device__ inline unsigned char* DeviceByteField(DenseNeuronDeviceFieldTable fields, int field_id) {
    const DenseDeviceFieldSpan span = fields.field_spans[field_id];
    return fields.byte_pool + span.offset;
}

__device__ inline float ReadAndClearPendingChannel(DensePendingChannelDeviceView pending,
                                                   PendingChannel channel) {
    const int channel_index = static_cast<int>(channel);
    if (pending.pending_channels == nullptr ||
        channel_index < 0 ||
        channel_index >= pending.channel_count ||
        pending.neuron_index < 0) {
        return 0.0f;
    }
    float* slot =
        pending.pending_channels + channel_index * pending.channel_stride + pending.neuron_index;
    const float value = *slot;
    *slot = 0.0f;
    return value;
}

}  // namespace npgr

#endif
