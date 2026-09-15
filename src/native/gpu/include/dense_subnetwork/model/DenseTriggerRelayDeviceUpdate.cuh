#ifndef NPGR_DENSETRIGGERRELAYDEVICEUPDATE_CUH
#define NPGR_DENSETRIGGERRELAYDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"
#include "dense_subnetwork/model/DenseTriggerRelayModel.h"

namespace npgr {

__device__ inline unsigned char UpdateTriggerRelayDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int,
    DensePendingChannelDeviceView pending,
    float,
    int* fired_field_id) {
    *fired_field_id = field_ids[kTriggerRelayFired];
    const float arrival_exc =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float* threshold = DeviceFloatField(fields, field_ids[kTriggerRelayThreshold]);
    float* last_exc_input = DeviceFloatField(fields, field_ids[kTriggerRelayLastExcInput]);

    // The relay turns an ordinary excitatory boundary input into a dense-local
    // spike immediately. Trigger semantics are then expressed by an internal
    // dense trigger synapse, keeping propagation on the unified GPU path.
    last_exc_input[index] = arrival_exc;
    return arrival_exc > threshold[index] ? 1 : 0;
}


}  // namespace npgr

#endif
