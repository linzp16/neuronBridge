#ifndef NPGR_DENSELIFEXPONENTIALDOUBLEDEVICEUPDATE_CUH
#define NPGR_DENSELIFEXPONENTIALDOUBLEDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseLifExponentialDoubleModel.h"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"

namespace npgr {

__device__ inline unsigned char UpdateLifExponentialDoubleDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int,
    DensePendingChannelDeviceView pending,
    float,
    int* fired_field_id) {
    *fired_field_id = field_ids[kLifDoubleFired];
    const float arrival_exc =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float arrival_inh =
        ReadAndClearPendingChannel(pending, PendingChannel::InhibitoryConductance);
    const float input_current =
        ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* v = DeviceFloatField(fields, field_ids[kLifDoubleV]);
    float* gexc = DeviceFloatField(fields, field_ids[kLifDoubleGexc]);
    float* ginh = DeviceFloatField(fields, field_ids[kLifDoubleGinh]);
    int* steps_since_last_spike = DeviceIntField(fields, field_ids[kLifDoubleStepsSinceLastSpike]);
    const float* v_rest = DeviceFloatField(fields, field_ids[kLifDoubleVRest]);
    const float* v_reset = DeviceFloatField(fields, field_ids[kLifDoubleVReset]);
    const float* v_threshold = DeviceFloatField(fields, field_ids[kLifDoubleVThreshold]);
    const float* e_exc = DeviceFloatField(fields, field_ids[kLifDoubleEExc]);
    const float* e_inh = DeviceFloatField(fields, field_ids[kLifDoubleEInh]);
    const float* exc_decay = DeviceFloatField(fields, field_ids[kLifDoubleExcDecay]);
    const float* inh_decay = DeviceFloatField(fields, field_ids[kLifDoubleInhDecay]);
    const float* membrane_alpha = DeviceFloatField(fields, field_ids[kLifDoubleMembraneAlpha]);
    const int* refractory_steps = DeviceIntField(fields, field_ids[kLifDoubleRefractorySteps]);

    const float next_gexc = DenseDecayAndAdd(gexc[index], exc_decay[index], arrival_exc);
    const float next_ginh = DenseDecayAndAdd(ginh[index], inh_decay[index], arrival_inh);
    float next_v = v[index];
    const int last_spike_steps = steps_since_last_spike[index];
    if (last_spike_steps > refractory_steps[index]) {
        next_v = DenseLifEulerStep(
            v[index],
            membrane_alpha[index],
            v_rest[index] - v[index] +
                DenseConductanceCurrent(next_gexc, e_exc[index], v[index]) +
                DenseConductanceCurrent(next_ginh, e_inh[index], v[index]) +
                input_current);
    }
    int next_steps_since_last_spike = last_spike_steps;
    const unsigned char did_fire = DenseThresholdResetStep(
        &next_v,
        &next_steps_since_last_spike,
        v_threshold[index],
        v_reset[index]);
    v[index] = next_v;
    steps_since_last_spike[index] = next_steps_since_last_spike;
    gexc[index] = next_gexc;
    ginh[index] = next_ginh;
    return did_fire;
}

}  // namespace npgr

#endif
