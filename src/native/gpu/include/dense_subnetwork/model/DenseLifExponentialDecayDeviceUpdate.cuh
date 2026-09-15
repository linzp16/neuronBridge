#ifndef NPGR_DENSELIFEXPONENTIALDECAYDEVICEUPDATE_CUH
#define NPGR_DENSELIFEXPONENTIALDECAYDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseLifExponentialDecayModel.h"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"

namespace npgr {

__device__ inline unsigned char UpdateLifExponentialDecayDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int,
    DensePendingChannelDeviceView pending,
    float,
    int* fired_field_id) {
    *fired_field_id = field_ids[kLifDecayFired];
    const float arrival_exc =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float arrival_inh =
        ReadAndClearPendingChannel(pending, PendingChannel::InhibitoryConductance);
    const float input_current =
        ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* v = DeviceFloatField(fields, field_ids[kLifDecayV]);
    float* decay_g = DeviceFloatField(fields, field_ids[kLifDecayG]);
    int* steps_since_last_spike = DeviceIntField(fields, field_ids[kLifDecayStepsSinceLastSpike]);
    const float* v_rest = DeviceFloatField(fields, field_ids[kLifDecayVRest]);
    const float* v_reset = DeviceFloatField(fields, field_ids[kLifDecayVReset]);
    const float* v_threshold = DeviceFloatField(fields, field_ids[kLifDecayVThreshold]);
    const float* r = DeviceFloatField(fields, field_ids[kLifDecayR]);
    const float* e = DeviceFloatField(fields, field_ids[kLifDecayE]);
    const float* decay_g_alpha = DeviceFloatField(fields, field_ids[kLifDecayGAlpha]);
    const float* dt_inv_tau_m = DeviceFloatField(fields, field_ids[kLifDecayDtInvTauM]);
    const int* refractory_steps = DeviceIntField(fields, field_ids[kLifDecayRefractorySteps]);

    const float arrival_g = arrival_exc - arrival_inh;
    const float next_g = DenseDecayAndAdd(decay_g[index], decay_g_alpha[index], arrival_g);
    float next_v = v[index];
    const int last_spike_steps = steps_since_last_spike[index];
    if (last_spike_steps >= refractory_steps[index]) {
        next_v = DenseLifEulerStep(
            v[index],
            dt_inv_tau_m[index],
            r[index] * input_current +
                DenseConductanceCurrent(next_g, e[index], v[index]) +
                (v_rest[index] - v[index]));
    }
    int next_steps_since_last_spike = last_spike_steps;
    const unsigned char did_fire = DenseThresholdResetStep(
        &next_v,
        &next_steps_since_last_spike,
        v_threshold[index],
        v_reset[index]);
    v[index] = next_v;
    steps_since_last_spike[index] = next_steps_since_last_spike;
    decay_g[index] = next_g;
    return did_fire;
}

}  // namespace npgr

#endif
