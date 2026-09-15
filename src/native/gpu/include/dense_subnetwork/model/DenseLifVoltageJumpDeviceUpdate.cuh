#ifndef NPGR_DENSELIFVOLTAGEJUMPDEVICEUPDATE_CUH
#define NPGR_DENSELIFVOLTAGEJUMPDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseLifVoltageJumpModel.h"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"

namespace npgr {

__device__ inline unsigned char UpdateLifVoltageJumpDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int,
    DensePendingChannelDeviceView pending,
    float,
    int* fired_field_id) {
    *fired_field_id = field_ids[kLifVoltageJumpFired];
    const float arrival_exc =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float arrival_inh =
        ReadAndClearPendingChannel(pending, PendingChannel::InhibitoryConductance);
    const float input_current =
        ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* v = DeviceFloatField(fields, field_ids[kLifVoltageJumpV]);
    int* steps_since_last_spike = DeviceIntField(fields, field_ids[kLifVoltageJumpStepsSinceLastSpike]);
    const float* v_rest = DeviceFloatField(fields, field_ids[kLifVoltageJumpVRest]);
    const float* v_reset = DeviceFloatField(fields, field_ids[kLifVoltageJumpVReset]);
    const float* v_threshold = DeviceFloatField(fields, field_ids[kLifVoltageJumpVThreshold]);
    const float* r = DeviceFloatField(fields, field_ids[kLifVoltageJumpR]);
    const float* membrane_alpha = DeviceFloatField(fields, field_ids[kLifVoltageJumpMembraneAlpha]);
    const int* refractory_steps = DeviceIntField(fields, field_ids[kLifVoltageJumpRefractorySteps]);

    float next_v = v[index] + arrival_exc - arrival_inh;
    const int last_spike_steps = steps_since_last_spike[index];
    if (last_spike_steps >= refractory_steps[index]) {
        next_v = DenseLifEulerStep(
            next_v,
            membrane_alpha[index],
            v_rest[index] - next_v + r[index] * input_current);
    }
    int next_steps_since_last_spike = last_spike_steps;
    const unsigned char did_fire = DenseThresholdResetStep(
        &next_v,
        &next_steps_since_last_spike,
        v_threshold[index],
        v_reset[index]);
    v[index] = next_v;
    steps_since_last_spike[index] = next_steps_since_last_spike;
    return did_fire;
}

}  // namespace npgr

#endif
