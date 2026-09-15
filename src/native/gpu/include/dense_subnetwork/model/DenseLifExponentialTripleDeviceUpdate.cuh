#ifndef NPGR_DENSELIFEXPONENTIALTRIPLEDEVICEUPDATE_CUH
#define NPGR_DENSELIFEXPONENTIALTRIPLEDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseLifExponentialTripleModel.h"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"

namespace npgr {

__device__ inline unsigned char UpdateLifExponentialTripleDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int,
    DensePendingChannelDeviceView pending,
    float,
    int* fired_field_id) {
    *fired_field_id = field_ids[kLifTripleFired];
    const float arrival_ampa =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float arrival_gaba =
        ReadAndClearPendingChannel(pending, PendingChannel::InhibitoryConductance);
    const float arrival_nmda =
        ReadAndClearPendingChannel(pending, PendingChannel::NmdaConductance);
    const float input_current =
        ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* v = DeviceFloatField(fields, field_ids[kLifTripleV]);
    float* gampa = DeviceFloatField(fields, field_ids[kLifTripleGampa]);
    float* ggaba = DeviceFloatField(fields, field_ids[kLifTripleGgaba]);
    float* gnmda = DeviceFloatField(fields, field_ids[kLifTripleGnmda]);
    int* steps_since_last_spike = DeviceIntField(fields, field_ids[kLifTripleStepsSinceLastSpike]);
    const float* v_rest = DeviceFloatField(fields, field_ids[kLifTripleVRest]);
    const float* v_reset = DeviceFloatField(fields, field_ids[kLifTripleVReset]);
    const float* v_threshold = DeviceFloatField(fields, field_ids[kLifTripleVThreshold]);
    const float* r = DeviceFloatField(fields, field_ids[kLifTripleR]);
    const float* e_ampa = DeviceFloatField(fields, field_ids[kLifTripleEAmpa]);
    const float* e_gaba = DeviceFloatField(fields, field_ids[kLifTripleEGaba]);
    const float* ampa_decay = DeviceFloatField(fields, field_ids[kLifTripleAmpaDecay]);
    const float* gaba_decay = DeviceFloatField(fields, field_ids[kLifTripleGabaDecay]);
    const float* nmda_decay = DeviceFloatField(fields, field_ids[kLifTripleNmdaDecay]);
    const float* membrane_alpha = DeviceFloatField(fields, field_ids[kLifTripleMembraneAlpha]);
    const int* refractory_steps = DeviceIntField(fields, field_ids[kLifTripleRefractorySteps]);

    const float next_gampa = DenseDecayAndAdd(gampa[index], ampa_decay[index], arrival_ampa);
    const float next_ggaba = DenseDecayAndAdd(ggaba[index], gaba_decay[index], arrival_gaba);
    const float next_gnmda = DenseDecayAndAdd(gnmda[index], nmda_decay[index], arrival_nmda);
    float next_v = v[index];
    const int last_spike_steps = steps_since_last_spike[index];
    if (last_spike_steps > refractory_steps[index]) {
        const float g_nmda_inf = DenseNmdaVoltageGate(next_v);
        const float syn_current =
            DenseConductanceCurrent(next_gampa, e_ampa[index], next_v) +
            DenseConductanceCurrent(next_ggaba, e_gaba[index], next_v) +
            DenseConductanceCurrent(next_gnmda * g_nmda_inf, e_ampa[index], next_v);
        next_v = DenseLifEulerStep(
            next_v,
            membrane_alpha[index],
            v_rest[index] - next_v + r[index] * (syn_current + input_current));
    }
    int next_steps_since_last_spike = last_spike_steps;
    const unsigned char did_fire = DenseThresholdResetStep(
        &next_v,
        &next_steps_since_last_spike,
        v_threshold[index],
        v_reset[index]);
    v[index] = next_v;
    steps_since_last_spike[index] = next_steps_since_last_spike;
    gampa[index] = next_gampa;
    ggaba[index] = next_ggaba;
    gnmda[index] = next_gnmda;
    return did_fire;
}

}  // namespace npgr

#endif
