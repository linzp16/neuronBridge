#ifndef NPGR_DENSEIZHIKEVICHEXPONENTIALDECAYDEVICEUPDATE_CUH
#define NPGR_DENSEIZHIKEVICHEXPONENTIALDECAYDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseIzhikevichExponentialDecayModel.h"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"

namespace npgr {

__device__ inline unsigned char UpdateIzhikevichExponentialDecayDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int,
    DensePendingChannelDeviceView pending,
    float dt_ms,
    int* fired_field_id) {
    *fired_field_id = field_ids[kIzhFired];
    const float arrival_exc =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float arrival_inh =
        ReadAndClearPendingChannel(pending, PendingChannel::InhibitoryConductance);
    const float arrival_nmda =
        ReadAndClearPendingChannel(pending, PendingChannel::NmdaConductance);
    const float input_current =
        ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* v_field = DeviceFloatField(fields, field_ids[kIzhV]);
    float* u_field = DeviceFloatField(fields, field_ids[kIzhU]);
    float* gexc = DeviceFloatField(fields, field_ids[kIzhGexc]);
    float* ginh = DeviceFloatField(fields, field_ids[kIzhGinh]);
    float* nmda_g = DeviceFloatField(fields, field_ids[kIzhNmdaG]);
    const float* a = DeviceFloatField(fields, field_ids[kIzhA]);
    const float* b = DeviceFloatField(fields, field_ids[kIzhB]);
    const float* c = DeviceFloatField(fields, field_ids[kIzhC]);
    const float* d = DeviceFloatField(fields, field_ids[kIzhD]);
    const float* v_threshold = DeviceFloatField(fields, field_ids[kIzhVThreshold]);
    const float* r = DeviceFloatField(fields, field_ids[kIzhR]);
    const float* e_exc = DeviceFloatField(fields, field_ids[kIzhEExc]);
    const float* e_inh = DeviceFloatField(fields, field_ids[kIzhEInh]);
    const float* exc_decay = DeviceFloatField(fields, field_ids[kIzhExcDecay]);
    const float* inh_decay = DeviceFloatField(fields, field_ids[kIzhInhDecay]);
    const float* nmda_decay = DeviceFloatField(fields, field_ids[kIzhNmdaDecay]);

    const float next_gexc = DenseDecayAndAdd(gexc[index], exc_decay[index], arrival_exc);
    const float next_ginh = DenseDecayAndAdd(ginh[index], inh_decay[index], arrival_inh);
    const float next_gnmda = DenseDecayAndAdd(nmda_g[index], nmda_decay[index], arrival_nmda);
    float v = v_field[index];
    float u = u_field[index];
    const float g_nmda_inf = DenseNmdaVoltageGate(v);
    const float total_current =
        DenseConductanceCurrent(next_gexc, e_exc[index], v) +
        DenseConductanceCurrent(next_ginh, e_inh[index], v) +
        DenseConductanceCurrent(next_gnmda * g_nmda_inf, e_exc[index], v) +
        input_current;
    DenseIzhikevichEulerStep(&v, &u, a[index], b[index], r[index], total_current, dt_ms);
    unsigned char did_fire = 0;
    if (v >= v_threshold[index]) {
        v = c[index];
        u += d[index];
        did_fire = 1;
    }
    v_field[index] = v;
    u_field[index] = u;
    gexc[index] = next_gexc;
    ginh[index] = next_ginh;
    nmda_g[index] = next_gnmda;
    return did_fire;
}

}  // namespace npgr

#endif
