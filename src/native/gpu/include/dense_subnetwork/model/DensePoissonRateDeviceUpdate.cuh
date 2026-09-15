#ifndef NPGR_DENSEPOISSONRATEDEVICEUPDATE_CUH
#define NPGR_DENSEPOISSONRATEDEVICEUPDATE_CUH

#include "dense_subnetwork/model/DenseIntegrationKernels.cuh"
#include "dense_subnetwork/model/DenseNeuronDeviceRuntime.cuh"
#include "dense_subnetwork/model/DensePoissonRateModel.h"

namespace npgr {

__device__ inline unsigned char UpdatePoissonRateDeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* field_ids,
    int index,
    int current_time_step,
    DensePendingChannelDeviceView pending,
    float dt_ms,
    int* fired_field_id) {
    *fired_field_id = field_ids[kPoissonFired];
    const float arrival_exc =
        ReadAndClearPendingChannel(pending, PendingChannel::ExcitatoryConductance);
    const float arrival_inh =
        ReadAndClearPendingChannel(pending, PendingChannel::InhibitoryConductance);
    const float input_current =
        ReadAndClearPendingChannel(pending, PendingChannel::Current);
    float* rate_hz_field = DeviceFloatField(fields, field_ids[kPoissonRateHz]);
    const float* rate_bias_hz = DeviceFloatField(fields, field_ids[kPoissonRateBiasHz]);
    const float* rate_gain_hz_per_current = DeviceFloatField(fields, field_ids[kPoissonRateGainHzPerCurrent]);
    float rate_hz = rate_bias_hz[index] + rate_gain_hz_per_current[index] * input_current + arrival_exc - arrival_inh;
    if (rate_hz < 0.0f) {
        rate_hz = 0.0f;
    }
    rate_hz_field[index] = rate_hz;
    const float fire_probability = DensePoissonFireProbability(rate_hz, dt_ms);
    return DenseHashUniform01(index, current_time_step) < fire_probability ? 1 : 0;
}

}  // namespace npgr

#endif
