#ifndef NPGR_DENSEADDITIVEKERNELLEARNINGDEVICEUPDATE_CUH
#define NPGR_DENSEADDITIVEKERNELLEARNINGDEVICEUPDATE_CUH

#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseAdditiveKernelLearningModel.h"

namespace npgr {

__device__ inline void ApplyAdditiveKernelPreDevice(DenseLearningDeviceFieldTable fields,
                                                    const int* field_ids,
                                                    int synapse_id,
                                                    int time_step,
                                                    float dt_ms,
                                                    float* syn_weight) {
    float* pre_trace = LearningFloatField(fields, field_ids[kDenseAdditivePreTrace]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseAdditiveLastUpdateStep]);
    const float* a1pre = LearningFloatField(fields, field_ids[kDenseAdditiveA1Pre]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseAdditiveInvLtpTau]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseAdditiveMaxWeight]);
    const float delta_time = static_cast<float>(time_step - last_update_step[synapse_id]) * dt_ms;
    pre_trace[synapse_id] *= expf(-delta_time * inv_ltp_tau[synapse_id]);
    last_update_step[synapse_id] = time_step;
    pre_trace[synapse_id] += 1.0f;
    syn_weight[synapse_id] = ClampDenseLearningWeight(
        syn_weight[synapse_id] + a1pre[synapse_id],
        max_weight[synapse_id]);
}

__device__ inline void ApplyAdditiveKernelTriggerDevice(DenseLearningDeviceFieldTable fields,
                                                        const int* field_ids,
                                                        int trigger_synapse_id,
                                                        DenseDeviceLearningTriggerRouteTable routes,
                                                        int time_step,
                                                        float dt_ms,
                                                        float* syn_weight) {
    float* pre_trace = LearningFloatField(fields, field_ids[kDenseAdditivePreTrace]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseAdditiveLastUpdateStep]);
    const float* a2prepre = LearningFloatField(fields, field_ids[kDenseAdditiveA2PrePre]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseAdditiveInvLtpTau]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseAdditiveMaxWeight]);
    const int start = routes.start[trigger_synapse_id];
    const int count = routes.count[trigger_synapse_id];
    for (int offset = 0; offset < count; ++offset) {
        const int plastic_synapse_id = routes.synapse_ids[start + offset];
        const float delta_time =
            static_cast<float>(time_step - last_update_step[plastic_synapse_id]) * dt_ms;
        pre_trace[plastic_synapse_id] *= expf(-delta_time * inv_ltp_tau[plastic_synapse_id]);
        last_update_step[plastic_synapse_id] = time_step;
        syn_weight[plastic_synapse_id] = ClampDenseLearningWeight(
            syn_weight[plastic_synapse_id] +
                a2prepre[plastic_synapse_id] * pre_trace[plastic_synapse_id],
            max_weight[plastic_synapse_id]);
    }
}

}  // namespace npgr

#endif
