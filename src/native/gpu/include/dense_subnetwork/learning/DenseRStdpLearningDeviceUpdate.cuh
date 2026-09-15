#ifndef NPGR_DENSERSTDPLEARNINGDEVICEUPDATE_CUH
#define NPGR_DENSERSTDPLEARNINGDEVICEUPDATE_CUH

#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseRStdpLearningModel.h"

namespace npgr {

__device__ inline void ApplyRStdpPreDevice(DenseLearningDeviceFieldTable fields,
                                           const int* field_ids,
                                           int synapse_id,
                                           int time_step,
                                           float dt_ms) {
    float* apre = LearningFloatField(fields, field_ids[kDenseRStdpApre]);
    float* apost = LearningFloatField(fields, field_ids[kDenseRStdpApost]);
    float* eligibility = LearningFloatField(fields, field_ids[kDenseRStdpEligibility]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseRStdpLastUpdateStep]);
    const float* max_ltd = LearningFloatField(fields, field_ids[kDenseRStdpMaxLtd]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseRStdpInvLtpTau]);
    const float* inv_ltd_tau = LearningFloatField(fields, field_ids[kDenseRStdpInvLtdTau]);
    DecayPairTrace(synapse_id, time_step, dt_ms, apre, apost, last_update_step, inv_ltp_tau, inv_ltd_tau);
    apre[synapse_id] += 1.0f;
    eligibility[synapse_id] += -max_ltd[synapse_id] * apost[synapse_id];
}

__device__ inline void ApplyRStdpPostDevice(DenseLearningDeviceFieldTable fields,
                                            const int* field_ids,
                                            int synapse_id,
                                            int time_step,
                                            float dt_ms) {
    float* apre = LearningFloatField(fields, field_ids[kDenseRStdpApre]);
    float* apost = LearningFloatField(fields, field_ids[kDenseRStdpApost]);
    float* eligibility = LearningFloatField(fields, field_ids[kDenseRStdpEligibility]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseRStdpLastUpdateStep]);
    const float* max_ltp = LearningFloatField(fields, field_ids[kDenseRStdpMaxLtp]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseRStdpInvLtpTau]);
    const float* inv_ltd_tau = LearningFloatField(fields, field_ids[kDenseRStdpInvLtdTau]);
    DecayPairTrace(synapse_id, time_step, dt_ms, apre, apost, last_update_step, inv_ltp_tau, inv_ltd_tau);
    apost[synapse_id] += 1.0f;
    eligibility[synapse_id] += max_ltp[synapse_id] * apre[synapse_id];
}

__device__ inline void ApplyRStdpTriggerDevice(DenseLearningDeviceFieldTable fields,
                                               const int* field_ids,
                                               int trigger_synapse_id,
                                               unsigned char trigger_type,
                                               DenseDeviceLearningTriggerRouteTable routes,
                                               int time_step,
                                               float dt_ms,
                                               float* syn_weight) {
    (void)trigger_synapse_id;
    float* apre = LearningFloatField(fields, field_ids[kDenseRStdpApre]);
    float* apost = LearningFloatField(fields, field_ids[kDenseRStdpApost]);
    float* eligibility = LearningFloatField(fields, field_ids[kDenseRStdpEligibility]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseRStdpLastUpdateStep]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseRStdpInvLtpTau]);
    const float* inv_ltd_tau = LearningFloatField(fields, field_ids[kDenseRStdpInvLtdTau]);
    const float* reward_factor = LearningFloatField(fields, field_ids[kDenseRStdpRewardFactor]);
    const float* punishment_factor = LearningFloatField(fields, field_ids[kDenseRStdpPunishmentFactor]);
    const unsigned char* clear_flag = LearningByteField(fields, field_ids[kDenseRStdpClearEligibility]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseRStdpMaxWeight]);
    const int start = routes.start[trigger_synapse_id];
    const int count = routes.count[trigger_synapse_id];
    const float factor = (trigger_type == 1u)
                             ? punishment_factor[trigger_synapse_id]
                             : reward_factor[trigger_synapse_id];
    for (int offset = 0; offset < count; ++offset) {
        const int plastic_synapse_id = routes.synapse_ids[start + offset];
        DecayPairTrace(plastic_synapse_id,
                       time_step,
                       dt_ms,
                       apre,
                       apost,
                       last_update_step,
                       inv_ltp_tau,
                       inv_ltd_tau);
        syn_weight[plastic_synapse_id] = ClampDenseLearningWeight(
            syn_weight[plastic_synapse_id] + factor * eligibility[plastic_synapse_id],
            max_weight[plastic_synapse_id]);
        if (clear_flag[plastic_synapse_id] != 0u) {
            eligibility[plastic_synapse_id] = 0.0f;
        }
    }
}

}  // namespace npgr

#endif
