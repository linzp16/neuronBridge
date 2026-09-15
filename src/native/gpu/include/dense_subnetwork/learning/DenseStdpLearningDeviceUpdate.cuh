#ifndef NPGR_DENSESTDPLEARNINGDEVICEUPDATE_CUH
#define NPGR_DENSESTDPLEARNINGDEVICEUPDATE_CUH

#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseStdpLearningModel.h"

namespace npgr {

__device__ inline void ApplyStdpPreDevice(DenseLearningDeviceFieldTable fields,
                                          const int* field_ids,
                                          int synapse_id,
                                          int time_step,
                                          float dt_ms,
                                          float* syn_weight) {
    float* apre = LearningFloatField(fields, field_ids[kDenseStdpApre]);
    float* apost = LearningFloatField(fields, field_ids[kDenseStdpApost]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseStdpLastUpdateStep]);
    const float* max_ltd = LearningFloatField(fields, field_ids[kDenseStdpMaxLtd]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseStdpInvLtpTau]);
    const float* inv_ltd_tau = LearningFloatField(fields, field_ids[kDenseStdpInvLtdTau]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseStdpMaxWeight]);
    DecayPairTrace(synapse_id, time_step, dt_ms, apre, apost, last_update_step, inv_ltp_tau, inv_ltd_tau);
    apre[synapse_id] += 1.0f;
    syn_weight[synapse_id] = ClampDenseLearningWeight(
        syn_weight[synapse_id] - max_ltd[synapse_id] * apost[synapse_id],
        max_weight[synapse_id]);
}

__device__ inline void ApplyStdpPostDevice(DenseLearningDeviceFieldTable fields,
                                           const int* field_ids,
                                           int synapse_id,
                                           int time_step,
                                           float dt_ms,
                                           float* syn_weight) {
    float* apre = LearningFloatField(fields, field_ids[kDenseStdpApre]);
    float* apost = LearningFloatField(fields, field_ids[kDenseStdpApost]);
    int* last_update_step = LearningIntField(fields, field_ids[kDenseStdpLastUpdateStep]);
    const float* max_ltp = LearningFloatField(fields, field_ids[kDenseStdpMaxLtp]);
    const float* inv_ltp_tau = LearningFloatField(fields, field_ids[kDenseStdpInvLtpTau]);
    const float* inv_ltd_tau = LearningFloatField(fields, field_ids[kDenseStdpInvLtdTau]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseStdpMaxWeight]);
    DecayPairTrace(synapse_id, time_step, dt_ms, apre, apost, last_update_step, inv_ltp_tau, inv_ltd_tau);
    apost[synapse_id] += 1.0f;
    syn_weight[synapse_id] = ClampDenseLearningWeight(
        syn_weight[synapse_id] + max_ltp[synapse_id] * apre[synapse_id],
        max_weight[synapse_id]);
}

}  // namespace npgr

#endif
