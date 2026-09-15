#ifndef NPGR_DENSE_LEARNING_DEVICE_ENTRY_WRAPPERS_CUH
#define NPGR_DENSE_LEARNING_DEVICE_ENTRY_WRAPPERS_CUH

#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"

namespace npgr {

// Empty entry points keep unsupported phases explicit in the generated dispatch
// table while preserving one uniform call signature for every learning model.
__device__ inline void ApplyDenseLearningNoPreDeviceEntry(
    DenseLearningDeviceFieldTable,
    const int*,
    DenseDeviceLearningSpikeBufferTable,
    int,
    int,
    float,
    float*) {}

__device__ inline void ApplyDenseLearningNoPostDeviceEntry(
    DenseLearningDeviceFieldTable,
    const int*,
    int,
    int,
    float,
    float*) {}

__device__ inline void ApplyDenseLearningNoTriggerDeviceEntry(
    DenseLearningDeviceFieldTable,
    const int*,
    DenseDeviceLearningSpikeBufferTable,
    int,
    unsigned char,
    DenseDeviceLearningTriggerRouteTable,
    int,
    float,
    float*) {}

// Entry wrappers adapt model-specific device functions to the uniform dispatch
// signatures used by the dense propagation kernels.
__device__ inline void ApplyStdpPreDeviceEntry(DenseLearningDeviceFieldTable fields,
                                               const int* field_ids,
                                               DenseDeviceLearningSpikeBufferTable,
                                               int synapse_id,
                                               int pre_time_step,
                                               float dt_ms,
                                               float* syn_weight) {
    ApplyStdpPreDevice(fields, field_ids, synapse_id, pre_time_step, dt_ms, syn_weight);
}

__device__ inline void ApplyStdpPostDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                const int* field_ids,
                                                int synapse_id,
                                                int post_time_step,
                                                float dt_ms,
                                                float* syn_weight) {
    ApplyStdpPostDevice(fields, field_ids, synapse_id, post_time_step, dt_ms, syn_weight);
}

__device__ inline void ApplyRStdpPreDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                const int* field_ids,
                                                DenseDeviceLearningSpikeBufferTable,
                                                int synapse_id,
                                                int pre_time_step,
                                                float dt_ms,
                                                float*) {
    ApplyRStdpPreDevice(fields, field_ids, synapse_id, pre_time_step, dt_ms);
}

__device__ inline void ApplyRStdpPostDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                 const int* field_ids,
                                                 int synapse_id,
                                                 int post_time_step,
                                                 float dt_ms,
                                                 float*) {
    ApplyRStdpPostDevice(fields, field_ids, synapse_id, post_time_step, dt_ms);
}

__device__ inline void ApplyRStdpTriggerDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                    const int* field_ids,
                                                    DenseDeviceLearningSpikeBufferTable,
                                                    int trigger_synapse_id,
                                                    unsigned char trigger_type,
                                                    DenseDeviceLearningTriggerRouteTable routes,
                                                    int trigger_time_step,
                                                    float dt_ms,
                                                    float* syn_weight) {
    ApplyRStdpTriggerDevice(fields,
                            field_ids,
                            trigger_synapse_id,
                            trigger_type,
                            routes,
                            trigger_time_step,
                            dt_ms,
                            syn_weight);
}

__device__ inline void ApplyAdditiveKernelPreDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                         const int* field_ids,
                                                         DenseDeviceLearningSpikeBufferTable,
                                                         int synapse_id,
                                                         int pre_time_step,
                                                         float dt_ms,
                                                         float* syn_weight) {
    ApplyAdditiveKernelPreDevice(fields, field_ids, synapse_id, pre_time_step, dt_ms, syn_weight);
}

__device__ inline void ApplyAdditiveKernelTriggerDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                             const int* field_ids,
                                                             DenseDeviceLearningSpikeBufferTable,
                                                             int trigger_synapse_id,
                                                             unsigned char,
                                                             DenseDeviceLearningTriggerRouteTable routes,
                                                             int trigger_time_step,
                                                             float dt_ms,
                                                             float* syn_weight) {
    ApplyAdditiveKernelTriggerDevice(fields,
                                     field_ids,
                                     trigger_synapse_id,
                                     routes,
                                     trigger_time_step,
                                     dt_ms,
                                     syn_weight);
}

__device__ inline void ApplyCerebellarPreDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                     const int* field_ids,
                                                     DenseDeviceLearningSpikeBufferTable spike_buffers,
                                                     int synapse_id,
                                                     int pre_time_step,
                                                     float dt_ms,
                                                     float* syn_weight) {
    ApplyCerebellarPreDevice(fields, field_ids, spike_buffers, synapse_id, pre_time_step, dt_ms, syn_weight);
}

__device__ inline void ApplyCerebellarTriggerDeviceEntry(DenseLearningDeviceFieldTable fields,
                                                         const int* field_ids,
                                                         DenseDeviceLearningSpikeBufferTable spike_buffers,
                                                         int trigger_synapse_id,
                                                         unsigned char,
                                                         DenseDeviceLearningTriggerRouteTable routes,
                                                         int trigger_time_step,
                                                         float dt_ms,
                                                         float* syn_weight) {
    ApplyCerebellarTriggerDevice(fields,
                                 field_ids,
                                 spike_buffers,
                                 trigger_synapse_id,
                                 routes,
                                 trigger_time_step,
                                 dt_ms,
                                 syn_weight);
}



}  // namespace npgr

#endif
