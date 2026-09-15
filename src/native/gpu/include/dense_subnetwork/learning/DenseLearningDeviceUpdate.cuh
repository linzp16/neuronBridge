#ifndef NPGR_DENSE_LEARNING_DEVICE_UPDATE_CUH
#define NPGR_DENSE_LEARNING_DEVICE_UPDATE_CUH

#include "dense_subnetwork/learning/DenseBuiltinLearningDeviceUpdates.cuh"
#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"
#if defined(NR_ENABLE_MODEL_CODEGEN) && NR_ENABLE_MODEL_CODEGEN
#include "neuronbridge_codegen/CustomGeneratedDenseLearningDeviceUpdates.cuh"
#endif

namespace npgr {

__device__ inline void ApplyDenseLearningPreDevice(int factory_model_id,
                                                   DenseLearningDeviceFieldTable fields,
                                                   const int* field_ids,
                                                   DenseDeviceLearningSpikeBufferTable spike_buffers,
                                                   int synapse_id,
                                                   int pre_time_step,
                                                   float dt_ms,
                                                   float* syn_weight) {
    switch (factory_model_id) {
#define NPGR_DENSE_LEARNING_MODEL(symbol, id, host_class, pre_fn, post_fn, trigger_fn) \
        case symbol:                                                                  \
            pre_fn(fields,                                                            \
                   field_ids,                                                         \
                   spike_buffers,                                                     \
                   synapse_id,                                                        \
                   pre_time_step,                                                     \
                   dt_ms,                                                             \
                   syn_weight);                                                       \
            return;
#include "dense_subnetwork/learning/DenseLearningModelList.inc"
#undef NPGR_DENSE_LEARNING_MODEL
        default:
            return;
    }
}

__device__ inline void ApplyDenseLearningPostDevice(int factory_model_id,
                                                    DenseLearningDeviceFieldTable fields,
                                                    const int* field_ids,
                                                    int synapse_id,
                                                    int post_time_step,
                                                    float dt_ms,
                                                    float* syn_weight) {
    switch (factory_model_id) {
#define NPGR_DENSE_LEARNING_MODEL(symbol, id, host_class, pre_fn, post_fn, trigger_fn) \
        case symbol:                                                                  \
            post_fn(fields, field_ids, synapse_id, post_time_step, dt_ms, syn_weight); \
            return;
#include "dense_subnetwork/learning/DenseLearningModelList.inc"
#undef NPGR_DENSE_LEARNING_MODEL
        default:
            return;
    }
}

__device__ inline void ApplyDenseLearningTriggerDevice(int factory_model_id,
                                                       DenseLearningDeviceFieldTable fields,
                                                       const int* field_ids,
                                                       DenseDeviceLearningSpikeBufferTable spike_buffers,
                                                       int trigger_synapse_id,
                                                       unsigned char trigger_type,
                                                       DenseDeviceLearningTriggerRouteTable routes,
                                                       int trigger_time_step,
                                                       float dt_ms,
                                                       float* syn_weight) {
    switch (factory_model_id) {
#define NPGR_DENSE_LEARNING_MODEL(symbol, id, host_class, pre_fn, post_fn, trigger_fn) \
        case symbol:                                                                  \
            trigger_fn(fields,                                                        \
                       field_ids,                                                     \
                       spike_buffers,                                                 \
                       trigger_synapse_id,                                            \
                       trigger_type,                                                  \
                       routes,                                                        \
                       trigger_time_step,                                             \
                       dt_ms,                                                         \
                       syn_weight);                                                   \
            return;
#include "dense_subnetwork/learning/DenseLearningModelList.inc"
#undef NPGR_DENSE_LEARNING_MODEL
        default:
            return;
    }
}


}  // namespace npgr

#endif
