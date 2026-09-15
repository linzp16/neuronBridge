#ifndef NPGR_DENSE_LEARNING_DEVICE_RUNTIME_CUH
#define NPGR_DENSE_LEARNING_DEVICE_RUNTIME_CUH

#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"

namespace npgr {


__device__ inline float* LearningFloatField(DenseLearningDeviceFieldTable fields, int field_id) {
    const DenseDeviceFieldSpan span = fields.field_spans[field_id];
    return fields.float_pool + span.offset;
}

__device__ inline int* LearningIntField(DenseLearningDeviceFieldTable fields, int field_id) {
    const DenseDeviceFieldSpan span = fields.field_spans[field_id];
    return fields.int_pool + span.offset;
}

__device__ inline unsigned char* LearningByteField(DenseLearningDeviceFieldTable fields, int field_id) {
    const DenseDeviceFieldSpan span = fields.field_spans[field_id];
    return fields.byte_pool + span.offset;
}

__device__ inline float ClampDenseLearningWeight(float weight, float max_weight) {
    if (weight < 0.0f) {
        return 0.0f;
    }
    if (weight > max_weight) {
        return max_weight;
    }
    return weight;
}

__device__ inline float ClampDenseLearningWeightRange(float weight,
                                                      float min_weight,
                                                      float max_weight) {
    if (weight < min_weight) {
        return min_weight;
    }
    if (weight > max_weight) {
        return max_weight;
    }
    return weight;
}

__device__ inline float DenseStableUnitRandom(unsigned int seed,
                                              int rule_id,
                                              int synapse_id,
                                              int time_step) {
    unsigned int value = seed;
    value ^= static_cast<unsigned int>(rule_id) + 0x9e3779b9u;
    value *= 1664525u;
    value += 1013904223u;
    value ^= static_cast<unsigned int>(synapse_id) + 0x85ebca6bu;
    value *= 2246822519u;
    value ^= static_cast<unsigned int>(time_step) + 0xc2b2ae35u;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return static_cast<float>(value & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

__device__ inline void DecayPairTrace(int synapse_id,
                                      int time_step,
                                      float dt_ms,
                                      float* apre,
                                      float* apost,
                                      int* last_update_step,
                                      const float* inv_ltp_tau,
                                      const float* inv_ltd_tau) {
    const float delta_time = static_cast<float>(time_step - last_update_step[synapse_id]) * dt_ms;
    apre[synapse_id] *= expf(-delta_time * inv_ltp_tau[synapse_id]);
    apost[synapse_id] *= expf(-delta_time * inv_ltd_tau[synapse_id]);
    last_update_step[synapse_id] = time_step;
}


}  // namespace npgr

#endif
