#ifndef NPGR_DENSECEREBELLARLEARNINGDEVICEUPDATE_CUH
#define NPGR_DENSECEREBELLARLEARNINGDEVICEUPDATE_CUH

#include "dense_subnetwork/learning/DenseLearningDeviceRuntime.cuh"
#include "dense_subnetwork/learning/DenseCerebellarLearningModel.h"

namespace npgr {

__device__ inline float DenseCerebellarKernel(float elapsed,
                                              float initpos,
                                              float maxpos) {
    if (elapsed < initpos) {
        return 0.0f;
    }
    const float width = fmaxf(maxpos - initpos, 1.0e-6f);
    const float shifted = elapsed - initpos;
    return (1.0f / width) * shifted * expf(-(shifted / width) + 1.0f);
}

__device__ inline void ApplyCerebellarPreDevice(DenseLearningDeviceFieldTable fields,
                                                const int* field_ids,
                                                DenseDeviceLearningSpikeBufferTable spike_buffers,
                                                int synapse_id,
                                                int time_step,
                                                float dt_ms,
    float* syn_weight) {
    const float* a1pre = LearningFloatField(fields, field_ids[kDenseCerebellarA1Pre]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseCerebellarMaxWeight]);
    const float* min_weight = LearningFloatField(fields, field_ids[kDenseCerebellarMinWeight]);
    const int* random_seed = LearningIntField(fields, field_ids[kDenseCerebellarRandomSeed]);
    const int* rule_id = LearningIntField(fields, field_ids[kDenseCerebellarRuleId]);
    if (spike_buffers.synapse_bucket_id != nullptr) {
        const int bucket_id = spike_buffers.synapse_bucket_id[synapse_id];
        if (bucket_id >= 0 &&
            bucket_id < spike_buffers.bucket_count_value) {
            const int capacity = spike_buffers.bucket_capacity[bucket_id];
            if (capacity > 0) {
                // Ordinary Cerebellar pre spikes are buffered by target/rule
                // bucket, matching the legacy BufferedActivityTime ownership.
                const int ticket = atomicAdd(&spike_buffers.bucket_head[bucket_id], 1);
                const int local = ticket % capacity;
                const int write_index = spike_buffers.bucket_start[bucket_id] + local;
                if (write_index >= 0 && write_index < spike_buffers.payload_count) {
                    spike_buffers.spike_time[write_index] = static_cast<float>(time_step) * dt_ms;
                    spike_buffers.spike_synapse_id[write_index] = synapse_id;
                    spike_buffers.valid[write_index] = 1u;
                    const int previous_count = atomicAdd(&spike_buffers.bucket_count[bucket_id], 1);
                    if (previous_count >= capacity) {
                        spike_buffers.bucket_count[bucket_id] = capacity;
                    }
                }
            }
        }
    }
    // Match the legacy cerebellar compromise: ordinary pre spikes apply
    // alpha multiplied by a deterministic U(0,1) sample, while trigger spikes
    // later apply the kernel-shaped beta term to buffered plastic synapses.
    const float pre_random = DenseStableUnitRandom(
        static_cast<unsigned int>(random_seed[synapse_id]),
        rule_id[synapse_id],
        synapse_id,
        time_step);
    syn_weight[synapse_id] = ClampDenseLearningWeightRange(
        syn_weight[synapse_id] + a1pre[synapse_id] * pre_random,
        min_weight[synapse_id],
        max_weight[synapse_id]);
}

__device__ inline void ApplyCerebellarTriggerDevice(DenseLearningDeviceFieldTable fields,
                                                    const int* field_ids,
                                                    DenseDeviceLearningSpikeBufferTable spike_buffers,
                                                    int trigger_synapse_id,
                                                    DenseDeviceLearningTriggerRouteTable routes,
                                                    int time_step,
                                                    float dt_ms,
                                                    float* syn_weight) {
    const float current_time = static_cast<float>(time_step) * dt_ms;
    const float* a2prepre = LearningFloatField(fields, field_ids[kDenseCerebellarA2PrePre]);
    const float* initpos = LearningFloatField(fields, field_ids[kDenseCerebellarInitPos]);
    const float* maxpos = LearningFloatField(fields, field_ids[kDenseCerebellarMaxPos]);
    const float* max_time_measured = LearningFloatField(fields, field_ids[kDenseCerebellarMaxTimeMeasured]);
    const float* max_weight = LearningFloatField(fields, field_ids[kDenseCerebellarMaxWeight]);
    const float* min_weight = LearningFloatField(fields, field_ids[kDenseCerebellarMinWeight]);
    if (spike_buffers.synapse_bucket_id == nullptr) {
        return;
    }
    const int bucket_id = spike_buffers.synapse_bucket_id[trigger_synapse_id];
    if (bucket_id < 0 ||
        bucket_id >= spike_buffers.bucket_count_value) {
        return;
    }
    const int bucket_start = spike_buffers.bucket_start[bucket_id];
    const int capacity = spike_buffers.bucket_capacity[bucket_id];
    const int live_count = min(spike_buffers.bucket_count[bucket_id], capacity);
    // Trigger connections replay all buffered ordinary pre spikes still inside
    // the Cerebellar measurement window for this target/rule bucket.
    for (int offset = 0; offset < live_count; ++offset) {
        const int read_index = bucket_start + offset;
        if (read_index < 0 ||
            read_index >= spike_buffers.payload_count ||
            spike_buffers.valid[read_index] == 0u) {
            continue;
        }
        const int plastic_synapse_id = spike_buffers.spike_synapse_id[read_index];
        if (plastic_synapse_id < 0) {
            continue;
        }
        const float elapsed = current_time - spike_buffers.spike_time[read_index];
        if (elapsed < 0.0f || elapsed > max_time_measured[plastic_synapse_id]) {
            continue;
        }
        const float kernel = DenseCerebellarKernel(
            elapsed,
            initpos[plastic_synapse_id],
            maxpos[plastic_synapse_id]);
        syn_weight[plastic_synapse_id] = ClampDenseLearningWeightRange(
            syn_weight[plastic_synapse_id] + a2prepre[plastic_synapse_id] * kernel,
            min_weight[plastic_synapse_id],
            max_weight[plastic_synapse_id]);
    }
}

}  // namespace npgr

#endif
