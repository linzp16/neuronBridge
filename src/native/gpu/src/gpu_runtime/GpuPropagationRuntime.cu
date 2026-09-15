#include "gpu_runtime/GpuPropagationRuntime.h"
#include "gpu_runtime/GpuLaunchConfig.h"
#include "dense_subnetwork/learning/DenseLearningDeviceUpdate.cuh"
#include "dense_subnetwork/model/DenseNeuronModelFactory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace npgr {

namespace {

constexpr int kDrainPendingChannelCapacity = DenseNeuronModelFactory::kPendingChannelKeyStride;

bool CheckCuda(cudaError_t status, const char* action, std::string* reason) {
    if (status == cudaSuccess) {
        return true;
    }
    if (reason != nullptr) {
        std::ostringstream oss;
        oss << action << " failed: " << cudaGetErrorString(status);
        *reason = oss.str();
    }
    return false;
}

bool CopyIfNonEmpty(void* dst,
                    const void* src,
                    std::size_t byte_count,
                    cudaMemcpyKind kind,
                    const char* action,
                    std::string* reason) {
    if (byte_count == 0) {
        return true;
    }
    return CheckCuda(cudaMemcpy(dst, src, byte_count, kind), action, reason);
}

DenseDeviceFieldSpan ToDeviceFieldSpan(const DenseFieldSpan& host_span) {
    DenseDeviceFieldSpan device_span;
    device_span.field_id = host_span.field_id;
    device_span.storage = host_span.storage;
    device_span.role = host_span.role;
    device_span.offset = host_span.offset;
    device_span.count = host_span.count;
    return device_span;
}

__device__ inline int ResolveInterfaceTargetNeuron(int direct_target,
                                                   int interface_slot,
                                                   const int* interface_target_neuron_ids,
                                                   int interface_target_count) {
    if (direct_target >= 0) {
        return direct_target;
    }
    if (interface_slot >= 0 &&
        interface_slot < interface_target_count &&
        interface_target_neuron_ids != nullptr) {
        return interface_target_neuron_ids[interface_slot];
    }
    return -1;
}

__global__ void ApplyBoundaryStateVectorKernel(float* pending_channels,
                                               int pending_channel_count,
                                               int pending_channel_stride,
                                               const InterfaceChannelValue* values,
                                               int value_count,
                                               const int* interface_target_neuron_ids,
                                               int interface_target_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < value_count;
         index += blockDim.x * gridDim.x) {
        const InterfaceChannelValue value = values[index];
        if (value.value == 0.0f) {
            continue;
        }
        // get the dense target neuron id
        const int neuron_id = ResolveInterfaceTargetNeuron(value.target_neuron,
                                                           value.interface_slot,
                                                           interface_target_neuron_ids,
                                                           interface_target_count);
        // get the target channel of the target neurons
        const int channel = static_cast<int>(value.pending_channel);
        if (neuron_id < 0) {
            continue;
        }
        // Interface slots are finalized as unique (target neuron, pending channel)
        // pairs. Spike/conductance slots contain per-step event deltas, while
        // current slots contain the rebuilt per-connection state sum and must
        // overwrite the step's current channel.
        if (channel >= 0 && channel < pending_channel_count) {
            float* destination = &pending_channels[channel * pending_channel_stride + neuron_id];
            const float applied = value.value * value.scale;
            if (value.overwrite != 0) {
                *destination = applied;
            } else {
                *destination += applied;
            }
        }
    }
}

__global__ void ApplyDeviceInterfaceChannelValuesKernel(float* pending_channels,
                                                        int pending_channel_count,
                                                        int pending_channel_stride,
                                                        const float* source_values,
                                                        const DeviceToDeviceInterfaceCopySpec* specs,
                                                        int copy_count,
                                                        const int* interface_target_neuron_ids,
                                                        int interface_target_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < copy_count;
         index += blockDim.x * gridDim.x) {
        const DeviceToDeviceInterfaceCopySpec spec = specs[index];
        // get the target neuron id
        const int target = ResolveInterfaceTargetNeuron(spec.target_neuron,
                                                        spec.interface_slot,
                                                        interface_target_neuron_ids,
                                                        interface_target_count);
        const int channel = static_cast<int>(spec.pending_channel);
        if (target < 0 || channel < 0 || channel >= pending_channel_count || spec.source_index < 0) {
            continue;
        }
        const float value = source_values[spec.source_index] * spec.scale;
        if (value == 0.0f && !spec.overwrite) {
            continue;
        }
        float* destination = &pending_channels[channel * pending_channel_stride + target];
        if (spec.overwrite) {
            *destination = value;
        } else {
            atomicAdd(destination, value);
        }
    }
}

__global__ void ClearIsetWordsKernel(unsigned int* iset_words, int word_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < word_count;
         index += blockDim.x * gridDim.x) {
        iset_words[index] = 0u;
    }
}

__global__ void MarkIsetFromFiringHistorySharedKernel(const int* history_time_steps,
                                                      const int* history_counts,
                                                      const int* history_ids,
                                                      int steps_to_keep,
                                                      int slot_capacity,
                                                      const int* pre_slice_start,
                                                      const int* pre_slice_count,
                                                      int delay_slot_count,
                                                      int current_time_step,
                                                      const int* pre_slice_synapse_ids,
                                                      unsigned int* iset_words) {
    // assign the sharing memory
    __shared__ int shared_firing_ids[64];
    const int delay_slot = blockIdx.y;
    // scan the source firing time
    const int source_time = current_time_step - delay_slot;
    if (source_time < 0) {
        return;
    }
    const int slot_index = source_time % steps_to_keep;
    if (history_time_steps[slot_index] != source_time) {
        return;
    }
    // count the firing neurons
    const int firing_count = history_counts[slot_index];
    // the first 64 threads of the block scan the firing neurons
    const int chunk_start = blockIdx.x * 64;
    if (chunk_start >= firing_count) {
        return;
    }

    const int load_count = min(64, firing_count - chunk_start);
    // load the 64 firing neuron ids to the shared memory in each block
    if (threadIdx.x < 64) {
        shared_firing_ids[threadIdx.x] =
            (threadIdx.x < load_count)
                ? history_ids[slot_index * slot_capacity + chunk_start + threadIdx.x]
                : -1;
    }
    __syncthreads();
    // mark the iset words,each wrap process a single neuron, a wrap contain 32 threads, a block contain 4 wraps
    const int lane = threadIdx.x & 31; 
    const int warp_id = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    // wrap 0 process the 0,4,8....
    // wrap 1 process the 1,5,9....
    // 4 spiking neuron a processed in the same time by 4 wraps
    for (int local_index = warp_id; local_index < load_count; local_index += warp_count) {
        // get the pre_neuron id
        const int pre_neuron = shared_firing_ids[local_index];
        if (pre_neuron < 0) {
            continue;
        }
        // find the synapse stretch out from the pre_neuron
        const int flat_index = pre_neuron * delay_slot_count + delay_slot;
        const int slice_start = pre_slice_start[flat_index];
        const int slice_count = pre_slice_count[flat_index];
        for (int offset = lane; offset < slice_count; offset += 32) {
            // each thread marks the iset
            const int synapse_id = pre_slice_synapse_ids[slice_start + offset];
            const int word_index = synapse_id >> 5;
            const unsigned int mask = 1u << (synapse_id & 31);
            atomicOr(&iset_words[word_index], mask);
        }
    }
}

__global__ void DrainIsetToPendingKernel(float* pending_channels,
                                         int pending_channel_count,
                                         int pending_channel_stride,
                                         const int* post_synapse_start,
                                         const int* post_synapse_count,
                                         const int* post_synapse_ids,
                                         const float* syn_weight,
                                         const unsigned char* syn_effect_channel,
                                         const float* syn_effect_scale,
                                         const unsigned int* iset_words,
                                         int neuron_count) {
    // get the post neuron id
    for (int post_neuron = blockIdx.x * blockDim.x + threadIdx.x;
         post_neuron < neuron_count;
         post_neuron += blockDim.x * gridDim.x) {
        // get the start of the synapse slice
        const int start = post_synapse_start[post_neuron];
        // get the number of synapse in the slice
        const int count = post_synapse_count[post_neuron];
        // init the channel count
        float total_channels[kDrainPendingChannelCapacity] = {};
        for (int offset = 0; offset < count; ++offset) {
            // scan all the input synapse
            const int synapse_id = post_synapse_ids[start + offset];
            // get the iset mark of the synapse
            const unsigned int word = iset_words[synapse_id >> 5];
            const unsigned int mask = 1u << (synapse_id & 31);
            if ((word & mask) == 0u) {
                continue;
            }
            const float weight = syn_weight[synapse_id];
            const int channel = static_cast<int>(syn_effect_channel[synapse_id]);
            const float scale = syn_effect_scale != nullptr ? syn_effect_scale[synapse_id] : 1.0f;
            if (channel >= 0 &&
                channel < pending_channel_count &&
                channel < kDrainPendingChannelCapacity) {
                // append the effect on the channel
                total_channels[channel] += weight * scale;
            }
        }
        for (int channel = 0;
             channel < pending_channel_count && channel < kDrainPendingChannelCapacity;
             ++channel) {
            // add the effect on the target channel
            if (total_channels[channel] != 0.0f) {
                pending_channels[channel * pending_channel_stride + post_neuron] += total_channels[channel];
            }
        }
    }
}

__global__ void ApplyDenseLearningPreKernel(float* syn_weight,
                                            const int* syn_plastic_rule_id,
                                            const int* syn_plastic_model_id,
                                            const unsigned char* syn_plastic_flags,
                                            const unsigned int* iset_words,
                                            DenseLearningDeviceFieldTable learning_fields,
                                            DenseDeviceLearningModelFieldIndexTable learning_model_indices,
                                            DenseDeviceLearningSpikeBufferTable learning_spike_buffers,
                                            int synapse_count,
                                            int current_time_step,
                                            float dt_ms) {
    // Pre learning is driven by the same Iset bits used for propagation. Each
    // CUDA thread owns one synapse and applies the rule only when the pre spike
    // reached that synapse in this propagation step.
    for (int synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
         synapse_id < synapse_count;
         synapse_id += blockDim.x * gridDim.x) {
        // get the iset word and mask
        const unsigned int word = iset_words[synapse_id >> 5];
        const unsigned int mask = 1u << (synapse_id & 31);
        if ((word & mask) == 0u) {
            continue;
        }
        // get the rule id to see whether it is a learnable weight
        const int plastic_rule_id = syn_plastic_rule_id[synapse_id];
        if (plastic_rule_id < 0) {
            continue;
        }
        if ((syn_plastic_flags[synapse_id] & kDenseLearningUsesPre) == 0u) {
            continue;
        }
        // get the learning model id
        const int model_id = syn_plastic_model_id[synapse_id];
        if (model_id < 0 || model_id >= learning_model_indices.span_count) {
            continue;
        }
        const DenseLearningModelSpan model_span = learning_model_indices.spans[model_id];
        // get the field_id
        const int* field_ids = learning_model_indices.field_indices + model_span.field_index_offset;
        ApplyDenseLearningPreDevice(model_span.factory_model_id,
                                    learning_fields,
                                    field_ids,
                                    learning_spike_buffers,
                                    synapse_id,
                                    current_time_step,
                                    dt_ms,
                                    syn_weight);
    }
}

__global__ void ApplyDenseLearningTriggerKernel(float* syn_weight,
                                                const unsigned char* syn_type,
                                                const int* syn_trigger_rule_id,
                                                const int* syn_trigger_model_id,
                                                const unsigned char* syn_trigger_flags,
                                                const unsigned int* iset_words,
                                                DenseLearningDeviceFieldTable learning_fields,
                                                DenseDeviceLearningModelFieldIndexTable learning_model_indices,
                                                DenseDeviceLearningTriggerRouteTable learning_trigger_routes,
                                                DenseDeviceLearningSpikeBufferTable learning_spike_buffers,
                                                int synapse_count,
                                                int current_time_step,
                                                float dt_ms) {
    // Trigger learning is separate from plastic pre learning because legacy
    // TriggerRule events route updates to other plastic synapses on the target neuron.
    for (int synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
         synapse_id < synapse_count;
         synapse_id += blockDim.x * gridDim.x) {
        // find the iset mark
        const unsigned int word = iset_words[synapse_id >> 5];
        const unsigned int mask = 1u << (synapse_id & 31);
        if ((word & mask) == 0u) {
            continue;
        }
        // get the trigger rule id to see whether it is a learnable weight
        const int trigger_rule_id = syn_trigger_rule_id[synapse_id];
        if (trigger_rule_id < 0 ||
            (syn_trigger_flags[synapse_id] & kDenseLearningUsesTrigger) == 0u) {
            continue;
        }
        const int model_id = syn_trigger_model_id[synapse_id];
        if (model_id < 0 || model_id >= learning_model_indices.span_count) {
            continue;
        }
        // get the learning model id
        const DenseLearningModelSpan model_span = learning_model_indices.spans[model_id];
        const int* field_ids = learning_model_indices.field_indices + model_span.field_index_offset;
        // apply the trigger 
        ApplyDenseLearningTriggerDevice(model_span.factory_model_id,
                                        learning_fields,
                                        field_ids,
                                        learning_spike_buffers,
                                        synapse_id,
                                        syn_type[synapse_id],
                                        learning_trigger_routes,
                                        current_time_step,
                                        dt_ms,
                                        syn_weight);
    }
}

__global__ void ApplyDenseLearningPostKernel(float* syn_weight,
                                             const int* syn_post_neuron,
                                             const int* syn_plastic_rule_id,
                                             const int* syn_plastic_model_id,
                                             const unsigned char* syn_plastic_flags,
                                             const unsigned char* current_did_fire,
                                             DenseLearningDeviceFieldTable learning_fields,
                                             DenseDeviceLearningModelFieldIndexTable learning_model_indices,
                                             int synapse_count,
                                             int current_time_step,
                                             float dt_ms) {
    // Post learning stays synapse-parallel. The per-neuron did-fire array gives
    // O(1) access to the post event without building a separate firing bitmap.
    for (int synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
         synapse_id < synapse_count;
         synapse_id += blockDim.x * gridDim.x) {
        const int plastic_rule_id = syn_plastic_rule_id[synapse_id];
        if (plastic_rule_id < 0 ||
            (syn_plastic_flags[synapse_id] & kDenseLearningUsesPost) == 0u) {
            continue;
        }
        const int post_neuron = syn_post_neuron[synapse_id];
        if (post_neuron < 0 || current_did_fire[post_neuron] == 0u) {
            continue;
        }
        const int model_id = syn_plastic_model_id[synapse_id];
        if (model_id < 0 || model_id >= learning_model_indices.span_count) {
            continue;
        }
        const DenseLearningModelSpan model_span = learning_model_indices.spans[model_id];
        const int* field_ids = learning_model_indices.field_indices + model_span.field_index_offset;
        ApplyDenseLearningPostDevice(model_span.factory_model_id,
                                     learning_fields,
                                     field_ids,
                                     synapse_id,
                                     current_time_step,
                                     dt_ms,
                                     syn_weight);
    }
}

__global__ void InitializeHistoryMetadataKernel(int* time_steps, int* counts, int steps_to_keep) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < steps_to_keep) {
        time_steps[index] = -1;
        counts[index] = 0;
    }
}

__global__ void RecordCurrentFiringsToHistoryKernel(const int* current_firing_ids,
                                                    int current_firing_count,
                                                    int* history_time_steps,
                                                    int* history_counts,
                                                    int* history_ids,
                                                    int steps_to_keep,
                                                    int slot_capacity,
                                                    int current_time_step) {
    const int slot_index = current_time_step % steps_to_keep;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // write the time to history time step
        history_time_steps[slot_index] = current_time_step;
        // write the current firing count
        history_counts[slot_index] = current_firing_count;
    }
    // write the firing ids to the history ids
    const int base = slot_index * slot_capacity;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < current_firing_count;
         index += blockDim.x * gridDim.x) {
        history_ids[base + index] = current_firing_ids[index];
    }
}

}  // namespace

bool GpuPropagationRuntime::InitializeDeviceBuffers(const GpuPropagationLayout& layout, std::string* reason) {
    DestroyDeviceBuffers();
    const std::vector<std::uint8_t>* host_effect_channels = &layout.synapses.effect_channel;
    const std::vector<float>* host_effect_scales = &layout.synapses.effect_scale;
    const std::vector<int>* host_plastic_rule_ids = &layout.synapses.plastic_rule_id;
    const std::vector<int>* host_plastic_model_ids = &layout.synapses.plastic_model_id;
    const std::vector<int>* host_trigger_rule_ids = &layout.synapses.trigger_rule_id;
    const std::vector<int>* host_trigger_model_ids = &layout.synapses.trigger_model_id;
    const std::vector<unsigned char>* host_plastic_flags = &layout.synapses.plastic_flags;
    const std::vector<unsigned char>* host_trigger_flags = &layout.synapses.trigger_flags;
    const std::vector<int>* host_plastic_state_indices = &layout.synapses.plastic_state_index;
    const DenseLearningHostFieldTable* host_learning_fields = &layout.learning_fields;
    const DenseLearningModelFieldIndexTable* host_learning_model_indices = &layout.learning_model_indices;
    const DenseLearningTriggerRouteTable* host_trigger_routes = &layout.learning_trigger_routes;
    const DenseLearningSpikeBufferTable* host_spike_buffers = &layout.learning_spike_buffers;
    // a function check whether the size of the host vector is equal to the synapse count
    auto require_synapse_ints = [&](const std::vector<int>& values, const char* name) {
        if (synapse_count_ == 0) {
            return true;
        }
        if (values.size() == static_cast<std::size_t>(synapse_count_)) {
            return true;
        }
        if (reason != nullptr) {
            *reason = std::string(name) + " must be compiled for every dense synapse before GPU initialization";
        }
        return false;
    };
    // a function check whether the size of the host vector is equal to the ynapse count
    auto require_synapse_bytes = [&](const std::vector<unsigned char>& values, const char* name) {
        if (synapse_count_ == 0) {
            return true;
        }
        if (values.size() == static_cast<std::size_t>(synapse_count_)) {
            return true;
        }
        if (reason != nullptr) {
            *reason = std::string(name) + " must be compiled for every dense synapse before GPU initialization";
        }
        return false;
    };

    if (!require_synapse_ints(*host_plastic_rule_ids, "synapses.plastic_rule_id") ||
        !require_synapse_ints(*host_plastic_model_ids, "synapses.plastic_model_id") ||
        !require_synapse_bytes(*host_plastic_flags, "synapses.plastic_flags") ||
        !require_synapse_ints(*host_plastic_state_indices, "synapses.plastic_state_index") ||
        !require_synapse_ints(*host_trigger_rule_ids, "synapses.trigger_rule_id") ||
        !require_synapse_ints(*host_trigger_model_ids, "synapses.trigger_model_id") ||
        !require_synapse_bytes(*host_trigger_flags, "synapses.trigger_flags")) {
        return false;
    }
    for (int synapse_id = 0; synapse_id < synapse_count_; ++synapse_id) {
        const std::size_t index = static_cast<std::size_t>(synapse_id);
        if ((*host_plastic_rule_ids)[index] >= 0 &&
            (*host_plastic_model_ids)[index] < 0) {
            if (reason != nullptr) {
                *reason = "dense plastic synapse has a rule id but no compiled learning model id";
            }
            return false;
        }
        if ((*host_trigger_rule_ids)[index] >= 0 &&
            (*host_trigger_model_ids)[index] < 0) {
            if (reason != nullptr) {
                *reason = "dense trigger synapse has a rule id but no compiled learning model id";
            }
            return false;
        }
    }
    // to check whether there is a legal learning model in the subnetwork
    bool has_learning_model = false;
    for (int model_id : *host_plastic_model_ids) {
        if (model_id >= 0) {
            has_learning_model = true;
            break;
        }
    }
    for (int model_id : *host_trigger_model_ids) {
        if (model_id >= 0) {
            has_learning_model = true;
            break;
        }
    }
    // if there is a effective learning model, then check whether it is legal
    if (has_learning_model) {
        if (host_learning_fields->fields.empty() ||
            host_learning_model_indices->spans.empty() ||
            host_learning_model_indices->field_indices.empty()) {
            if (reason != nullptr) {
                *reason = "dense learning field tables must be finalized before GPU initialization";
            }
            return false;
        }
        if (host_trigger_routes->start.size() != static_cast<std::size_t>(synapse_count_) ||
            host_trigger_routes->count.size() != static_cast<std::size_t>(synapse_count_)) {
            if (reason != nullptr) {
                *reason = "dense learning trigger routes must be finalized before GPU initialization";
            }
            return false;
        }
        if (host_spike_buffers->synapse_bucket_id.size() != static_cast<std::size_t>(synapse_count_)) {
            if (reason != nullptr) {
                *reason = "dense learning spike-buffer bindings must be finalized before GPU initialization";
            }
            return false;
        }
    }
    if (synapse_count_ > 0 && host_effect_channels->empty()) {
        if (reason != nullptr) {
            *reason = "synapse effect channels must be compiled before GPU buffer initialization";
        }
        return false;
    }
    if (host_effect_scales->empty() && synapse_count_ > 0) {
        if (reason != nullptr) {
            *reason = "synapse effect scales must be compiled before GPU buffer initialization";
        }
        return false;
    }
    // allocate the synapse propogation structure memories on the device
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_pre_slice_synapse_ids_),
                              sizeof(int) * layout.pre_delay_slices.synapse_ids.size()),
                   "cudaMalloc(d_pre_slice_synapse_ids_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_pre_slice_start_),
                              sizeof(int) * layout.pre_delay_slices.start.size()),
                   "cudaMalloc(d_pre_slice_start_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_pre_slice_count_),
                              sizeof(int) * layout.pre_delay_slices.count.size()),
                   "cudaMalloc(d_pre_slice_count_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_weight_),
                              sizeof(float) * layout.synapses.weight.size()),
                   "cudaMalloc(d_syn_weight_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (synapse_count_ > 0) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_post_neuron_),
                                  sizeof(int) * layout.synapses.post_neuron.size()),
                       "cudaMalloc(d_syn_post_neuron_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (!host_effect_channels->empty()) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_type_),
                                  sizeof(unsigned char) * layout.synapses.type.size()),
                       "cudaMalloc(d_syn_type_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_effect_channel_),
                                  sizeof(unsigned char) * host_effect_channels->size()),
                       "cudaMalloc(d_syn_effect_channel_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (!host_effect_scales->empty()) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_effect_scale_),
                                  sizeof(float) * host_effect_scales->size()),
                       "cudaMalloc(d_syn_effect_scale_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    // copy the learning rule marks of each synapse to the device
    if (synapse_count_ > 0) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_plastic_rule_id_),
                                  sizeof(int) * host_plastic_rule_ids->size()),
                       "cudaMalloc(d_syn_plastic_rule_id_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_plastic_model_id_),
                                  sizeof(int) * host_plastic_model_ids->size()),
                       "cudaMalloc(d_syn_plastic_model_id_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_plastic_flags_),
                                  sizeof(unsigned char) * host_plastic_flags->size()),
                       "cudaMalloc(d_syn_plastic_flags_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_plastic_state_index_),
                                  sizeof(int) * host_plastic_state_indices->size()),
                       "cudaMalloc(d_syn_plastic_state_index_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_trigger_rule_id_),
                                  sizeof(int) * host_trigger_rule_ids->size()),
                       "cudaMalloc(d_syn_trigger_rule_id_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_trigger_model_id_),
                                  sizeof(int) * host_trigger_model_ids->size()),
                       "cudaMalloc(d_syn_trigger_model_id_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_syn_trigger_flags_),
                                  sizeof(unsigned char) * host_trigger_flags->size()),
                       "cudaMalloc(d_syn_trigger_flags_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    // 
    const std::size_t learning_field_span_bytes =
        sizeof(DenseDeviceFieldSpan) * host_learning_fields->fields.size();
    const std::size_t learning_float_pool_bytes =
        sizeof(float) * host_learning_fields->float_pool.size();
    const std::size_t learning_int_pool_bytes =
        sizeof(int) * host_learning_fields->int_pool.size();
    const std::size_t learning_byte_pool_bytes =
        sizeof(unsigned char) * host_learning_fields->byte_pool.size();
    const std::size_t learning_model_span_bytes =
        sizeof(DenseLearningModelSpan) * host_learning_model_indices->spans.size();
    const std::size_t learning_field_index_bytes =
        sizeof(int) * host_learning_model_indices->field_indices.size();
    //  copy the int,float,byte pool to the device
    if (has_learning_model) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_field_spans_),
                                  std::max<std::size_t>(1, learning_field_span_bytes)),
                       "cudaMalloc(d_learning_field_spans_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_float_pool_),
                                  std::max<std::size_t>(1, learning_float_pool_bytes)),
                       "cudaMalloc(d_learning_float_pool_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_int_pool_),
                                  std::max<std::size_t>(1, learning_int_pool_bytes)),
                       "cudaMalloc(d_learning_int_pool_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_byte_pool_),
                                  std::max<std::size_t>(1, learning_byte_pool_bytes)),
                       "cudaMalloc(d_learning_byte_pool_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_reset_float_pool_),
                                  std::max<std::size_t>(1, learning_float_pool_bytes)),
                       "cudaMalloc(d_learning_reset_float_pool_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_reset_int_pool_),
                                  std::max<std::size_t>(1, learning_int_pool_bytes)),
                       "cudaMalloc(d_learning_reset_int_pool_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_reset_byte_pool_),
                                  std::max<std::size_t>(1, learning_byte_pool_bytes)),
                       "cudaMalloc(d_learning_reset_byte_pool_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_model_spans_),
                                  std::max<std::size_t>(1, learning_model_span_bytes)),
                       "cudaMalloc(d_learning_model_spans_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_field_indices_),
                                  std::max<std::size_t>(1, learning_field_index_bytes)),
                       "cudaMalloc(d_learning_field_indices_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_trigger_route_start_),
                                  sizeof(int) * std::max<std::size_t>(1, host_trigger_routes->start.size())),
                       "cudaMalloc(d_learning_trigger_route_start_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_trigger_route_count_),
                                  sizeof(int) * std::max<std::size_t>(1, host_trigger_routes->count.size())),
                       "cudaMalloc(d_learning_trigger_route_count_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_trigger_route_synapse_ids_),
                                  sizeof(int) * std::max<std::size_t>(1, host_trigger_routes->synapse_ids.size())),
                       "cudaMalloc(d_learning_trigger_route_synapse_ids_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_synapse_bucket_id_),
                                  sizeof(int) * std::max<std::size_t>(1, host_spike_buffers->synapse_bucket_id.size())),
                       "cudaMalloc(d_learning_spike_buffer_synapse_bucket_id_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_bucket_start_),
                                  sizeof(int) * std::max<std::size_t>(1, host_spike_buffers->bucket_start.size())),
                       "cudaMalloc(d_learning_spike_buffer_bucket_start_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_bucket_capacity_),
                                  sizeof(int) * std::max<std::size_t>(1, host_spike_buffers->bucket_capacity.size())),
                       "cudaMalloc(d_learning_spike_buffer_bucket_capacity_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_bucket_head_),
                                  sizeof(int) * std::max<std::size_t>(1, host_spike_buffers->bucket_head.size())),
                       "cudaMalloc(d_learning_spike_buffer_bucket_head_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_bucket_count_),
                                  sizeof(int) * std::max<std::size_t>(1, host_spike_buffers->bucket_count.size())),
                       "cudaMalloc(d_learning_spike_buffer_bucket_count_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_time_),
                                  sizeof(float) * std::max<std::size_t>(1, host_spike_buffers->spike_time.size())),
                       "cudaMalloc(d_learning_spike_buffer_time_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_synapse_id_),
                                  sizeof(int) * std::max<std::size_t>(1, host_spike_buffers->spike_synapse_id.size())),
                       "cudaMalloc(d_learning_spike_buffer_synapse_id_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_learning_spike_buffer_valid_),
                                  sizeof(unsigned char) * std::max<std::size_t>(1, host_spike_buffers->valid.size())),
                       "cudaMalloc(d_learning_spike_buffer_valid_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    // Allocate the channel-major pending input pool used by model-specific neuron runtimes.
    if (!pending_.pending_channels.empty()) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_pending_channels_),
                                  sizeof(float) * pending_.pending_channels.size()),
                       "cudaMalloc(d_pending_channels_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_firing_history_time_steps_),
                              sizeof(int) * config_.steps_to_keep),
                   "cudaMalloc(d_firing_history_time_steps_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_firing_history_counts_),
                              sizeof(int) * config_.steps_to_keep),
                   "cudaMalloc(d_firing_history_counts_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    // allocate the firing history table memory
    const std::size_t firing_history_capacity =
        static_cast<std::size_t>(config_.steps_to_keep) * static_cast<std::size_t>(std::max(1, neuron_count_));
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_firing_history_ids_),
                              sizeof(int) * firing_history_capacity),
                   "cudaMalloc(d_firing_history_ids_)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    d_post_synapse_start_ = nullptr;
    d_post_synapse_count_ = nullptr;
    d_post_synapse_ids_ = nullptr;
    d_iset_words_ = nullptr;
    d_interface_target_neuron_ids_ = nullptr;
    d_interface_channel_values_ = nullptr;
    d_d2d_copy_specs_ = nullptr;
    d_d2d_copy_count_ = 0;
    device_interface_target_capacity_ = 0;
    device_interface_state_count_ = 0;
    // allocate the post synapse propogation memory
    if (!post_synapse_start_.empty()) {
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_post_synapse_start_),
                                  sizeof(int) * post_synapse_start_.size()),
                       "cudaMalloc(d_post_synapse_start_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_post_synapse_count_),
                                  sizeof(int) * post_synapse_count_.size()),
                       "cudaMalloc(d_post_synapse_count_)", reason) ||
            !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_post_synapse_ids_),
                                  sizeof(int) * post_synapse_ids_.size()),
                       "cudaMalloc(d_post_synapse_ids_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
        if (!CheckCuda(cudaMemcpy(d_post_synapse_start_,
                                  post_synapse_start_.data(),
                                  sizeof(int) * post_synapse_start_.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(post_synapse_start_)", reason) ||
            !CheckCuda(cudaMemcpy(d_post_synapse_count_,
                                  post_synapse_count_.data(),
                                  sizeof(int) * post_synapse_count_.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(post_synapse_count_)", reason) ||
            !CheckCuda(cudaMemcpy(d_post_synapse_ids_,
                                  post_synapse_ids_.data(),
                                  sizeof(int) * post_synapse_ids_.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(post_synapse_ids_)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
        // caculate the number of words needed to store the iset
        const std::size_t iset_word_count = (static_cast<std::size_t>(synapse_count_) + 31u) / 32u;
        if (iset_word_count > 0) {
            if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_iset_words_),
                                      sizeof(unsigned int) * iset_word_count),
                           "cudaMalloc(d_iset_words_)", reason)) {
                DestroyDeviceBuffers();
                return false;
            }
            device_iset_word_capacity_ = iset_word_count;
        }
        device_post_synapse_capacity_ = post_synapse_ids_.size();
    }
    // copy the synapse layout to the device
    if (!CheckCuda(cudaMemcpy(d_pre_slice_synapse_ids_,
                              layout.pre_delay_slices.synapse_ids.data(),
                              sizeof(int) * layout.pre_delay_slices.synapse_ids.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(pre_slice_synapse_ids)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMemcpy(d_pre_slice_start_,
                              layout.pre_delay_slices.start.data(),
                              sizeof(int) * layout.pre_delay_slices.start.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(pre_slice_start)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMemcpy(d_pre_slice_count_,
                              layout.pre_delay_slices.count.data(),
                              sizeof(int) * layout.pre_delay_slices.count.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(pre_slice_count)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (!CheckCuda(cudaMemcpy(d_syn_weight_,
                              layout.synapses.weight.data(),
                              sizeof(float) * layout.synapses.weight.size(),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(syn_weight)", reason)) {
        DestroyDeviceBuffers();
        return false;
    }
    if (d_syn_post_neuron_ != nullptr) {
        if (!CheckCuda(cudaMemcpy(d_syn_post_neuron_,
                                  layout.synapses.post_neuron.data(),
                                  sizeof(int) * layout.synapses.post_neuron.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_post_neuron)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (d_syn_effect_channel_ != nullptr) {
        if (!CheckCuda(cudaMemcpy(d_syn_type_,
                                  layout.synapses.type.data(),
                                  sizeof(unsigned char) * layout.synapses.type.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_type)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_effect_channel_,
                                  host_effect_channels->data(),
                                  sizeof(unsigned char) * host_effect_channels->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_effect_channel)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (d_syn_effect_scale_ != nullptr) {
        if (!CheckCuda(cudaMemcpy(d_syn_effect_scale_,
                                  host_effect_scales->data(),
                                  sizeof(float) * host_effect_scales->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_effect_scale)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (d_syn_plastic_rule_id_ != nullptr) {
        if (!CheckCuda(cudaMemcpy(d_syn_plastic_rule_id_,
                                  host_plastic_rule_ids->data(),
                                  sizeof(int) * host_plastic_rule_ids->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_plastic_rule_id)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_plastic_model_id_,
                                  host_plastic_model_ids->data(),
                                  sizeof(int) * host_plastic_model_ids->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_plastic_model_id)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_plastic_flags_,
                                  host_plastic_flags->data(),
                                  sizeof(unsigned char) * host_plastic_flags->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_plastic_flags)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_plastic_state_index_,
                                  host_plastic_state_indices->data(),
                                  sizeof(int) * host_plastic_state_indices->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_plastic_state_index)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_trigger_rule_id_,
                                  host_trigger_rule_ids->data(),
                                  sizeof(int) * host_trigger_rule_ids->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_trigger_rule_id)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_trigger_model_id_,
                                  host_trigger_model_ids->data(),
                                  sizeof(int) * host_trigger_model_ids->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_trigger_model_id)", reason) ||
            !CheckCuda(cudaMemcpy(d_syn_trigger_flags_,
                                  host_trigger_flags->data(),
                                  sizeof(unsigned char) * host_trigger_flags->size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(syn_trigger_flags)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (has_learning_model) {
        std::vector<DenseDeviceFieldSpan> device_learning_spans(host_learning_fields->fields.size());
        for (std::size_t index = 0; index < host_learning_fields->fields.size(); ++index) {
            device_learning_spans[index] = ToDeviceFieldSpan(host_learning_fields->fields[index]);
        }
        if (!CopyIfNonEmpty(d_learning_field_spans_,
                            device_learning_spans.data(),
                            learning_field_span_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_field_spans_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_float_pool_,
                            host_learning_fields->float_pool.data(),
                            learning_float_pool_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_float_pool_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_int_pool_,
                            host_learning_fields->int_pool.data(),
                            learning_int_pool_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_int_pool_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_byte_pool_,
                            host_learning_fields->byte_pool.data(),
                            learning_byte_pool_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_byte_pool_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_reset_float_pool_,
                            host_learning_fields->float_pool.data(),
                            learning_float_pool_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_reset_float_pool_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_reset_int_pool_,
                            host_learning_fields->int_pool.data(),
                            learning_int_pool_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_reset_int_pool_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_reset_byte_pool_,
                            host_learning_fields->byte_pool.data(),
                            learning_byte_pool_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_reset_byte_pool_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_model_spans_,
                            host_learning_model_indices->spans.data(),
                            learning_model_span_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_model_spans_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_field_indices_,
                            host_learning_model_indices->field_indices.data(),
                            learning_field_index_bytes,
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_field_indices_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_trigger_route_start_,
                            host_trigger_routes->start.data(),
                            sizeof(int) * host_trigger_routes->start.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_trigger_route_start_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_trigger_route_count_,
                            host_trigger_routes->count.data(),
                            sizeof(int) * host_trigger_routes->count.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_trigger_route_count_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_trigger_route_synapse_ids_,
                            host_trigger_routes->synapse_ids.data(),
                            sizeof(int) * host_trigger_routes->synapse_ids.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_trigger_route_synapse_ids_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_synapse_bucket_id_,
                            host_spike_buffers->synapse_bucket_id.data(),
                            sizeof(int) * host_spike_buffers->synapse_bucket_id.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_synapse_bucket_id_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_bucket_start_,
                            host_spike_buffers->bucket_start.data(),
                            sizeof(int) * host_spike_buffers->bucket_start.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_bucket_start_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_bucket_capacity_,
                            host_spike_buffers->bucket_capacity.data(),
                            sizeof(int) * host_spike_buffers->bucket_capacity.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_bucket_capacity_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_bucket_head_,
                            host_spike_buffers->bucket_head.data(),
                            sizeof(int) * host_spike_buffers->bucket_head.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_bucket_head_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_bucket_count_,
                            host_spike_buffers->bucket_count.data(),
                            sizeof(int) * host_spike_buffers->bucket_count.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_bucket_count_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_time_,
                            host_spike_buffers->spike_time.data(),
                            sizeof(float) * host_spike_buffers->spike_time.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_time_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_synapse_id_,
                            host_spike_buffers->spike_synapse_id.data(),
                            sizeof(int) * host_spike_buffers->spike_synapse_id.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_synapse_id_)",
                            reason) ||
            !CopyIfNonEmpty(d_learning_spike_buffer_valid_,
                            host_spike_buffers->valid.data(),
                            sizeof(unsigned char) * host_spike_buffers->valid.size(),
                            cudaMemcpyHostToDevice,
                            "cudaMemcpy(d_learning_spike_buffer_valid_)",
                            reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    if (d_pending_channels_ != nullptr) {
        if (!CheckCuda(cudaMemcpy(d_pending_channels_,
                                  pending_.pending_channels.data(),
                                  sizeof(float) * pending_.pending_channels.size(),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(pending_channels)", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }
    {
        const int threads = 128;
        const int blocks = CeilDivInt(config_.steps_to_keep, threads);
        // init the firing table count
        InitializeHistoryMetadataKernel<<<blocks, threads>>>(
            d_firing_history_time_steps_,
            d_firing_history_counts_,
            config_.steps_to_keep);
        if (!CheckCuda(cudaGetLastError(), "launch InitializeHistoryMetadataKernel", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
        if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", reason)) {
            DestroyDeviceBuffers();
            return false;
        }
    }

    d_learning_fields_.field_spans = d_learning_field_spans_;
    d_learning_fields_.field_count = static_cast<int>(host_learning_fields->fields.size());
    d_learning_fields_.float_pool = d_learning_float_pool_;
    d_learning_fields_.int_pool = d_learning_int_pool_;
    d_learning_fields_.byte_pool = d_learning_byte_pool_;
    d_learning_model_indices_.spans = d_learning_model_spans_;
    d_learning_model_indices_.span_count = static_cast<int>(host_learning_model_indices->spans.size());
    d_learning_model_indices_.field_indices = d_learning_field_indices_;
    d_learning_model_indices_.field_index_count =
        static_cast<int>(host_learning_model_indices->field_indices.size());
    d_learning_trigger_routes_.start = d_learning_trigger_route_start_;
    d_learning_trigger_routes_.count = d_learning_trigger_route_count_;
    d_learning_trigger_routes_.synapse_ids = d_learning_trigger_route_synapse_ids_;
    d_learning_trigger_routes_.synapse_count = synapse_count_;
    d_learning_spike_buffers_.synapse_bucket_id = d_learning_spike_buffer_synapse_bucket_id_;
    d_learning_spike_buffers_.bucket_start = d_learning_spike_buffer_bucket_start_;
    d_learning_spike_buffers_.bucket_capacity = d_learning_spike_buffer_bucket_capacity_;
    d_learning_spike_buffers_.bucket_head = d_learning_spike_buffer_bucket_head_;
    d_learning_spike_buffers_.bucket_count = d_learning_spike_buffer_bucket_count_;
    d_learning_spike_buffers_.spike_time = d_learning_spike_buffer_time_;
    d_learning_spike_buffers_.spike_synapse_id = d_learning_spike_buffer_synapse_id_;
    d_learning_spike_buffers_.valid = d_learning_spike_buffer_valid_;
    d_learning_spike_buffers_.bucket_count_value =
        static_cast<int>(host_spike_buffers->bucket_start.size());
    d_learning_spike_buffers_.payload_count =
        static_cast<int>(host_spike_buffers->spike_time.size());
    device_learning_float_pool_size_ = host_learning_fields->float_pool.size();
    device_learning_int_pool_size_ = host_learning_fields->int_pool.size();
    device_learning_byte_pool_size_ = host_learning_fields->byte_pool.size();

    device_ready_ = true;
    device_firing_history_capacity_ = firing_history_capacity;
    device_post_synapse_capacity_ = post_synapse_ids_.size();
    device_iset_word_capacity_ = (static_cast<std::size_t>(synapse_count_) + 31u) / 32u;
    device_interface_target_capacity_ = 0;
    device_interface_state_capacity_ = 0;
    device_interface_state_count_ = 0;
    device_d2d_copy_capacity_ = 0;
    pending_host_view_dirty_ = false;
    BuildCachedLaunchConfigs();
    return true;
}

bool GpuPropagationRuntime::ResetDeviceState(std::string* reason) {
    if (!device_ready_) {
        return true;
    }
    if (d_pending_channels_ != nullptr && !pending_.pending_channels.empty()) {
        const std::size_t pending_bytes = sizeof(float) * pending_.pending_channels.size();
        if (!CheckCuda(cudaMemset(d_pending_channels_, 0, pending_bytes),
                       "cudaMemset(d_pending_channels_)", reason)) {
            return false;
        }
    }
    if (d_firing_history_time_steps_ != nullptr && d_firing_history_counts_ != nullptr) {
        const int threads = 128;
        const int blocks = CeilDivInt(config_.steps_to_keep, threads);
        InitializeHistoryMetadataKernel<<<blocks, threads>>>(
            d_firing_history_time_steps_,
            d_firing_history_counts_,
            config_.steps_to_keep);
        if (!CheckCuda(cudaGetLastError(), "launch InitializeHistoryMetadataKernel(reset)", reason)) {
            return false;
        }
    }
    if (d_iset_words_ != nullptr && device_iset_word_capacity_ > 0) {
        if (!CheckCuda(cudaMemset(d_iset_words_,
                                  0,
                                  sizeof(unsigned int) * device_iset_word_capacity_),
                       "cudaMemset(d_iset_words_)", reason)) {
            return false;
        }
    }
    if (d_learning_float_pool_ != nullptr && device_learning_float_pool_size_ > 0) {
        if (!CheckCuda(cudaMemcpy(d_learning_float_pool_,
                                  d_learning_reset_float_pool_,
                                  sizeof(float) * device_learning_float_pool_size_,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpyDeviceToDevice(reset learning float pool)", reason)) {
            return false;
        }
    }
    if (d_learning_int_pool_ != nullptr && device_learning_int_pool_size_ > 0) {
        if (!CheckCuda(cudaMemcpy(d_learning_int_pool_,
                                  d_learning_reset_int_pool_,
                                  sizeof(int) * device_learning_int_pool_size_,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpyDeviceToDevice(reset learning int pool)", reason)) {
            return false;
        }
    }
    if (d_learning_byte_pool_ != nullptr && device_learning_byte_pool_size_ > 0) {
        if (!CheckCuda(cudaMemcpy(d_learning_byte_pool_,
                                  d_learning_reset_byte_pool_,
                                  sizeof(unsigned char) * device_learning_byte_pool_size_,
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpyDeviceToDevice(reset learning byte pool)", reason)) {
            return false;
        }
    }
    if (d_learning_spike_buffer_bucket_head_ != nullptr &&
        d_learning_spike_buffers_.bucket_count_value > 0) {
        if (!CheckCuda(cudaMemset(d_learning_spike_buffer_bucket_head_,
                                  0,
                                  sizeof(int) * d_learning_spike_buffers_.bucket_count_value),
                       "cudaMemset(d_learning_spike_buffer_bucket_head_)", reason) ||
            !CheckCuda(cudaMemset(d_learning_spike_buffer_bucket_count_,
                                  0,
                                  sizeof(int) * d_learning_spike_buffers_.bucket_count_value),
                       "cudaMemset(d_learning_spike_buffer_bucket_count_)", reason)) {
            return false;
        }
    }
    if (d_learning_spike_buffer_valid_ != nullptr &&
        d_learning_spike_buffers_.payload_count > 0) {
        if (!CheckCuda(cudaMemset(d_learning_spike_buffer_valid_,
                                  0,
                                  sizeof(unsigned char) * d_learning_spike_buffers_.payload_count),
                       "cudaMemset(d_learning_spike_buffer_valid_)", reason)) {
            return false;
        }
    }
    if (d_interface_channel_values_ != nullptr && device_interface_state_capacity_ > 0) {
        if (!CheckCuda(cudaMemset(d_interface_channel_values_,
                                  0,
                                  sizeof(InterfaceChannelValue) * device_interface_state_capacity_),
                       "cudaMemset(d_interface_channel_values_)", reason)) {
            return false;
        }
    }
    device_interface_state_count_ = 0;
    d_d2d_copy_count_ = 0;
    pending_host_view_dirty_ = false;
    return CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(reset propagation state)", reason);
}

void GpuPropagationRuntime::DestroyDeviceBuffers() {
    if (d_iset_words_ != nullptr) {
        cudaFree(d_iset_words_);
        d_iset_words_ = nullptr;
    }
    if (d_post_synapse_ids_ != nullptr) {
        cudaFree(d_post_synapse_ids_);
        d_post_synapse_ids_ = nullptr;
    }
    if (d_post_synapse_count_ != nullptr) {
        cudaFree(d_post_synapse_count_);
        d_post_synapse_count_ = nullptr;
    }
    if (d_post_synapse_start_ != nullptr) {
        cudaFree(d_post_synapse_start_);
        d_post_synapse_start_ = nullptr;
    }
    if (d_interface_channel_values_ != nullptr) {
        cudaFree(d_interface_channel_values_);
        d_interface_channel_values_ = nullptr;
    }
    if (d_d2d_copy_specs_ != nullptr) {
        cudaFree(d_d2d_copy_specs_);
        d_d2d_copy_specs_ = nullptr;
    }
    if (d_interface_target_neuron_ids_ != nullptr) {
        cudaFree(d_interface_target_neuron_ids_);
        d_interface_target_neuron_ids_ = nullptr;
    }
    if (d_firing_history_ids_ != nullptr) {
        cudaFree(d_firing_history_ids_);
        d_firing_history_ids_ = nullptr;
    }
    if (d_firing_history_counts_ != nullptr) {
        cudaFree(d_firing_history_counts_);
        d_firing_history_counts_ = nullptr;
    }
    if (d_firing_history_time_steps_ != nullptr) {
        cudaFree(d_firing_history_time_steps_);
        d_firing_history_time_steps_ = nullptr;
    }
    if (d_pending_channels_ != nullptr) {
        cudaFree(d_pending_channels_);
        d_pending_channels_ = nullptr;
    }
    if (d_learning_spike_buffer_valid_ != nullptr) {
        cudaFree(d_learning_spike_buffer_valid_);
        d_learning_spike_buffer_valid_ = nullptr;
    }
    if (d_learning_spike_buffer_synapse_id_ != nullptr) {
        cudaFree(d_learning_spike_buffer_synapse_id_);
        d_learning_spike_buffer_synapse_id_ = nullptr;
    }
    if (d_learning_spike_buffer_time_ != nullptr) {
        cudaFree(d_learning_spike_buffer_time_);
        d_learning_spike_buffer_time_ = nullptr;
    }
    if (d_learning_spike_buffer_bucket_count_ != nullptr) {
        cudaFree(d_learning_spike_buffer_bucket_count_);
        d_learning_spike_buffer_bucket_count_ = nullptr;
    }
    if (d_learning_spike_buffer_bucket_head_ != nullptr) {
        cudaFree(d_learning_spike_buffer_bucket_head_);
        d_learning_spike_buffer_bucket_head_ = nullptr;
    }
    if (d_learning_spike_buffer_bucket_capacity_ != nullptr) {
        cudaFree(d_learning_spike_buffer_bucket_capacity_);
        d_learning_spike_buffer_bucket_capacity_ = nullptr;
    }
    if (d_learning_spike_buffer_bucket_start_ != nullptr) {
        cudaFree(d_learning_spike_buffer_bucket_start_);
        d_learning_spike_buffer_bucket_start_ = nullptr;
    }
    if (d_learning_spike_buffer_synapse_bucket_id_ != nullptr) {
        cudaFree(d_learning_spike_buffer_synapse_bucket_id_);
        d_learning_spike_buffer_synapse_bucket_id_ = nullptr;
    }
    if (d_syn_trigger_flags_ != nullptr) {
        cudaFree(d_syn_trigger_flags_);
        d_syn_trigger_flags_ = nullptr;
    }
    if (d_syn_trigger_model_id_ != nullptr) {
        cudaFree(d_syn_trigger_model_id_);
        d_syn_trigger_model_id_ = nullptr;
    }
    if (d_syn_trigger_rule_id_ != nullptr) {
        cudaFree(d_syn_trigger_rule_id_);
        d_syn_trigger_rule_id_ = nullptr;
    }
    if (d_syn_plastic_state_index_ != nullptr) {
        cudaFree(d_syn_plastic_state_index_);
        d_syn_plastic_state_index_ = nullptr;
    }
    if (d_syn_plastic_flags_ != nullptr) {
        cudaFree(d_syn_plastic_flags_);
        d_syn_plastic_flags_ = nullptr;
    }
    if (d_syn_plastic_rule_id_ != nullptr) {
        cudaFree(d_syn_plastic_rule_id_);
        d_syn_plastic_rule_id_ = nullptr;
    }
    if (d_syn_plastic_model_id_ != nullptr) {
        cudaFree(d_syn_plastic_model_id_);
        d_syn_plastic_model_id_ = nullptr;
    }
    if (d_learning_field_indices_ != nullptr) {
        cudaFree(d_learning_field_indices_);
        d_learning_field_indices_ = nullptr;
    }
    if (d_learning_trigger_route_synapse_ids_ != nullptr) {
        cudaFree(d_learning_trigger_route_synapse_ids_);
        d_learning_trigger_route_synapse_ids_ = nullptr;
    }
    if (d_learning_trigger_route_count_ != nullptr) {
        cudaFree(d_learning_trigger_route_count_);
        d_learning_trigger_route_count_ = nullptr;
    }
    if (d_learning_trigger_route_start_ != nullptr) {
        cudaFree(d_learning_trigger_route_start_);
        d_learning_trigger_route_start_ = nullptr;
    }
    if (d_learning_model_spans_ != nullptr) {
        cudaFree(d_learning_model_spans_);
        d_learning_model_spans_ = nullptr;
    }
    if (d_learning_reset_byte_pool_ != nullptr) {
        cudaFree(d_learning_reset_byte_pool_);
        d_learning_reset_byte_pool_ = nullptr;
    }
    if (d_learning_reset_int_pool_ != nullptr) {
        cudaFree(d_learning_reset_int_pool_);
        d_learning_reset_int_pool_ = nullptr;
    }
    if (d_learning_reset_float_pool_ != nullptr) {
        cudaFree(d_learning_reset_float_pool_);
        d_learning_reset_float_pool_ = nullptr;
    }
    if (d_learning_byte_pool_ != nullptr) {
        cudaFree(d_learning_byte_pool_);
        d_learning_byte_pool_ = nullptr;
    }
    if (d_learning_int_pool_ != nullptr) {
        cudaFree(d_learning_int_pool_);
        d_learning_int_pool_ = nullptr;
    }
    if (d_learning_float_pool_ != nullptr) {
        cudaFree(d_learning_float_pool_);
        d_learning_float_pool_ = nullptr;
    }
    if (d_learning_field_spans_ != nullptr) {
        cudaFree(d_learning_field_spans_);
        d_learning_field_spans_ = nullptr;
    }
    if (d_syn_effect_scale_ != nullptr) {
        cudaFree(d_syn_effect_scale_);
        d_syn_effect_scale_ = nullptr;
    }
    if (d_syn_effect_channel_ != nullptr) {
        cudaFree(d_syn_effect_channel_);
        d_syn_effect_channel_ = nullptr;
    }
    if (d_syn_type_ != nullptr) {
        cudaFree(d_syn_type_);
        d_syn_type_ = nullptr;
    }
    if (d_syn_weight_ != nullptr) {
        cudaFree(d_syn_weight_);
        d_syn_weight_ = nullptr;
    }
    if (d_syn_post_neuron_ != nullptr) {
        cudaFree(d_syn_post_neuron_);
        d_syn_post_neuron_ = nullptr;
    }
    if (d_pre_slice_synapse_ids_ != nullptr) {
        cudaFree(d_pre_slice_synapse_ids_);
        d_pre_slice_synapse_ids_ = nullptr;
    }
    if (d_pre_slice_count_ != nullptr) {
        cudaFree(d_pre_slice_count_);
        d_pre_slice_count_ = nullptr;
    }
    if (d_pre_slice_start_ != nullptr) {
        cudaFree(d_pre_slice_start_);
        d_pre_slice_start_ = nullptr;
    }
    device_ready_ = false;
    device_firing_history_capacity_ = 0;
    device_post_synapse_capacity_ = 0;
    device_iset_word_capacity_ = 0;
    device_interface_target_capacity_ = 0;
    device_interface_state_capacity_ = 0;
    device_interface_state_count_ = 0;
    device_d2d_copy_capacity_ = 0;
    device_learning_float_pool_size_ = 0;
    device_learning_int_pool_size_ = 0;
    device_learning_byte_pool_size_ = 0;
    d_learning_fields_ = DenseLearningDeviceFieldTable{};
    d_learning_model_indices_ = DenseDeviceLearningModelFieldIndexTable{};
    d_learning_trigger_routes_ = DenseDeviceLearningTriggerRouteTable{};
    d_learning_spike_buffers_ = DenseDeviceLearningSpikeBufferTable{};
    d_d2d_copy_count_ = 0;
    device_sm_count_ = 1;
    linear_scan_max_blocks_ = 1;
    history_scan_max_blocks_ = 1;
    launch_clear_iset_ = LaunchConfig1D{};
    launch_drain_iset_ = LaunchConfig1D{};
    launch_learning_synapse_ = LaunchConfig1D{};
    launch_post_learning_ = LaunchConfig1D{};
    history_blocks_x_ = 1;
    has_dense_learning_ = false;
    has_dense_pre_learning_ = false;
    has_dense_post_learning_ = false;
    has_dense_trigger_learning_ = false;
    pending_host_view_dirty_ = false;
}

bool GpuPropagationRuntime::UploadInterfaceChannelValuesGpu(const std::vector<InterfaceChannelValue>& values,
                                                            std::string* reason) {
    const int interface_count = static_cast<int>(values.size());
    if (interface_count == 0) {
        device_interface_state_count_ = 0;
        return true;
    }
    if (!interface_target_neuron_ids_.empty() &&
        interface_count != static_cast<int>(interface_target_neuron_ids_.size())) {
        if (reason != nullptr) {
            *reason = "host interface record count must match interface target mapping count";
        }
        return false;
    }
    // this is construct only once
    if (static_cast<std::size_t>(interface_count) > device_interface_state_capacity_) {
        if (d_interface_channel_values_ != nullptr) {
            cudaFree(d_interface_channel_values_);
            d_interface_channel_values_ = nullptr;
            device_interface_state_capacity_ = 0;
        }
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_interface_channel_values_),
                                  sizeof(InterfaceChannelValue) * interface_count),
                       "cudaMalloc(d_interface_channel_values_)", reason)) {
            return false;
        }
        device_interface_state_capacity_ = static_cast<std::size_t>(interface_count);
    }
    if (!CheckCuda(cudaMemcpy(d_interface_channel_values_,
                              values.data(),
                              sizeof(InterfaceChannelValue) * interface_count,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(interface channel value records)", reason)) {
        return false;
    }
    device_interface_state_count_ = interface_count;
    return true;
}

bool GpuPropagationRuntime::UploadInterfaceChannelValuesDeviceGpu(
    const float* device_values,
    const std::vector<DeviceToDeviceInterfaceCopySpec>& specs,
    std::string* reason) {
    if (device_values == nullptr) {
        if (reason != nullptr) {
            *reason = "device_values must not be null";
        }
        return false;
    }
    const int copy_count = static_cast<int>(specs.size());
    if (copy_count == 0) {
        d_d2d_copy_count_ = 0;
        return true;
    }
    if (d_pending_channels_ == nullptr) {
        if (reason != nullptr) {
            *reason = "device interface channel copy requires pending channel buffers";
        }
        return false;
    }
    if (static_cast<std::size_t>(copy_count) > device_d2d_copy_capacity_) {
        if (d_d2d_copy_specs_ != nullptr) {
            cudaFree(d_d2d_copy_specs_);
            d_d2d_copy_specs_ = nullptr;
            device_d2d_copy_capacity_ = 0;
        }
        if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_d2d_copy_specs_),
                                  sizeof(DeviceToDeviceInterfaceCopySpec) * copy_count),
                       "cudaMalloc(d_d2d_copy_specs_)", reason)) {
            return false;
        }
        device_d2d_copy_capacity_ = static_cast<std::size_t>(copy_count);
    }
    if (!CheckCuda(cudaMemcpy(d_d2d_copy_specs_,
                              specs.data(),
                              sizeof(DeviceToDeviceInterfaceCopySpec) * copy_count,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(d2d copy spec records)", reason)) {
        return false;
    }

    const LaunchConfig1D launch =
        MakeRuntimeLaunchConfig(copy_count, 256, linear_scan_max_blocks_);
    ApplyDeviceInterfaceChannelValuesKernel<<<launch.blocks, launch.threads>>>(
        d_pending_channels_,
        pending_channel_count_,
        pending_channel_stride_,
        device_values,
        d_d2d_copy_specs_,
        copy_count,
        d_interface_target_neuron_ids_,
        static_cast<int>(interface_target_neuron_ids_.size()));
    if (!CheckCuda(cudaGetLastError(), "launch ApplyDeviceInterfaceChannelValuesKernel", reason) ||
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(device interface channel values)", reason)) {
        return false;
    }
    d_d2d_copy_count_ = copy_count;
    pending_host_view_dirty_ = true;
    return true;
}

bool GpuPropagationRuntime::ApplyInterfaceStateVectorsGpu(std::string* reason) {
    const int interface_count = device_interface_state_count_;
    if (interface_count <= 0) {
        return true;
    }
    const LaunchConfig1D launch =
        MakeRuntimeLaunchConfig(interface_count, 256, linear_scan_max_blocks_);
    ApplyBoundaryStateVectorKernel<<<launch.blocks, launch.threads>>>(
        d_pending_channels_,
        pending_channel_count_,
        pending_channel_stride_,
        d_interface_channel_values_,
        interface_count,
        d_interface_target_neuron_ids_,
        static_cast<int>(interface_target_neuron_ids_.size()));
    if (!CheckCuda(cudaGetLastError(), "launch ApplyBoundaryStateVectorKernel", reason) ||
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(apply interface state vectors)", reason)) {
        return false;
    }
    pending_host_view_dirty_ = true;
    return true;
}

bool GpuPropagationRuntime::DownloadSynapticWeightsGpu(std::vector<float>* weights,
                                                       std::string* reason) const {
    if (weights == nullptr) {
        if (reason != nullptr) {
            *reason = "weight output vector must not be null";
        }
        return false;
    }
    weights->clear();
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "CUDA device buffers are not initialized";
        }
        return false;
    }
    if (synapse_count_ <= 0) {
        return true;
    }
    if (d_syn_weight_ == nullptr) {
        if (reason != nullptr) {
            *reason = "dense synaptic weight device buffer is not allocated";
        }
        return false;
    }
    weights->resize(static_cast<std::size_t>(synapse_count_));
    return CheckCuda(cudaMemcpy(weights->data(),
                                d_syn_weight_,
                                sizeof(float) * weights->size(),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy(download dense synaptic weights)",
                     reason);
}

bool GpuPropagationRuntime::UploadSynapticWeightsGpu(const std::vector<float>& weights,
                                                     std::string* reason) {
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "CUDA device buffers are not initialized";
        }
        return false;
    }
    if (weights.size() != static_cast<std::size_t>(synapse_count_)) {
        if (reason != nullptr) {
            *reason = "dense synaptic weight count does not match runtime synapse_count";
        }
        return false;
    }
    if (synapse_count_ <= 0) {
        return true;
    }
    if (d_syn_weight_ == nullptr) {
        if (reason != nullptr) {
            *reason = "dense synaptic weight device buffer is not allocated";
        }
        return false;
    }
    return CheckCuda(cudaMemcpy(d_syn_weight_,
                                weights.data(),
                                sizeof(float) * weights.size(),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(upload dense synaptic weights)",
                     reason);
}

bool GpuPropagationRuntime::GetSynapseWeightGpu(int synapse_index,
                                                float* weight,
                                                std::string* reason) const {
    if (weight == nullptr) {
        if (reason != nullptr) {
            *reason = "weight output pointer must not be null";
        }
        return false;
    }
    if (d_syn_weight_ == nullptr) {
        if (reason != nullptr) {
            *reason = "dense synaptic weight device buffer is not allocated";
        }
        return false;
    }
    return CheckCuda(cudaMemcpy(weight,
                                d_syn_weight_ + synapse_index,
                                sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy(get dense synaptic weight)",
                     reason);
}

bool GpuPropagationRuntime::SetSynapseWeightGpu(int synapse_index,
                                                float weight,
                                                std::string* reason) {
    if (d_syn_weight_ == nullptr) {
        if (reason != nullptr) {
            *reason = "dense synaptic weight device buffer is not allocated";
        }
        return false;
    }
    return CheckCuda(cudaMemcpy(d_syn_weight_ + synapse_index,
                                &weight,
                                sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(set dense synaptic weight)",
                     reason);
}

LaunchConfig1D GpuPropagationRuntime::MakeRuntimeLaunchConfig(int item_count,
                                                              int threads,
                                                              int max_blocks) const {
    return MakeLaunchConfig1DWithMaxBlocks(item_count, threads, max_blocks);
}

void GpuPropagationRuntime::BuildCachedLaunchConfigs() {
    device_sm_count_ = QuerySmCount();
    linear_scan_max_blocks_ =
        MaxBlocksFromSmCount(device_sm_count_, kLinearScanBlocksPerSm);
    history_scan_max_blocks_ =
        MaxBlocksFromSmCount(device_sm_count_, kHistoryScanBlocksPerSm);
    const int word_count = static_cast<int>(device_iset_word_capacity_);
    launch_clear_iset_ =
        MakeRuntimeLaunchConfig(word_count, 256, linear_scan_max_blocks_);
    launch_drain_iset_ =
        MakeRuntimeLaunchConfig(neuron_count_, 256, linear_scan_max_blocks_);
    launch_learning_synapse_ =
        MakeRuntimeLaunchConfig(synapse_count_, 256, linear_scan_max_blocks_);
    launch_post_learning_ =
        MakeRuntimeLaunchConfig(synapse_count_, 256, linear_scan_max_blocks_);
    const int firing_capacity = std::max(1, neuron_count_);
    history_blocks_x_ =
        MakeCappedGridXWithMaxBlocks(
            std::max(1, CeilDivInt(firing_capacity, 64)),
            history_scan_max_blocks_);
}

bool GpuPropagationRuntime::RunGpuIsetPropagation(std::string* reason) {
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "CUDA device buffers are not initialized";
        }
        return false;
    }
    if (synapse_count_ <= 0 || neuron_count_ <= 0) {
        ClearLearning();
        profiling_.slice_propagation_ms = 0.0;
        profiling_.pending_flush_ms = 0.0;
        profiling_.learning_apply_ms = 0.0;
        return true;
    }
    if (d_iset_words_ == nullptr ||
        d_firing_history_time_steps_ == nullptr ||
        d_firing_history_counts_ == nullptr ||
        d_firing_history_ids_ == nullptr ||
        d_pre_slice_synapse_ids_ == nullptr ||
        d_pre_slice_start_ == nullptr ||
        d_pre_slice_count_ == nullptr ||
        d_post_synapse_start_ == nullptr ||
        d_post_synapse_count_ == nullptr ||
        d_post_synapse_ids_ == nullptr ||
        d_syn_type_ == nullptr ||
        d_syn_plastic_rule_id_ == nullptr ||
        d_syn_plastic_model_id_ == nullptr ||
        d_syn_plastic_flags_ == nullptr ||
        d_syn_trigger_rule_id_ == nullptr ||
        d_syn_trigger_model_id_ == nullptr ||
        d_syn_trigger_flags_ == nullptr ||
        d_pending_channels_ == nullptr) {
        if (reason != nullptr) {
            *reason = "unified GPU Iset propagation buffers are not initialized";
        }
        return false;
    }

    ClearLearning();

    const auto start = std::chrono::steady_clock::now();

    const int word_count = static_cast<int>(device_iset_word_capacity_);
    if (word_count > 0) {
        ClearIsetWordsKernel<<<launch_clear_iset_.blocks, launch_clear_iset_.threads>>>(
            d_iset_words_,
            word_count);
        if (!CheckCuda(cudaGetLastError(), "launch ClearIsetWordsKernel(history scan)", reason)) {
            return false;
        }
    }

    const int firing_capacity = std::max(1, neuron_count_);
    const int history_threads = 128;
    // Each Y-grid row scans one delay slot directly from the GPU firing history ring.
    dim3 history_grid(static_cast<unsigned int>(history_blocks_x_),
                      static_cast<unsigned int>(std::max(1, delay_slot_count_)),
                      1u);
    MarkIsetFromFiringHistorySharedKernel<<<history_grid, history_threads>>>(
        d_firing_history_time_steps_,
        d_firing_history_counts_,
        d_firing_history_ids_,
        config_.steps_to_keep,
        firing_capacity,
        d_pre_slice_start_,
        d_pre_slice_count_,
        delay_slot_count_,
        current_time_step_,
        d_pre_slice_synapse_ids_,
        d_iset_words_);
    if (!CheckCuda(cudaGetLastError(), "launch MarkIsetFromFiringHistorySharedKernel", reason)) {
        return false;
    }

    DrainIsetToPendingKernel<<<launch_drain_iset_.blocks, launch_drain_iset_.threads>>>(
        d_pending_channels_,
        pending_channel_count_,
        pending_channel_stride_,
        d_post_synapse_start_,
        d_post_synapse_count_,
        d_post_synapse_ids_,
        d_syn_weight_,
        d_syn_effect_channel_,
        d_syn_effect_scale_,
        d_iset_words_,
        neuron_count_);
    if (!CheckCuda(cudaGetLastError(), "launch DrainIsetToPendingKernel(history scan)", reason)) {
        return false;
    }

    // Keep the learning hook adjacent to propagation while preserving synapse
    // parallelism instead of scanning connections serially inside the drain.
    if (has_dense_pre_learning_) {
        ApplyDenseLearningPreKernel<<<launch_learning_synapse_.blocks, launch_learning_synapse_.threads>>>(
            d_syn_weight_,
            d_syn_plastic_rule_id_,
            d_syn_plastic_model_id_,
            d_syn_plastic_flags_,
            d_iset_words_,
            d_learning_fields_,
            d_learning_model_indices_,
            d_learning_spike_buffers_,
            synapse_count_,
            current_time_step_,
            config_.dt_ms);
        if (!CheckCuda(cudaGetLastError(), "launch ApplyDenseLearningPreKernel", reason)) {
            return false;
        }
    }
    if (has_dense_trigger_learning_) {
        ApplyDenseLearningTriggerKernel<<<launch_learning_synapse_.blocks, launch_learning_synapse_.threads>>>(
            d_syn_weight_,
            d_syn_type_,
            d_syn_trigger_rule_id_,
            d_syn_trigger_model_id_,
            d_syn_trigger_flags_,
            d_iset_words_,
            d_learning_fields_,
            d_learning_model_indices_,
            d_learning_trigger_routes_,
            d_learning_spike_buffers_,
            synapse_count_,
            current_time_step_,
            config_.dt_ms);
        if (!CheckCuda(cudaGetLastError(), "launch ApplyDenseLearningTriggerKernel", reason)) {
            return false;
        }
    }

    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", reason)) {
        return false;
    }

    profiling_.slice_propagation_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    profiling_.pending_flush_ms = 0.0;
    profiling_.learning_apply_ms = 0.0;

    pending_host_view_dirty_ = true;
    return true;
}

bool GpuPropagationRuntime::ApplyPostLearningFromDeviceFiringsGpu(const unsigned char* d_current_did_fire,
                                                                  std::string* reason) {
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "CUDA device buffers are not initialized";
        }
        return false;
    }
    if (synapse_count_ <= 0) {
        return true;
    }
    if (d_current_did_fire == nullptr ||
        d_syn_post_neuron_ == nullptr ||
        d_syn_weight_ == nullptr ||
        d_syn_plastic_rule_id_ == nullptr ||
        d_syn_plastic_model_id_ == nullptr ||
        d_syn_plastic_flags_ == nullptr) {
        if (reason != nullptr) {
            *reason = "dense post learning buffers are not initialized";
        }
        return false;
    }

    const auto start = std::chrono::steady_clock::now();
    // The neuron runtime already wrote one did-fire byte per neuron; post
    // learning can therefore scan synapses directly and test their post neuron.
    ApplyDenseLearningPostKernel<<<launch_post_learning_.blocks, launch_post_learning_.threads>>>(
        d_syn_weight_,
        d_syn_post_neuron_,
        d_syn_plastic_rule_id_,
        d_syn_plastic_model_id_,
        d_syn_plastic_flags_,
        d_current_did_fire,
        d_learning_fields_,
        d_learning_model_indices_,
        synapse_count_,
        current_time_step_,
        config_.dt_ms);
    if (!CheckCuda(cudaGetLastError(), "launch ApplyDenseLearningPostKernel", reason) ||
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(dense post learning)", reason)) {
        return false;
    }
    profiling_.learning_apply_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return true;
}

bool GpuPropagationRuntime::CommitCurrentDeviceFiringsFromExternal(int time_step,
                                                                   const int* d_external_firing_ids,
                                                                   const int* d_external_firing_count,
                                                                   std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
    if (!device_ready_) {
        if (reason != nullptr) {
            *reason = "external device firing commit requires the GPU CARLsim-like path";
        }
        return false;
    }
    if (d_external_firing_ids == nullptr || d_external_firing_count == nullptr) {
        if (reason != nullptr) {
            *reason = "external device firing buffers must not be null";
        }
        return false;
    }
    const int firing_capacity = std::max(1, neuron_count_);
    // Copy only the count to host for launch sizing and validation. The firing
    // ids stay in the producer-owned device buffer and are read directly by the
    // history-recording kernel below.
    int current_firing_count = 0;
    if (!CheckCuda(cudaMemcpy(&current_firing_count,
                              d_external_firing_count,
                              sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(external firing count -> host)", reason)) {
        return false;
    }
    if (current_firing_count < 0 || current_firing_count > firing_capacity) {
        if (reason != nullptr) {
            *reason = "external firing count is out of range";
        }
        return false;
    }
    // if there are some firing neurons in current steps
    if (current_firing_count > 0) {
        const LaunchConfig1D launch =
            MakeRuntimeLaunchConfig(current_firing_count, 256, linear_scan_max_blocks_);
        RecordCurrentFiringsToHistoryKernel<<<launch.blocks, launch.threads>>>(
            d_external_firing_ids,
            current_firing_count,
            d_firing_history_time_steps_,
            d_firing_history_counts_,
            d_firing_history_ids_,
            config_.steps_to_keep,
            firing_capacity,
            time_step);
    } else {
        RecordCurrentFiringsToHistoryKernel<<<1, 1>>>(
            d_external_firing_ids,
            0,
            d_firing_history_time_steps_,
            d_firing_history_counts_,
            d_firing_history_ids_,
            config_.steps_to_keep,
            firing_capacity,
            time_step);
    }
    if (!CheckCuda(cudaGetLastError(), "launch RecordCurrentFiringsToHistoryKernel(external)", reason) ||
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(record external firings)", reason)) {
        return false;
    }

    return true;
}

}  // namespace npgr
