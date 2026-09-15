#include "gpu_runtime/GpuPropagationRuntime.h"

#include "dense_subnetwork/learning/DenseLearningRuleFactory.h"

#include <algorithm>
#include <chrono>
#include <sstream>

#if NPGR_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace npgr {

GpuPropagationRuntime::GpuPropagationRuntime()
    : initialized_(false),
      current_time_step_(0),
      neuron_count_(0),
      synapse_count_(0),
      delay_slot_count_(0),
      device_ready_(false),
      device_sm_count_(1),
      linear_scan_max_blocks_(1),
      history_scan_max_blocks_(1),
      launch_clear_iset_(),
      launch_drain_iset_(),
      launch_learning_synapse_(),
      launch_post_learning_(),
      history_blocks_x_(1),
      has_dense_learning_(false),
      has_dense_pre_learning_(false),
      has_dense_post_learning_(false),
      has_dense_trigger_learning_(false),
      d_pre_slice_synapse_ids_(nullptr),
      d_pre_slice_start_(nullptr),
      d_pre_slice_count_(nullptr),
      d_syn_post_neuron_(nullptr),
      d_syn_weight_(nullptr),
      d_syn_type_(nullptr),
      d_syn_effect_channel_(nullptr),
      d_syn_effect_scale_(nullptr),
      d_syn_plastic_rule_id_(nullptr),
      d_syn_plastic_model_id_(nullptr),
      d_syn_plastic_flags_(nullptr),
      d_syn_plastic_state_index_(nullptr),
      d_syn_trigger_rule_id_(nullptr),
      d_syn_trigger_model_id_(nullptr),
      d_syn_trigger_flags_(nullptr),
      d_learning_field_spans_(nullptr),
      d_learning_float_pool_(nullptr),
      d_learning_int_pool_(nullptr),
      d_learning_byte_pool_(nullptr),
      d_learning_reset_float_pool_(nullptr),
      d_learning_reset_int_pool_(nullptr),
      d_learning_reset_byte_pool_(nullptr),
      d_learning_fields_(),
      d_learning_model_spans_(nullptr),
      d_learning_field_indices_(nullptr),
      d_learning_model_indices_(),
      d_learning_trigger_routes_(),
      d_learning_spike_buffers_(),
      d_learning_trigger_route_start_(nullptr),
      d_learning_trigger_route_count_(nullptr),
      d_learning_trigger_route_synapse_ids_(nullptr),
      d_learning_spike_buffer_synapse_bucket_id_(nullptr),
      d_learning_spike_buffer_bucket_start_(nullptr),
      d_learning_spike_buffer_bucket_capacity_(nullptr),
      d_learning_spike_buffer_bucket_head_(nullptr),
      d_learning_spike_buffer_bucket_count_(nullptr),
      d_learning_spike_buffer_time_(nullptr),
      d_learning_spike_buffer_synapse_id_(nullptr),
      d_learning_spike_buffer_valid_(nullptr),
      device_learning_float_pool_size_(0),
      device_learning_int_pool_size_(0),
      device_learning_byte_pool_size_(0),
      d_pending_channels_(nullptr),
      pending_channel_count_(0),
      pending_channel_stride_(0),
      d_firing_history_time_steps_(nullptr),
      d_firing_history_counts_(nullptr),
      d_firing_history_ids_(nullptr),
      d_post_synapse_start_(nullptr),
      d_post_synapse_count_(nullptr),
      d_post_synapse_ids_(nullptr),
      d_iset_words_(nullptr),
      d_interface_target_neuron_ids_(nullptr),
      d_interface_channel_values_(nullptr),
      d_d2d_copy_specs_(nullptr),
      d_d2d_copy_count_(0),
      device_firing_history_capacity_(0),
      device_post_synapse_capacity_(0),
      device_iset_word_capacity_(0),
      device_interface_target_capacity_(0),
      device_interface_state_capacity_(0),
      device_interface_state_count_(0),
      device_d2d_copy_capacity_(0),
      pending_host_view_dirty_(false) {}

GpuPropagationRuntime::~GpuPropagationRuntime() {
    DestroyDeviceBuffers();
}

bool GpuPropagationRuntime::Initialize(const GpuPropagationLayout& layout, const RuntimeConfig& config, std::string* reason) {
    std::string validation_reason;
    if (!layout.IsValid(&validation_reason)) {
        if (reason != nullptr) {
            *reason = validation_reason;
        }
        return false;
    }
    if (layout.stats.delay_slot_count <= 0) {
        if (reason != nullptr) {
            *reason = "layout delay_slot_count must be positive";
        }
        return false;
    }
    if (config.steps_to_keep < layout.stats.delay_slot_count + 1) {
        if (reason != nullptr) {
            *reason = "steps_to_keep must be at least layout delay_slot_count + 1";
        }
        return false;
    }

    config_ = config;
    profiling_ = ProfilingStats{};
    neuron_count_ = layout.stats.neuron_count;
    synapse_count_ = layout.stats.synapse_count;
    delay_slot_count_ = layout.stats.delay_slot_count;
    pending_channel_count_ = std::max(3, layout.stats.pending_channel_count);
    pending_channel_stride_ = neuron_count_;
    pending_.pending_channels.assign(
        static_cast<std::size_t>(pending_channel_count_) *
            static_cast<std::size_t>(pending_channel_stride_),
        0.0f);
    learning_.syn_pending_dwt.assign(synapse_count_, 0.0f);
    // build the post synapse structure
    post_synapse_start_.assign(static_cast<std::size_t>(neuron_count_), 0);
    post_synapse_count_.assign(static_cast<std::size_t>(neuron_count_), 0);
    post_synapse_ids_.clear();
    // generate the mapping from post neurons to their incoming synapses
    if (neuron_count_ > 0 && synapse_count_ > 0) {
        std::vector<std::vector<int> > incoming(static_cast<std::size_t>(neuron_count_));
        for (int synapse_id = 0; synapse_id < synapse_count_; ++synapse_id) {
            const int post_neuron = layout.synapses.post_neuron[static_cast<std::size_t>(synapse_id)];
            if (post_neuron >= 0 && post_neuron < neuron_count_) {
                incoming[static_cast<std::size_t>(post_neuron)].push_back(synapse_id);
            }
        }
        int running_start = 0;
        // generate the CSR term
        for (int neuron_id = 0; neuron_id < neuron_count_; ++neuron_id) {
            post_synapse_start_[static_cast<std::size_t>(neuron_id)] = running_start;
            post_synapse_count_[static_cast<std::size_t>(neuron_id)] =
                static_cast<int>(incoming[static_cast<std::size_t>(neuron_id)].size());
            post_synapse_ids_.insert(post_synapse_ids_.end(),
                                     incoming[static_cast<std::size_t>(neuron_id)].begin(),
                                     incoming[static_cast<std::size_t>(neuron_id)].end());
            running_start += post_synapse_count_[static_cast<std::size_t>(neuron_id)];
        }
    }
    current_time_step_ = 0;
    pending_host_view_dirty_ = false;
    initialized_ = true;
#if NPGR_ENABLE_CUDA
    if (!InitializeDeviceBuffers(layout, reason)) {
        initialized_ = false;
        return false;
    }
#endif
    BuildLearningFeatureFlags(layout);
    return true;
}

void GpuPropagationRuntime::BuildLearningFeatureFlags(const GpuPropagationLayout& layout) {
    has_dense_pre_learning_ = false;
    has_dense_post_learning_ = false;
    has_dense_trigger_learning_ = false;
    const int count = layout.stats.synapse_count;
    for (int synapse_id = 0; synapse_id < count; ++synapse_id) {
        const std::size_t index = static_cast<std::size_t>(synapse_id);
        if (index < layout.synapses.plastic_rule_id.size() &&
            index < layout.synapses.plastic_flags.size() &&
            layout.synapses.plastic_rule_id[index] >= 0) {
            const unsigned char flags = layout.synapses.plastic_flags[index];
            has_dense_pre_learning_ =
                has_dense_pre_learning_ || ((flags & kDenseLearningUsesPre) != 0u);
            has_dense_post_learning_ =
                has_dense_post_learning_ || ((flags & kDenseLearningUsesPost) != 0u);
        }
        if (index < layout.synapses.trigger_rule_id.size() &&
            index < layout.synapses.trigger_flags.size() &&
            layout.synapses.trigger_rule_id[index] >= 0) {
            const unsigned char flags = layout.synapses.trigger_flags[index];
            has_dense_trigger_learning_ =
                has_dense_trigger_learning_ || ((flags & kDenseLearningUsesTrigger) != 0u);
        }
    }
    has_dense_learning_ =
        has_dense_pre_learning_ || has_dense_post_learning_ || has_dense_trigger_learning_;
}

bool GpuPropagationRuntime::IsInitialized() const {
    return initialized_;
}

bool GpuPropagationRuntime::ResetState(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
    profiling_ = ProfilingStats{};
    current_time_step_ = 0;
    std::fill(pending_.pending_channels.begin(), pending_.pending_channels.end(), 0.0f);
    std::fill(learning_.syn_pending_dwt.begin(), learning_.syn_pending_dwt.end(), 0.0f);
    pending_host_view_dirty_ = false;
#if NPGR_ENABLE_CUDA
    if (!ResetDeviceState(reason)) {
        return false;
    }
#endif
    return true;
}

bool GpuPropagationRuntime::BeginStep(int time_step, std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
    if (time_step < 0) {
        if (reason != nullptr) {
            *reason = "time_step must be non-negative";
        }
        return false;
    }
    current_time_step_ = time_step;
    return true;
}

bool GpuPropagationRuntime::RunPropagationOnly(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    return RunGpuIsetPropagation(reason);
#else
    if (reason != nullptr) {
        *reason = "RunPropagationOnly requires CUDA support in the current dense backend";
    }
    return false;
#endif
}

bool GpuPropagationRuntime::ApplyPostLearningFromDeviceFirings(const unsigned char* d_current_did_fire,
                                                               std::string* reason) {
    // Post learning is called immediately after neuron update. The propagation
    // runtime owns synapse weights and learning metadata, while the neuron
    // runtime provides the per-neuron did-fire device array.
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        return this->ApplyPostLearningFromDeviceFiringsGpu(d_current_did_fire, reason);
    }
#endif
    (void)d_current_did_fire;
    if (reason != nullptr) {
        *reason = "dense post learning requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::SetInterfaceTargetNeurons(const std::vector<int>& interface_target_neuron_ids,
                                                      std::string* reason) {
    for (std::size_t index = 0; index < interface_target_neuron_ids.size(); ++index) {
        const int neuron_id = interface_target_neuron_ids[index];
        if (neuron_id < 0 || neuron_id >= neuron_count_) {
            if (reason != nullptr) {
                *reason = "interface target neuron id is out of range";
            }
            return false;
        }
    }
    interface_target_neuron_ids_ = interface_target_neuron_ids;
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        const std::size_t count = interface_target_neuron_ids_.size();
        if (count == 0) {
            return true;
        }
        if (count > device_interface_target_capacity_) {
            if (d_interface_target_neuron_ids_ != nullptr) {
                cudaFree(d_interface_target_neuron_ids_);
                d_interface_target_neuron_ids_ = nullptr;
                device_interface_target_capacity_ = 0;
            }
            const cudaError_t alloc_status =
                cudaMalloc(reinterpret_cast<void**>(&d_interface_target_neuron_ids_),
                           sizeof(int) * count);
            if (alloc_status != cudaSuccess) {
                if (reason != nullptr) {
                    std::ostringstream oss;
                    oss << "cudaMalloc(d_interface_target_neuron_ids_) failed: "
                        << cudaGetErrorString(alloc_status);
                    *reason = oss.str();
                }
                return false;
            }
            device_interface_target_capacity_ = count;
        }
        // upload the interface target neuron ids to the GPU
        const cudaError_t copy_status =
            cudaMemcpy(d_interface_target_neuron_ids_,
                       interface_target_neuron_ids_.data(),
                       sizeof(int) * count,
                       cudaMemcpyHostToDevice);
        if (copy_status != cudaSuccess) {
            if (reason != nullptr) {
                std::ostringstream oss;
                oss << "cudaMemcpy(interface target neuron ids) failed: "
                    << cudaGetErrorString(copy_status);
                *reason = oss.str();
            }
            return false;
        }
    }
#else
    (void)reason;
#endif
    return true;
}

bool GpuPropagationRuntime::UploadInterfaceChannelValues(const std::vector<InterfaceChannelValue>& values,
                                                         std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    // upload the interface channel values to the GPU
    if (device_ready_) {
        return this->UploadInterfaceChannelValuesGpu(values, reason);
    }
#endif
    (void)values;
    if (reason != nullptr) {
        *reason = "interface channel upload requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::UploadInterfaceChannelValuesDevice(
    const float* device_values,
    const std::vector<DeviceToDeviceInterfaceCopySpec>& specs,
    std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        return this->UploadInterfaceChannelValuesDeviceGpu(device_values, specs, reason);
    }
#endif
    (void)device_values;
    (void)specs;
    if (reason != nullptr) {
        *reason = "device interface channel upload requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::ApplyInterfaceStateVectors(std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        return this->ApplyInterfaceStateVectorsGpu(reason);
    }
#endif
    if (reason != nullptr) {
        *reason = "interface state application requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::DownloadSynapticWeights(std::vector<float>* weights,
                                                    std::string* reason) const {
    if (weights == nullptr) {
        if (reason != nullptr) {
            *reason = "weight output vector must not be null";
        }
        return false;
    }
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        // SaveWeight must observe learning updates, so read the live device
        // weight buffer instead of any host-side construction-time layout.
        return this->DownloadSynapticWeightsGpu(weights, reason);
    }
#endif
    if (reason != nullptr) {
        *reason = "synaptic weight download requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::UploadSynapticWeights(const std::vector<float>& weights,
                                                  std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        // LoadWeight writes directly into the live device weight buffer used by
        // propagation and dense learning kernels.
        return this->UploadSynapticWeightsGpu(weights, reason);
    }
#endif
    if (reason != nullptr) {
        *reason = "synaptic weight upload requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::GetSynapseWeight(int synapse_index,
                                             float* weight,
                                             std::string* reason) const {
    if (weight == nullptr) {
        if (reason != nullptr) {
            *reason = "weight output pointer must not be null";
        }
        return false;
    }
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
    if (synapse_index < 0 || synapse_index >= synapse_count_) {
        if (reason != nullptr) {
            *reason = "synapse index is out of range";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        return this->GetSynapseWeightGpu(synapse_index, weight, reason);
    }
#endif
    if (reason != nullptr) {
        *reason = "single synaptic weight read requires CUDA device buffers";
    }
    return false;
}

bool GpuPropagationRuntime::SetSynapseWeight(int synapse_index,
                                             float weight,
                                             std::string* reason) {
    if (!initialized_) {
        if (reason != nullptr) {
            *reason = "runtime is not initialized";
        }
        return false;
    }
    if (synapse_index < 0 || synapse_index >= synapse_count_) {
        if (reason != nullptr) {
            *reason = "synapse index is out of range";
        }
        return false;
    }
#if NPGR_ENABLE_CUDA
    if (device_ready_) {
        return this->SetSynapseWeightGpu(synapse_index, weight, reason);
    }
#endif
    if (reason != nullptr) {
        *reason = "single synaptic weight write requires CUDA device buffers";
    }
    return false;
}

const RuntimeConfig& GpuPropagationRuntime::config() const {
    return config_;
}

const ProfilingStats& GpuPropagationRuntime::profiling() const {
    return profiling_;
}

const PendingBuffersHostView& GpuPropagationRuntime::pending() const {
    this->SyncPendingHostView(nullptr);
    return pending_;
}

const LearningBuffersHostView& GpuPropagationRuntime::learning() const {
    return learning_;
}

bool GpuPropagationRuntime::IsCudaEnabledBuild() const {
#if NPGR_ENABLE_CUDA
    return true;
#else
    return false;
#endif
}

bool GpuPropagationRuntime::IsDeviceReady() const {
    return device_ready_;
}

bool GpuPropagationRuntime::IsCarlsimLikeGpuPathActive() const {
    return this->IsCudaEnabledBuild() && device_ready_;
}

bool GpuPropagationRuntime::SyncPendingHostView(std::string* reason) const {
#if NPGR_ENABLE_CUDA
    if (!device_ready_ || !pending_host_view_dirty_) {
        return true;
    }
    if (d_pending_channels_ != nullptr && !pending_.pending_channels.empty()) {
        if (cudaMemcpy(pending_.pending_channels.data(),
                       d_pending_channels_,
                       sizeof(float) * pending_.pending_channels.size(),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            if (reason != nullptr) {
                *reason = "cudaMemcpy(pending_channels -> host) failed";
            }
            return false;
        }
    }
    pending_host_view_dirty_ = false;
#else
    (void)reason;
#endif
    return true;
}

bool GpuPropagationRuntime::ExportFiringHistoryRing(std::vector<FiringStepView>* history, std::string* reason) const {
    if (history == nullptr) {
        if (reason != nullptr) {
            *reason = "history output must not be null";
        }
        return false;
    }
    history->clear();
#if NPGR_ENABLE_CUDA
    if (!device_ready_ || d_firing_history_time_steps_ == nullptr || d_firing_history_counts_ == nullptr || d_firing_history_ids_ == nullptr) {
        if (reason != nullptr) {
            *reason = "CARLsim-like GPU firing history buffers are not ready";
        }
        return false;
    }
    std::vector<int> host_time_steps(static_cast<std::size_t>(config_.steps_to_keep), -1);
    std::vector<int> host_counts(static_cast<std::size_t>(config_.steps_to_keep), 0);
    const int slot_capacity = std::max(1, neuron_count_);
    std::vector<int> host_ids(static_cast<std::size_t>(config_.steps_to_keep) * static_cast<std::size_t>(slot_capacity), -1);
    if (cudaMemcpy(host_time_steps.data(),
                   d_firing_history_time_steps_,
                   sizeof(int) * host_time_steps.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        if (reason != nullptr) {
            *reason = "cudaMemcpy(firing_history_time_steps -> host) failed";
        }
        return false;
    }
    if (cudaMemcpy(host_counts.data(),
                   d_firing_history_counts_,
                   sizeof(int) * host_counts.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        if (reason != nullptr) {
            *reason = "cudaMemcpy(firing_history_counts -> host) failed";
        }
        return false;
    }
    if (cudaMemcpy(host_ids.data(),
                   d_firing_history_ids_,
                   sizeof(int) * host_ids.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        if (reason != nullptr) {
            *reason = "cudaMemcpy(firing_history_ids -> host) failed";
        }
        return false;
    }
    history->assign(static_cast<std::size_t>(config_.steps_to_keep), FiringStepView{});
    for (int slot = 0; slot < config_.steps_to_keep; ++slot) {
        FiringStepView entry;
        entry.time_step = host_time_steps[static_cast<std::size_t>(slot)];
        const int count = std::max(0, std::min(slot_capacity, host_counts[static_cast<std::size_t>(slot)]));
        if (entry.time_step >= 0 && count > 0) {
            const int base = slot * slot_capacity;
            entry.firing_ids.assign(host_ids.begin() + base, host_ids.begin() + base + count);
        }
        (*history)[static_cast<std::size_t>(slot)] = entry;
    }
    return true;
#else
    if (reason != nullptr) {
        *reason = "firing history export requires CUDA support";
    }
    return false;
#endif
}

bool GpuPropagationRuntime::DownloadPendingChannels(std::vector<float>* pending_channels,
                                                    int* channel_count,
                                                    int* channel_stride,
                                                    std::string* reason) const {
    if (pending_channels == nullptr) {
        if (reason != nullptr) {
            *reason = "pending channel output must not be null";
        }
        return false;
    }
    // Monitor capture should pay the device-to-host copy cost only for steps
    // that request pending-channel records.
    if (!this->SyncPendingHostView(reason)) {
        return false;
    }
    *pending_channels = pending_.pending_channels;
    if (channel_count != nullptr) {
        *channel_count = pending_channel_count_;
    }
    if (channel_stride != nullptr) {
        *channel_stride = pending_channel_stride_;
    }
    return true;
}

DeviceCommonBuffersView GpuPropagationRuntime::GetDeviceCommonBuffersView() const {
    DeviceCommonBuffersView view;
    view.d_pending_channels = d_pending_channels_;
    view.pending_channel_count = pending_channel_count_;
    view.pending_channel_stride = pending_channel_stride_;
    view.neuron_count = neuron_count_;
    return view;
}

void GpuPropagationRuntime::ClearLearning() {
    std::fill(learning_.syn_pending_dwt.begin(), learning_.syn_pending_dwt.end(), 0.0f);
}

#if !NPGR_ENABLE_CUDA
bool GpuPropagationRuntime::InitializeDeviceBuffers(const GpuPropagationLayout&, std::string*) {
    device_ready_ = false;
    return true;
}

void GpuPropagationRuntime::DestroyDeviceBuffers() {
    device_ready_ = false;
}

bool GpuPropagationRuntime::RunGpuIsetPropagation(std::string* reason) {
    if (reason != nullptr) {
        *reason = "runtime was built without CUDA support";
    }
    return false;
}

bool GpuPropagationRuntime::ApplyPostLearningFromDeviceFiringsGpu(const unsigned char*, std::string* reason) {
    if (reason != nullptr) {
        *reason = "runtime was built without CUDA support";
    }
    return false;
}
#endif

}  // namespace npgr
