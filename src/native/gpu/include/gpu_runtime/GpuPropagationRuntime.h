/*
 * GpuPropagationRuntime.h
 *
 * Low-level propagation runtime for dense subnetworks. It stores the packed
 * synapse layout, device buffers, firing history, pending conductance/current
 * buffers, and profiling counters used by the higher-level dense runtime.
 */
#ifndef NPGR_GPU_PROPAGATION_RUNTIME_H
#define NPGR_GPU_PROPAGATION_RUNTIME_H

#include "dense_subnetwork/DenseNeuronDeviceViews.h"
#include "gpu_runtime/GpuLaunchConfig.h"
#include "gpu_runtime/GpuPropagationLayout.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace npgr {

struct RuntimeConfig {
    // Number of recent steps retained in debug firing history.
    int steps_to_keep = 0;
    // Enables buffers for pending synaptic learning increments.
    bool enable_learning_buffers = false;
    // Dense runtime timestep in milliseconds.
    float dt_ms = 1.0f;
};

struct ProfilingStats {
    // Time spent updating dense neuron state.
    double neuron_update_ms = 0.0;
    // Time spent propagating synaptic slices.
    double slice_propagation_ms = 0.0;
    // Time spent flushing pending input buffers into neuron state.
    double pending_flush_ms = 0.0;
    // Time spent applying learning updates.
    double learning_apply_ms = 0.0;
};

struct FiringStepView {
    // Simulation timestep represented by this view.
    int time_step = 0;
    // Neuron IDs that fired at time_step.
    std::vector<int> firing_ids;
};

struct PendingBuffersHostView {
    // Channel-major mirror flattened as pending_channels[channel * runtime_stride + neuron].
    std::vector<float> pending_channels;
};

struct InterfaceChannelValue {
    int interface_slot = -1;
    int target_neuron = -1;
    std::uint8_t pending_channel = static_cast<std::uint8_t>(PendingChannel::ExcitatoryConductance);
    float value = 0.0f;
    float scale = 1.0f;
    // Current inputs are stateful and replace the destination channel for the
    // step; spike/conductance inputs are event-like and accumulate.
    unsigned char overwrite = 0;
};

struct DeviceToDeviceInterfaceCopySpec {
    int interface_slot = -1;
    int target_neuron = -1;
    std::uint8_t pending_channel = static_cast<std::uint8_t>(PendingChannel::Current);
    int source_index = -1;
    float scale = 1.0f;
    bool overwrite = false;
};

struct LearningBuffersHostView {
    // Host mirror of pending weight increments.
    std::vector<float> syn_pending_dwt;
};

class GpuPropagationRuntime {
public:
    GpuPropagationRuntime();
    ~GpuPropagationRuntime();

    // Validates the packed layout and allocates host/device buffers.
    bool Initialize(const GpuPropagationLayout& layout, const RuntimeConfig& config, std::string* reason = nullptr);
    // Returns true after successful initialization.
    bool IsInitialized() const;
    // Clears runtime state while keeping the already allocated layout/device buffers.
    bool ResetState(std::string* reason = nullptr);

    // Starts a new propagation step and resets current-step propagation state.
    bool BeginStep(int time_step, std::string* reason = nullptr);
    // Runs only spike propagation, leaving neuron update to the caller.
    bool RunPropagationOnly(std::string* reason = nullptr);
    // Commits firings produced by an external device buffer.
    bool CommitCurrentDeviceFiringsFromExternal(int time_step,
                                                const int* d_external_firing_ids,
                                                const int* d_external_firing_count,
                                                std::string* reason = nullptr);
    // Applies dense post-learning hooks after the neuron runtime has produced
    // the per-neuron current-step did-fire flags.
    bool ApplyPostLearningFromDeviceFirings(const unsigned char* d_current_did_fire,
                                            std::string* reason = nullptr);
    // Maps interface slots to dense target neurons.
    bool SetInterfaceTargetNeurons(const std::vector<int>& interface_target_neuron_ids, std::string* reason = nullptr);
    bool UploadInterfaceChannelValues(const std::vector<InterfaceChannelValue>& values,
                                      std::string* reason = nullptr);
    bool UploadInterfaceChannelValuesDevice(const float* device_values,
                                            const std::vector<DeviceToDeviceInterfaceCopySpec>& specs,
                                            std::string* reason = nullptr);
    // Applies uploaded interface state to pending neuron buffers.
    bool ApplyInterfaceStateVectors(std::string* reason = nullptr);

    // Runtime configuration.
    const RuntimeConfig& config() const;
    // Profiling counters for the most recent operations.
    const ProfilingStats& profiling() const;
    // Host mirror of pending input buffers.
    const PendingBuffersHostView& pending() const;
    // Host mirror of learning buffers.
    const LearningBuffersHostView& learning() const;
    // Returns true when the binary was built with CUDA support.
    bool IsCudaEnabledBuild() const;
    // Returns true when device buffers are allocated and ready.
    bool IsDeviceReady() const;
    // Returns true when the CARLsim-like GPU path is active.
    bool IsCarlsimLikeGpuPathActive() const;
    // Exports the retained firing history ring to host memory.
    bool ExportFiringHistoryRing(std::vector<FiringStepView>* history, std::string* reason = nullptr) const;
    // Downloads the live pending-channel pool only when a monitor explicitly requests it.
    bool DownloadPendingChannels(std::vector<float>* pending_channels,
                                 int* channel_count,
                                 int* channel_stride,
                                 std::string* reason = nullptr) const;
    // Downloads the live synaptic weights from d_syn_weight_.
    bool DownloadSynapticWeights(std::vector<float>* weights, std::string* reason = nullptr) const;
    // Uploads live synaptic weights into d_syn_weight_.
    bool UploadSynapticWeights(const std::vector<float>& weights, std::string* reason = nullptr);
    // Reads one live synaptic weight by dense runtime synapse index.
    bool GetSynapseWeight(int synapse_index, float* weight, std::string* reason = nullptr) const;
    // Writes one live synaptic weight by dense runtime synapse index.
    bool SetSynapseWeight(int synapse_index, float weight, std::string* reason = nullptr);
    // Returns raw device-buffer pointers for CUDA neuron runtimes.
    DeviceCommonBuffersView GetDeviceCommonBuffersView() const;

private:
    // Refreshes host pending-buffer mirrors from device memory when needed.
    bool SyncPendingHostView(std::string* reason) const;
    // Dense-local neuron count retained after the packed layout is uploaded.
    int neuron_count_;
    // Synapse count retained for Iset sizing and empty-propagation fast paths.
    int synapse_count_;
    // Delay slot count retained for firing-history scan launch geometry.
    int delay_slot_count_;
    // Runtime behavior shared by propagation and dense neuron update.
    RuntimeConfig config_;
    // Accumulated profiling counters.
    ProfilingStats profiling_;
    // Host debug/export mirrors only. Device runtime state is authoritative
    // whenever the CUDA path is active.
    mutable PendingBuffersHostView pending_;
    // Host mirror of learning buffers; retained for debug/export plumbing.
    LearningBuffersHostView learning_;

    // Whether Initialize completed successfully.
    bool initialized_;
    // Current simulation step owned by this runtime.
    int current_time_step_;
    // Dense target neuron IDs for each main-network interface slot.
    std::vector<int> interface_target_neuron_ids_;
    // Host CSR reverse lookup from post neuron to incoming synapse IDs.
    // The unified Iset drain kernel uses the device copy of this table.
    std::vector<int> post_synapse_start_;
    std::vector<int> post_synapse_count_;
    std::vector<int> post_synapse_ids_;

    // Allocates all CUDA device buffers.
    bool InitializeDeviceBuffers(const GpuPropagationLayout& layout, std::string* reason);
    // Releases all CUDA device buffers.
    void DestroyDeviceBuffers();
    // Runs the unified GPU propagation path: firing history -> Iset -> pending channels.
    bool RunGpuIsetPropagation(std::string* reason);
    bool ApplyPostLearningFromDeviceFiringsGpu(const unsigned char* d_current_did_fire,
                                               std::string* reason);
    // Clears CUDA-side mutable state while retaining immutable layout buffers.
    bool ResetDeviceState(std::string* reason);
    bool UploadInterfaceChannelValuesGpu(const std::vector<InterfaceChannelValue>& values,
                                         std::string* reason);
    bool UploadInterfaceChannelValuesDeviceGpu(const float* device_values,
                                               const std::vector<DeviceToDeviceInterfaceCopySpec>& specs,
                                               std::string* reason);
    // Applies uploaded interface state on the GPU path.
    bool ApplyInterfaceStateVectorsGpu(std::string* reason);
    // CUDA implementation for live dense synaptic weight export/import.
    bool DownloadSynapticWeightsGpu(std::vector<float>* weights, std::string* reason) const;
    bool UploadSynapticWeightsGpu(const std::vector<float>& weights, std::string* reason);
    bool GetSynapseWeightGpu(int synapse_index, float* weight, std::string* reason) const;
    bool SetSynapseWeightGpu(int synapse_index, float weight, std::string* reason);
    // Clears pending learning buffers.
    void ClearLearning();
    // Computes whether pre/post/trigger learning kernels are needed for this layout.
    void BuildLearningFeatureFlags(const GpuPropagationLayout& layout);
    // Caches CUDA launch geometry that depends only on the initialized layout/device.
    void BuildCachedLaunchConfigs();
    // Builds launch geometry from cached max block counts without querying device properties.
    LaunchConfig1D MakeRuntimeLaunchConfig(int item_count, int threads, int max_blocks) const;
    // True after CUDA buffers are successfully allocated.
    bool device_ready_;
    // CUDA SM count cached during device-buffer initialization.
    int device_sm_count_;
    // Cached maximum block count for linear scan kernels.
    int linear_scan_max_blocks_;
    // Cached maximum block count for history scan kernels.
    int history_scan_max_blocks_;
    // Cached launch for clearing the Iset bitmap.
    LaunchConfig1D launch_clear_iset_;
    // Cached launch for draining marked Iset bits into pending channels.
    LaunchConfig1D launch_drain_iset_;
    // Cached launch for synapse-parallel pre/trigger learning kernels.
    LaunchConfig1D launch_learning_synapse_;
    // Cached launch for synapse-parallel post learning.
    LaunchConfig1D launch_post_learning_;
    // Cached X grid dimension for firing-history scans.
    int history_blocks_x_;
    // True when any dense learning hook exists.
    bool has_dense_learning_;
    // True when pre-synaptic learning hooks exist.
    bool has_dense_pre_learning_;
    // True when post-synaptic learning hooks exist.
    bool has_dense_post_learning_;
    // True when trigger learning hooks exist.
    bool has_dense_trigger_learning_;
    // Device copy of pre-delay slices, flattened as [pre_neuron, delay_slot] -> synapse range.
    int* d_pre_slice_synapse_ids_;
    // Device copy of each pre-delay slice start offset into d_pre_slice_synapse_ids_.
    int* d_pre_slice_start_;
    // Device copy of each pre-delay slice synapse count.
    int* d_pre_slice_count_;
    // Per-synapse dense-local post neuron id used by post-learning hooks.
    int* d_syn_post_neuron_;
    // Per-synapse base weight used when a marked Iset bit is drained to pending channels.
    float* d_syn_weight_;
    // Per-synapse legacy synapse type, used by learning rules such as R-STDP triggers.
    unsigned char* d_syn_type_;
    // Per-synapse pending channel selected by the source neuron's spike effect description.
    unsigned char* d_syn_effect_channel_;
    // Per-synapse multiplier applied after d_syn_weight_ before accumulation.
    float* d_syn_effect_scale_;
    // Plastic-side rule id derived from legacy SynapseRule; -1 disables plastic learning.
    int* d_syn_plastic_rule_id_;
    // Plastic-side dense model id used for pre/post device dispatch.
    int* d_syn_plastic_model_id_;
    // Plastic-side DenseLearningFlags mask used by pre/post hooks.
    unsigned char* d_syn_plastic_flags_;
    // Plastic-side state index. It currently equals synapse id because fields are synapse-flat.
    int* d_syn_plastic_state_index_;
    // Trigger-side rule id derived from legacy TriggerRule; -1 disables trigger learning.
    int* d_syn_trigger_rule_id_;
    // Trigger-side dense model id used for trigger device dispatch.
    int* d_syn_trigger_model_id_;
    // Trigger-side DenseLearningFlags mask used by trigger hooks.
    unsigned char* d_syn_trigger_flags_;
    // Generic learning field spans uploaded from DenseLearningRuleFactory.
    DenseDeviceFieldSpan* d_learning_field_spans_;
    // Live generic learning float fields.
    float* d_learning_float_pool_;
    // Live generic learning int fields.
    int* d_learning_int_pool_;
    // Live generic learning byte fields.
    unsigned char* d_learning_byte_pool_;
    // Reset copy for generic learning float fields, copied device-to-device on ResetState.
    float* d_learning_reset_float_pool_;
    // Reset copy for generic learning int fields, copied device-to-device on ResetState.
    int* d_learning_reset_int_pool_;
    // Reset copy for generic learning byte fields, copied device-to-device on ResetState.
    unsigned char* d_learning_reset_byte_pool_;
    // Device view passed to learning kernels; points at the live field pools above.
    DenseLearningDeviceFieldTable d_learning_fields_;
    // Generic learning model spans uploaded from DenseLearningRuleFactory.
    DenseLearningModelSpan* d_learning_model_spans_;
    // Concatenated model-local field ids used by device learning dispatch.
    int* d_learning_field_indices_;
    // Device view passed to learning kernels for model-local field lookup.
    DenseDeviceLearningModelFieldIndexTable d_learning_model_indices_;
    // Trigger synapse -> ordinary plastic synapse CSR route table.
    DenseDeviceLearningTriggerRouteTable d_learning_trigger_routes_;
    // Buffered ordinary plastic pre-spike history used by Cerebellar trigger learning.
    DenseDeviceLearningSpikeBufferTable d_learning_spike_buffers_;
    // Device CSR starts for trigger routes, indexed by synapse id.
    int* d_learning_trigger_route_start_;
    // Device CSR counts for trigger routes, indexed by synapse id.
    int* d_learning_trigger_route_count_;
    // Device CSR payload listing plastic synapses affected by trigger synapses.
    int* d_learning_trigger_route_synapse_ids_;
    // Per-synapse Cerebellar spike-buffer bucket id.
    int* d_learning_spike_buffer_synapse_bucket_id_;
    // Per-bucket ring start offset.
    int* d_learning_spike_buffer_bucket_start_;
    // Per-bucket ring capacity.
    int* d_learning_spike_buffer_bucket_capacity_;
    // Per-bucket ring write head.
    int* d_learning_spike_buffer_bucket_head_;
    // Per-bucket live entry count.
    int* d_learning_spike_buffer_bucket_count_;
    // Buffered spike event times.
    float* d_learning_spike_buffer_time_;
    // Buffered plastic synapse ids.
    int* d_learning_spike_buffer_synapse_id_;
    // Buffered entry validity flags.
    unsigned char* d_learning_spike_buffer_valid_;
    // Number of float elements in the generic learning field pool.
    std::size_t device_learning_float_pool_size_;
    // Number of int elements in the generic learning field pool.
    std::size_t device_learning_int_pool_size_;
    // Number of byte elements in the generic learning field pool.
    std::size_t device_learning_byte_pool_size_;
    // Channel-major pending input pool: pending[channel * stride + neuron].
    // Unified propagation, host interface records, and D2D interface records all accumulate here.
    float* d_pending_channels_;
    // Number of pending channels allocated, equal to the max channel count required by all models.
    int pending_channel_count_;
    // Distance between two consecutive channel planes; currently the dense neuron count.
    int pending_channel_stride_;
    // Ring metadata: simulation time step stored in each firing-history slot.
    int* d_firing_history_time_steps_;
    // Ring metadata: number of firing neuron IDs stored in each slot.
    int* d_firing_history_counts_;
    // Ring payload: firing IDs flattened as [history_slot * neuron_capacity + local_index].
    int* d_firing_history_ids_;
    // Device CSR reverse lookup from post neuron to incoming synapses.
    int* d_post_synapse_start_;
    // Device CSR incoming synapse count per post neuron.
    int* d_post_synapse_count_;
    // Device CSR incoming synapse IDs consumed by DrainIsetToPendingKernel.
    int* d_post_synapse_ids_;
    // Bitset over synapse IDs marked by direct firing-history scan for the current step.
    unsigned int* d_iset_words_;
    // Device mapping from interface slot to dense target neuron id.
    int* d_interface_target_neuron_ids_;
    // Device staging buffer for the fixed per-slot host-uploaded interface record table.
    InterfaceChannelValue* d_interface_channel_values_;
    // Device staging buffer for the fixed per-slot device-to-device interface copy table.
    DeviceToDeviceInterfaceCopySpec* d_d2d_copy_specs_;
    // Number of currently staged device-to-device interface copy specs.
    int d_d2d_copy_count_;
    // Allocated capacity of the flattened firing-history ID ring.
    std::size_t device_firing_history_capacity_;
    // Allocated capacity of d_post_synapse_ids_.
    std::size_t device_post_synapse_capacity_;
    // Number of 32-bit words allocated in d_iset_words_.
    std::size_t device_iset_word_capacity_;
    // Allocated capacity of d_interface_target_neuron_ids_.
    std::size_t device_interface_target_capacity_;
    // Allocated capacity of d_interface_channel_values_.
    std::size_t device_interface_state_capacity_;
    // Number of currently staged host-uploaded per-slot interface records.
    int device_interface_state_count_;
    // Allocated capacity of d_d2d_copy_specs_.
    std::size_t device_d2d_copy_capacity_;
    // True when pending_.pending_channels is stale relative to d_pending_channels_.
    mutable bool pending_host_view_dirty_;
};

}  // namespace npgr

#endif
